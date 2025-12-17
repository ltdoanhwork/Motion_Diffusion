#!/usr/bin/env python3
"""
Multi-GPU Training Script for Motion Diffusion Model
=====================================================
Distributed Data Parallel (DDP) training with:
- 4x A100 80GB GPU support
- EMA (Exponential Moving Average) model weights
- Mixed precision training (AMP)
- Gradient accumulation
- Motion-specific losses
- Cosine beta schedule
- Classifier-Free Guidance training

Usage:
    torchrun --nproc_per_node=4 tools/train_ddp.py --dataset_name beat --num_epochs 300

Author: Motion Diffusion Improvement Project
"""

import os
import sys
import time
import random
import numpy as np
from collections import OrderedDict
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tqdm import tqdm
from options.train_options import TrainCompOptions
from models.transformer import MotionTransformer
from models.motion_losses import MotionLossModule, create_motion_loss
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
from datasets import Beat2MotionDataset
from datasets.dataset import Text2MotionDataset
import utils.paramUtil as paramUtil


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model weights that is updated as:
        shadow = decay * shadow + (1 - decay) * model_params
    
    This provides smoother model weights for inference.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


class DDPMTrainerDDP:
    """
    Distributed DDPM Trainer for Motion Diffusion.
    
    Features:
    - Multi-GPU training with DDP
    - EMA for stable inference
    - Motion-specific losses
    - Mixed precision training
    - Gradient accumulation
    - Classifier-free guidance training
    """
    
    def __init__(self, args, encoder, rank, world_size):
        self.opt = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Move encoder to device
        self.encoder = encoder.to(self.device)
        
        # Wrap with DDP
        self.encoder = DDP(
            self.encoder, 
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=False
        )
        
        # Diffusion setup
        self.diffusion_steps = args.diffusion_steps
        beta_schedule = getattr(args, 'beta_schedule', 'cosine')
        betas = get_named_beta_schedule(beta_schedule, self.diffusion_steps)
        
        # Determine model mean type based on prediction type
        prediction_type = getattr(args, 'prediction_type', 'epsilon')
        if prediction_type == 'epsilon':
            model_mean_type = ModelMeanType.EPSILON
        elif prediction_type == 'x_start':
            model_mean_type = ModelMeanType.START_X
        else:  # v_prediction - we'll handle this specially
            model_mean_type = ModelMeanType.EPSILON  # Fall back to epsilon
        
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        
        self.sampler = create_named_schedule_sampler('uniform', self.diffusion)
        
        # Motion loss module
        use_velocity = getattr(args, 'use_velocity_loss', True)
        use_acceleration = getattr(args, 'use_acceleration_loss', False)
        use_geometric = getattr(args, 'use_geometric_loss', True)
        
        if use_velocity or use_acceleration or use_geometric:
            self.motion_loss = MotionLossModule(
                joints_num=55,  # BEAT skeleton
                fps=60.0,
                use_velocity=use_velocity,
                use_acceleration=use_acceleration,
                use_bone_length=use_geometric,
                use_foot_contact=False,
                velocity_weight=getattr(args, 'velocity_weight', 0.5),
                acceleration_weight=getattr(args, 'acceleration_weight', 0.1),
                bone_length_weight=getattr(args, 'geometric_weight', 0.3),
            ).to(self.device)
        else:
            self.motion_loss = None
        
        # EMA
        ema_decay = getattr(args, 'ema_decay', 0.9999)
        if ema_decay > 0:
            self.ema = EMA(self.encoder.module, decay=ema_decay)
        else:
            self.ema = None
        
        # Mixed precision
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = getattr(args, 'gradient_accumulation_steps', 1)
        
        # CFG dropout
        self.cfg_dropout = getattr(args, 'cfg_dropout', 0.1)
        
        # Loss criterion
        self.mse_criterion = nn.MSELoss(reduction='none')
        
        # Max gradient norm
        self.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)
        
        # TensorBoard (only on rank 0)
        if rank == 0:
            log_dir = pjoin(args.save_root, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def forward(self, batch_data, eval_mode=False):
        """Forward pass through diffusion model."""
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()
        
        # CFG: randomly drop text condition during training
        if self.training and not eval_mode and self.cfg_dropout > 0:
            # Replace some captions with empty string
            caption = list(caption)
            for i in range(len(caption)):
                if random.random() < self.cfg_dropout:
                    caption[i] = ""
            caption = tuple(caption)
        
        self.caption = caption
        self.motions = motions
        x_start = motions
        B, T = x_start.shape[:2]
        
        cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)
        
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )
        
        self.real_noise = output['target']
        self.fake_noise = output['pred']
        self.timesteps = t
        
        # Predicted clean motion for motion losses
        if self.motion_loss is not None:
            with torch.no_grad():
                # Predict x_0 from model output
                pred_x0 = self.diffusion._predict_xstart_from_eps(
                    x_t=self.diffusion.q_sample(x_start, t),
                    t=t,
                    eps=self.fake_noise
                )
            self.pred_motion = pred_x0
            self.gt_motion = motions
        
        # Generate source mask
        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)
    
    def backward_G(self):
        """Compute all losses."""
        # Base diffusion loss (MSE on noise prediction)
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        
        loss_logs = OrderedDict()
        loss_logs['loss_diffusion'] = loss_mot_rec.item()
        
        total_loss = loss_mot_rec
        
        # Motion-specific losses
        if self.motion_loss is not None:
            motion_losses = self.motion_loss(
                self.pred_motion, 
                self.gt_motion,
                mask=self.src_mask
            )
            
            if 'velocity' in motion_losses:
                loss_logs['loss_velocity'] = motion_losses['velocity'].item()
            if 'acceleration' in motion_losses:
                loss_logs['loss_acceleration'] = motion_losses['acceleration'].item()
            if 'bone_length' in motion_losses:
                loss_logs['loss_geometric'] = motion_losses['bone_length'].item()
            
            total_loss = total_loss + motion_losses['total']
        
        self.total_loss = total_loss
        loss_logs['loss_total'] = total_loss.item()
        
        return loss_logs
    
    def train(self, train_dataset):
        """Main training loop with DDP."""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.encoder.parameters(),
            lr=self.opt.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Learning rate scheduler (cosine annealing)
        total_steps = self.opt.num_epochs * (len(train_dataset) // (self.opt.batch_size * self.world_size))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6
        )
        
        # Distributed sampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.opt.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        # Resume if needed
        start_epoch = 0
        global_step = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            if os.path.exists(model_dir):
                start_epoch, global_step = self.load(model_dir)
                if self.rank == 0:
                    print(f"Resumed from epoch {start_epoch}, step {global_step}")
        
        start_time = time.time()
        
        for epoch in range(start_epoch, self.opt.num_epochs):
            train_sampler.set_epoch(epoch)
            self.encoder.train()
            self.training = True
            
            epoch_logs = OrderedDict()
            
            if self.rank == 0:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            else:
                pbar = train_loader
            
            for batch_idx, batch_data in enumerate(pbar):
                if batch_data is None:
                    continue
                
                # Forward
                with autocast(enabled=self.use_amp):
                    self.forward(batch_data)
                    loss_logs = self.backward_G()
                
                # Scale loss for gradient accumulation
                loss = self.total_loss / self.grad_accum_steps
                
                # Backward
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Gradient step
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.encoder.parameters(), 
                            self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.encoder.parameters(), 
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    
                    # EMA update
                    if self.ema is not None:
                        self.ema.update()
                    
                    global_step += 1
                
                # Accumulate epoch logs
                for k, v in loss_logs.items():
                    epoch_logs[k] = epoch_logs.get(k, 0) + v
                
                # Log progress
                if self.rank == 0 and (batch_idx + 1) % self.opt.log_every == 0:
                    mean_logs = {k: v / self.opt.log_every for k, v in epoch_logs.items()}
                    epoch_logs = OrderedDict()
                    
                    # TensorBoard
                    for k, v in mean_logs.items():
                        self.writer.add_scalar(f'train/{k}', v, global_step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], global_step)
                    
                    # Progress bar
                    elapsed = time.time() - start_time
                    pbar.set_postfix({
                        'loss': f"{mean_logs.get('loss_total', 0):.4f}",
                        'time': f"{elapsed/60:.1f}m"
                    })
            
            # Epoch checkpointing
            if self.rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch + 1, global_step)
                
                if (epoch + 1) % self.opt.save_every_e == 0:
                    self.save(
                        pjoin(self.opt.model_dir, f'ckpt_e{epoch+1:03d}.tar'),
                        epoch + 1,
                        global_step
                    )
                    print(f"💾 Saved checkpoint at epoch {epoch + 1}")
        
        if self.rank == 0:
            self.writer.close()
            print("🚀 Training complete!")
    
    def save(self, file_name, epoch, global_step):
        """Save checkpoint."""
        state = {
            'encoder': self.encoder.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
        }
        
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        
        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()
        
        torch.save(state, file_name)
    
    def load(self, model_dir):
        """Load checkpoint."""
        checkpoint = torch.load(model_dir, map_location=self.device)
        
        self.encoder.module.load_state_dict(checkpoint['encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if self.ema is not None and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
        
        if self.scaler is not None and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        return checkpoint['epoch'], checkpoint.get('global_step', 0)


def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def build_model(opt, dim_pose):
    """Build MotionTransformer model."""
    return MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
    )


def main():
    # Parse arguments
    parser = TrainCompOptions()
    opt = parser.parse()
    
    # Get rank and world size from environment (set by torchrun)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Setup DDP
    if world_size > 1:
        setup_ddp(rank, world_size)
    
    # Set device
    opt.device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    
    # Set random seed for reproducibility
    seed = 42 + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if rank == 0:
        print(f"🚀 Starting DDP training with {world_size} GPUs")
        print(f"   Beta schedule: {getattr(opt, 'beta_schedule', 'cosine')}")
        print(f"   Velocity loss: {getattr(opt, 'use_velocity_loss', True)}")
        print(f"   Geometric loss: {getattr(opt, 'use_geometric_loss', True)}")
        print(f"   EMA decay: {getattr(opt, 'ema_decay', 0.9999)}")
    
    # Dataset config for BEAT
    if opt.dataset_name == 'beat':
        opt.data_root = pjoin(ROOT, 'datasets/BEAT_numpy')
        opt.motion_dir = pjoin(opt.data_root, 'npy')
        opt.text_dir = pjoin(opt.data_root, 'txt')
        opt.joints_num = 55
        fps = 60
        opt.max_motion_length = 360
        dim_pose = 264
        DatasetClass = Beat2MotionDataset
    elif opt.dataset_name == 't2m':
        opt.data_root = './datasets/HumanML3D/HumanML3D'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        fps = 20
        opt.max_motion_length = 196
        dim_pose = 263
        DatasetClass = Text2MotionDataset
    else:
        raise KeyError(f'Unknown dataset {opt.dataset_name}')
    
    # Paths
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    
    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    
    # Synchronize before loading data
    if world_size > 1:
        dist.barrier()
    
    # Load mean/std
    mean_path = pjoin(opt.meta_dir, 'mean.npy')
    std_path = pjoin(opt.meta_dir, 'std.npy')
    mean = np.load(mean_path) if os.path.exists(mean_path) else None
    std = np.load(std_path) if os.path.exists(std_path) else None
    
    # Build model
    encoder = build_model(opt, dim_pose)
    
    if rank == 0:
        total_params = sum(p.numel() for p in encoder.parameters())
        print(f"   Model parameters: {total_params / 1e6:.2f}M")
    
    # Build trainer
    trainer = DDPMTrainerDDP(opt, encoder, rank, world_size)
    
    # Load dataset
    train_split = pjoin(opt.data_root, 'train.txt')
    train_dataset = DatasetClass(opt, mean, std, train_split, opt.times)
    
    if rank == 0:
        print(f"   Dataset size: {len(train_dataset)}")
    
    # Train!
    trainer.train(train_dataset)
    
    # Cleanup
    if world_size > 1:
        cleanup_ddp()


if __name__ == '__main__':
    main()
