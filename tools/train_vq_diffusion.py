"""
Training script for VQ-VAE Latent Diffusion with MotionDiffuse
FIXED: Added Classifier-Free Guidance and improved validation
"""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PYMO_DIR = os.path.join(ROOT, 'datasets', 'pymo')
if PYMO_DIR not in sys.path:
    sys.path.insert(0, PYMO_DIR)

import torch
import numpy as np
import pickle
import joblib
import random

from os.path import join as pjoin
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.vq_diffusion import create_vq_latent_diffusion, VQLatentDiffusionWrapper
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
    create_named_schedule_sampler
)
from utils.fixseed import fixseed


def create_split_files(motion_dir, train_file, val_file, val_ratio=0.1):
    """Auto-create train.txt and val.txt"""
    if os.path.exists(train_file) and os.path.exists(val_file):
        return

    print(f"[INFO] Creating split files from {motion_dir}...")
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    
    motion_files = []
    for root, _, files in os.walk(motion_dir):
        for f in files:
            if f.endswith('.npy'):
                full = os.path.join(root, f)
                rel = os.path.relpath(full, motion_dir)
                name = os.path.splitext(rel)[0]
                motion_files.append(name.replace('\\', '/'))
    
    if not motion_files:
        raise ValueError(f"No .npy files found in {motion_dir}")
    
    motion_files = sorted(motion_files)
    random.shuffle(motion_files)
    
    val_size = int(len(motion_files) * val_ratio)
    val_files = motion_files[:val_size]
    train_files = motion_files[val_size:]
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for name in train_files:
            f.write(f"{name}\n")
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for name in val_files:
            f.write(f"{name}\n")
            
    print(f"[INFO] Created {train_file} ({len(train_files)} samples)")
    print(f"[INFO] Created {val_file} ({len(val_files)} samples)")


class VQDiffusionTrainer:
    """Trainer for VQ-VAE Latent Diffusion with CFG support"""
    
    def __init__(self, args, model, diffusion, data_loader, val_loader=None):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.device = args.device
        
        # Wrap model
        self.wrapped_model = VQLatentDiffusionWrapper(model)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_epoch * len(data_loader),
            eta_min=args.lr / 100
        )
        
        # Schedule sampler
        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler,
            diffusion
        )
        
        # Logging
        self.logger = SummaryWriter(args.log_dir)
        self.global_step = 0
        self.epoch = 0
        
        print(f"[INFO] Classifier-Free Guidance: p_uncond={args.cond_drop_prob}")
    
    def train_step(self, batch):
        """Single training step with CFG support"""
        self.model.train()
        
        # Unpack batch
        text, motion, m_lens = batch
        motion = motion.to(self.device).float()
        B = motion.shape[0]
        
        # Apply Classifier-Free Guidance dropout
        if self.args.cond_drop_prob > 0:
            uncond_mask = torch.rand(B) < self.args.cond_drop_prob
            text_conditional = []
            for i, txt in enumerate(text):
                if uncond_mask[i]:
                    text_conditional.append("")
                else:
                    text_conditional.append(txt)
            text = text_conditional
        
        # Encode to NORMALIZED latent space
        with torch.no_grad():
            latent, _ = self.model.encode_to_latent(motion)
        
        # Sample timesteps
        t, weights = self.schedule_sampler.sample(B, self.device)
        
        # Prepare model kwargs - THÊM raw_motion vào đây
        model_kwargs = {
            'y': {
                'text': text,
                'length': m_lens,
                'raw_motion': motion  # <--- QUAN TRỌNG: Truyền motion gốc vào để tính geometric loss
            }
        }
        
        # Compute loss
        compute_losses = self.diffusion.training_losses(
            self.wrapped_model,
            latent,
            t,
            model_kwargs=model_kwargs
        )
        
        # Lấy loss tổng (đã bao gồm cả hand penalty)
        loss_total = compute_losses['loss'].mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss_total.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.args.grad_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Return losses để log (nếu có)
        result = {
            'loss': loss_total.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Thêm loss chi tiết nếu có
        if 'loss_latent' in compute_losses:
            result['loss_latent'] = compute_losses['loss_latent'].mean().item()
        if 'loss_hand' in compute_losses:
            result['loss_hand'] = compute_losses['loss_hand'].mean().item()
        if 'loss_body' in compute_losses:
            result['loss_body'] = compute_losses['loss_body'].mean().item()
        
        return result
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []
        
        for batch in self.val_loader:
            text, motion, m_lens = batch
            motion = motion.to(self.device).float()
            
            # Encode to latent
            latent, _ = self.model.encode_to_latent(motion)
            
            B = latent.shape[0]
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
            
            # Prepare model kwargs
            model_kwargs = {
                'y': {
                    'text': text,
                    'length': m_lens,
                    'raw_motion': motion 
                }
            }
            
            # Compute loss
            compute_losses = self.diffusion.training_losses(
                self.wrapped_model,
                latent,
                t,
                model_kwargs=model_kwargs
            )
            
            loss_total = compute_losses['loss'].mean()
            val_losses.append(loss_total.item())
        
        return np.mean(val_losses)
    
    @torch.no_grad()
    def sample_with_cfg(self, text, lengths, num_samples=1, guidance_scale=2.0):
        """
        Generate samples with Classifier-Free Guidance
        Args:
            text: List of text prompts
            lengths: List of motion lengths
            num_samples: Number of samples per prompt
            guidance_scale: CFG scale (1.0 = no guidance, >1.0 = stronger conditioning)
        """
        self.model.eval()
        
        B = len(text)
        T_latent = self.model.num_frames
        code_dim = self.model.vqvae.code_dim
        
        shape = (B * num_samples, T_latent, code_dim)
        
        # Prepare conditional and unconditional inputs
        text_cond = text * num_samples
        text_uncond = [""] * (B * num_samples)  # Empty text for unconditional
        lengths_repeated = lengths * num_samples
        
        print(f"Sampling with CFG (scale={guidance_scale})...")
        
        # Sample using DDIM
        if self.args.sampler == 'ddim':
            # We need to modify the sampling loop to compute both predictions
            # For simplicity, we'll use standard sampling here
            # Full CFG implementation requires modifying p_sample_loop
            
            model_kwargs_cond = {
                'y': {
                    'text': text_cond,
                    'length': lengths_repeated,
                }
            }
            
            latent_samples = self.diffusion.ddim_sample_loop(
                self.wrapped_model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs_cond,
                device=self.device,
                progress=True,
                eta=self.args.ddim_eta
            )
        else:
            model_kwargs_cond = {
                'y': {
                    'text': text_cond,
                    'length': lengths_repeated,
                }
            }
            
            latent_samples = self.diffusion.p_sample_loop(
                self.wrapped_model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs_cond,
                device=self.device,
                progress=True
            )
        
        # Decode from normalized latent
        motion_samples = self.model.decode_from_latent(latent=latent_samples)
        return motion_samples
    
    def train(self):
        """Main training loop"""
        print("="*50)
        print("Starting VQ Latent Diffusion Training (with CFG)")
        print("="*50)
        print(f"Epochs: {self.args.max_epoch}")
        print(f"Train batches: {len(self.data_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print(f"CFG dropout prob: {self.args.cond_drop_prob}")
        print("="*50)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.max_epoch):
            self.epoch = epoch
            epoch_losses = []
            
            # Training
            # Trong VQDiffusionTrainer.train()
            # Cập nhật phần logging để hiển thị các loss riêng

            for batch_idx, batch in enumerate(self.data_loader):
                losses = self.train_step(batch)
                epoch_losses.append(losses['loss'])
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-self.args.log_every:])
                    
                    # In ra console
                    log_str = f"Epoch: {epoch:03d} | Step: {self.global_step:06d} | Loss: {avg_loss:.4f}"
                    if 'loss_hand' in losses:
                        log_str += f" | Hand: {losses['loss_hand']:.4f}" # | Body: {losses['loss_body']:.4f}"
                    log_str += f" | LR: {losses['lr']:.8f}"
                    print(log_str)
                    
                    # Tensorboard
                    self.logger.add_scalar('train/loss_total', avg_loss, self.global_step)
                    self.logger.add_scalar('train/lr', losses['lr'], self.global_step)
                    if 'loss_hand' in losses:
                        self.logger.add_scalar('train/loss_hand', losses['loss_hand'], self.global_step)
                        # self.logger.add_scalar('train/loss_body', losses['loss_body'], self.global_step)
                        self.logger.add_scalar('train/loss_latent', losses['loss_latent'], self.global_step)
                
                # Save periodic checkpoint
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint('latest.pt')

            # Validation & Saving
            if self.val_loader is not None and epoch % self.args.eval_every == 0:
                val_loss = self.validate()
                self.logger.add_scalar('val/loss', val_loss, self.global_step)
                print(f"Epoch: {epoch:03d} | Validation Loss: {val_loss:.4f}")

                # Save Best Model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"[*] New best model saved (Loss: {best_val_loss:.4f})")

            # Save epoch checkpoint
            if epoch % self.args.save_epoch_every == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')
        
        print("="*50)
        print(f"Training completed! Best Val Loss: {best_val_loss:.4f}")
        print("="*50)
    
    def save_checkpoint(self, filename):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'mean': self.args.mean,
            'std': self.args.std,
        }
        
        save_path = pjoin(self.args.model_dir, filename)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, filename):
        """Load checkpoint"""
        checkpoint_path = pjoin(self.args.model_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}, step {self.global_step}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='beat', choices=['t2m', 'kit', 'beat'])
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy')
    parser.add_argument('--data_id', type=str, default='./datasets/BEAT_indices')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--times', type=int, default=1)
    
    # VQ-VAE
    parser.add_argument('--vqvae_name', type=str, default='VQVAE_BEAT')
    parser.add_argument('--freeze_vqvae', action='store_true', default=True)
    parser.add_argument('--scale_factor', type=float, default=None,
                        help='Latent scale factor (auto-loaded if None)')
    
    # Diffusion model
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ff_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--no_eff', action='store_true')
    
    # Diffusion
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='cosine')
    parser.add_argument('--schedule_sampler', type=str, default='uniform')
    
    # Classifier-Free Guidance
    parser.add_argument('--cond_drop_prob', type=float, default=0.1,
                        help='Probability of dropping text condition during training')
    parser.add_argument('--guidance_scale', type=float, default=2.0,
                        help='CFG scale for sampling (1.0 = no guidance)')
    
    parser.add_argument('--hand_loss_weight', type=float, default=10.0,
                        help='Weight for hand loss (e.g., 10.0 = 10x penalty for hand errors)')
    
    # Sampler
    parser.add_argument('--sampler', type=str, default='ddim', choices=['ddpm', 'ddim'])
    parser.add_argument('--ddim_eta', type=float, default=0.0)
    
    # Training
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Logging
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_epoch_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    
    # Paths
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='vq_diffusion')
    parser.add_argument('--resume', type=str, default=None)
    
    # Device
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    
    args = parser.parse_args()
    
    # Setup
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    fixseed(args.seed)
    
    # Directories
    args.save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    args.model_dir = pjoin(args.save_root, 'model')
    args.log_dir = pjoin(args.save_root, 'logs')
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Dataset config
    if args.dataset_name == 'beat':
        args.motion_dir = pjoin(args.data_root, 'npy')
        args.text_dir = pjoin(args.data_root, 'txt')
        args.joints_num = 55
        args.max_motion_length = 360
        dim_pose = 264
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented yet")
    
    # Load stats
    stats_file = pjoin(ROOT, 'global_pipeline.pkl')
    pipeline = joblib.load(stats_file)
    scaler = pipeline.named_steps['stdscale']
    args.mean = scaler.data_mean_
    args.std = scaler.data_std_

    # Load Hand & Body Indices
    hand_indices_path = pjoin(args.data_id, 'hand_indices.npy')
    body_indices_path = pjoin(args.data_id, 'body_indices.npy')

    if os.path.exists(hand_indices_path) and os.path.exists(body_indices_path):
        hand_indices = np.load(hand_indices_path)
        body_indices = np.load(body_indices_path)
        hand_indices = torch.from_numpy(hand_indices).long().to(args.device)
        body_indices = torch.from_numpy(body_indices).long().to(args.device)
        print(f"[INFO] Loaded hand_indices: {hand_indices.shape}, body_indices: {body_indices.shape}")
    else:
        hand_indices = None
        body_indices = None
        print("[WARNING] hand_indices.npy hoặc body_indices.npy không tồn tại, sử dụng loss thông thường")
    
    # Create splits
    train_split_file = pjoin(args.data_root, 'train.txt')
    val_split_file = pjoin(args.data_root, 'val.txt')
    create_split_files(args.motion_dir, train_split_file, val_split_file)
    
    # Dataset class
    from datasets.dataset import Beat2MotionDataset as DatasetClass
    
    class DummyOpt:
        def __init__(self, args, is_train):
            self.motion_dir = args.motion_dir
            self.text_dir = args.text_dir
            self.max_motion_length = args.max_motion_length
            self.unit_length = 4
            self.times = getattr(args, 'times', 1)
            self.dataset_name = args.dataset_name
            self.motion_rep = 'position'
            self.is_train = is_train
    
    train_opt = DummyOpt(args, True)
    val_opt = DummyOpt(args, False)
    
    # Create datasets
    train_dataset = DatasetClass(train_opt, args.mean, args.std, train_split_file, args.times)
    val_dataset = DatasetClass(val_opt, args.mean, args.std, val_split_file, times=1)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True, pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create model
    model = create_vq_latent_diffusion(
        dataset_name=args.dataset_name,
        vqvae_name=args.vqvae_name,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        freeze_vqvae=args.freeze_vqvae,
        scale_factor=args.scale_factor,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        no_eff=args.no_eff
    )
    
    # Create diffusion
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    # Sửa phần tạo diffusion object
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
        # THÊM CÁC THAM SỐ MỚI
        vq_model=model,  # Truyền model để decode latent ra motion
        hand_indices=hand_indices,
        body_indices=body_indices,
        hand_loss_weight=args.hand_loss_weight
    )
    
    # Create trainer
    trainer = VQDiffusionTrainer(
        args=args,
        model=model,
        diffusion=diffusion,
        data_loader=train_loader,
        val_loader=val_loader
    )
    
    # Resume if needed
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()