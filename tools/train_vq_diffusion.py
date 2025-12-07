"""
Training script for VQ-KL Latent Diffusion
Uses hybrid VQ-KL autoencoder with diffusion in latent space
Updated with Sobolev Norm (High-order derivative loss)
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
import torch.nn.functional as F
import numpy as np
import joblib
import random

from os.path import join as pjoin
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.vq_diffusion import create_vqkl_latent_diffusion, VQKLLatentDiffusionWrapper
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


def sobolev_loss(pred, target, depth=2):
    """
    Calculate Sobolev Loss (loss on high-order derivatives)
    Args:
        pred: (B, T, C) Predicted latent/motion
        target: (B, T, C) Target latent/motion
        depth: Number of derivatives to calculate (1=velocity, 2=acceleration)
    """
    loss = 0.0
    # Note: 0-th order (Position) is usually covered by the main reconstruction loss (L1/MSE/KL)
    # We focus on derivatives here.
    
    curr_pred = pred
    curr_target = target
    
    for d in range(depth):
        if curr_pred.shape[1] < 2:
            break
            
        # Calculate finite difference (derivative proxy) along time dimension (dim=1)
        curr_pred = curr_pred[:, 1:] - curr_pred[:, :-1]
        curr_target = curr_target[:, 1:] - curr_target[:, :-1]
        
        # Add MSE of the derivative
        loss += F.mse_loss(curr_pred, curr_target)
        
    return loss


class VQKLDiffusionTrainer:
    """Trainer for VQ-KL Latent Diffusion with CFG support"""
    
    def __init__(self, args, model, diffusion, data_loader, val_loader=None):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.device = args.device
        
        # Wrap model
        self.wrapped_model = VQKLLatentDiffusionWrapper(model)
        
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
        
        print(f"[INFO] VQ-KL Latent Diffusion Training Setup:")
        print(f"  - KL posterior mode: {model.use_kl_posterior}")
        print(f"  - Classifier-Free Guidance: p_uncond={args.cond_drop_prob}")
        print(f"  - Hand loss weight: {args.hand_loss_weight}")
        if args.sobolev_loss_weight > 0:
            print(f"  - Sobolev Loss Enabled: weight={args.sobolev_loss_weight}, depth={args.sobolev_depth}")
    
    def train_step(self, batch):
        """Single training step with CFG and optional hand boosting"""
        self.model.train()
        
        # Unpack batch
        text, motion, m_lens = batch
        motion = motion.to(self.device).float()
        B = motion.shape[0]
        
        # Apply hand signal boosting if configured
        if hasattr(self.args, 'hand_boost_factor') and self.args.hand_boost_factor > 1.0:
            if hasattr(self.diffusion, 'hand_indices') and self.diffusion.hand_indices is not None:
                hand_idx = self.diffusion.hand_indices
                motion_for_encode = motion.clone()
                motion_for_encode[:, :, hand_idx] *= self.args.hand_boost_factor
            else:
                motion_for_encode = motion
        else:
            motion_for_encode = motion
        
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
        
        # Encode to normalized latent space using VQ-KL
        with torch.no_grad():
            latent, encode_info = self.model.encode_to_latent(
                motion_for_encode, 
                sample_posterior=self.model.use_kl_posterior
            )
        
        # Sample timesteps
        t, weights = self.schedule_sampler.sample(B, self.device)
        
        # Prepare model kwargs
        model_kwargs = {  
            'y': {
                'text': text,
                'length': m_lens,
                'raw_motion': motion  # Original motion for loss
            }
        }
        
        # Compute diffusion loss
        compute_losses = self.diffusion.training_losses(
            self.wrapped_model,
            latent,
            t,
            model_kwargs=model_kwargs
        )
        
        # Base Diffusion Loss
        if 'loss' in compute_losses:
            loss_total = compute_losses['loss'].mean()
        elif 'l1' in compute_losses:
            loss_total = compute_losses['l1'].mean()
        elif 'mse' in compute_losses:
            loss_total = compute_losses['mse'].mean()
        else:
            # Fallback
            loss_total = list(compute_losses.values())[0].mean()
        
        # --- Add Sobolev Loss (Temporal Smoothness) ---
        sob_loss_val = 0.0
        if self.args.sobolev_loss_weight > 0:
            # We need the predicted x_start (clean latent) to calculate smoothness
            # Most diffusion implementations return 'pred_xstart' in the losses dict
            pred_xstart = compute_losses.get('pred_xstart')
            
            if pred_xstart is not None:
                # Calculate Sobolev loss between Predicted Latent and Ground Truth Latent
                sob_loss = sobolev_loss(pred_xstart, latent, depth=self.args.sobolev_depth)
                loss_total = loss_total + self.args.sobolev_loss_weight * sob_loss
                sob_loss_val = sob_loss.item()
            else:
                # If prediction is not available (e.g., simpler diffusion impl), try model_output
                # Only if model predicts X_START (not EPSILON)
                if self.diffusion.model_mean_type == ModelMeanType.START_X:
                    model_output = compute_losses.get('model_output')
                    if model_output is not None:
                        sob_loss = sobolev_loss(model_output, latent, depth=self.args.sobolev_depth)
                        loss_total = loss_total + self.args.sobolev_loss_weight * sob_loss
                        sob_loss_val = sob_loss.item()

        # Add KL loss if available
        if encode_info.get('kl_loss') is not None:
            kl_loss = encode_info['kl_loss'].mean()
            loss_total = loss_total + self.args.kl_weight * kl_loss
        else:
            kl_loss = 0.0
        
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
        
        # Return losses
        result = {
            'loss': loss_total.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'sobolev_loss': sob_loss_val,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
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
        val_kl_losses = []
        
        for batch in self.val_loader:
            text, motion, m_lens = batch
            motion = motion.to(self.device).float()
            
            # Encode to latent
            latent, encode_info = self.model.encode_to_latent(
                motion, 
                sample_posterior=False  # Use deterministic encoding for validation
            )
            
            B = latent.shape[0]
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
            
            model_kwargs = {
                'y': {
                    'text': text,
                    'length': m_lens,
                    'raw_motion': motion 
                }
            }
            
            compute_losses = self.diffusion.training_losses(
                self.wrapped_model,
                latent,
                t,
                model_kwargs=model_kwargs
            )
            
            loss = compute_losses['loss'].mean()
            val_losses.append(loss.item())
            
            if encode_info.get('kl_loss') is not None:
                val_kl_losses.append(encode_info['kl_loss'].mean().item())
        
        result = {'diffusion_loss': np.mean(val_losses)}
        if val_kl_losses:
            result['kl_loss'] = np.mean(val_kl_losses)
        
        return result
    
    @torch.no_grad()
    def sample_with_cfg(self, text, lengths, num_samples=1, guidance_scale=2.0):
        """Generate samples with Classifier-Free Guidance"""
        self.model.eval()
        
        B = len(text)
        T_latent = self.model.num_frames
        
        # Get latent dimension
        if hasattr(self.model.vqkl, 'embed_dim'):
            latent_dim = self.model.vqkl.embed_dim
        else:
            latent_dim = self.model.vqkl.code_dim
        
        shape = (B * num_samples, T_latent, latent_dim)
        
        # Prepare conditional inputs
        text_cond = text * num_samples
        lengths_repeated = lengths * num_samples
        
        print(f"Sampling with CFG (scale={guidance_scale})...")
        
        # Sample using configured sampler
        if self.args.sampler == 'ddim':
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
        
        # Decode from latent
        motion_samples = self.model.decode_from_latent(latent=latent_samples)
        return motion_samples
    
    def train(self):
        """Main training loop"""
        print("="*70)
        print("Starting VQ-KL Latent Diffusion Training")
        print("="*70)
        print(f"Epochs: {self.args.max_epoch}")
        print(f"Train batches: {len(self.data_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print(f"CFG dropout prob: {self.args.cond_drop_prob}")
        print(f"KL weight: {self.args.kl_weight}")
        print(f"Sobolev weight: {self.args.sobolev_loss_weight}")
        print("="*70)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.max_epoch):
            self.epoch = epoch
            epoch_losses = []
            epoch_kl_losses = []
            epoch_sob_losses = []
            
            # Training
            for batch_idx, batch in enumerate(self.data_loader):
                losses = self.train_step(batch)
                epoch_losses.append(losses['loss'])
                if losses['kl_loss'] > 0:
                    epoch_kl_losses.append(losses['kl_loss'])
                if losses['sobolev_loss'] > 0:
                    epoch_sob_losses.append(losses['sobolev_loss'])
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-self.args.log_every:])
                    
                    log_str = f"Epoch: {epoch:03d} | Step: {self.global_step:06d} | Loss: {avg_loss:.4f}"
                    if epoch_kl_losses:
                        avg_kl = np.mean(epoch_kl_losses[-self.args.log_every:])
                        log_str += f" | KL: {avg_kl:.6f}"
                    if epoch_sob_losses:
                        avg_sob = np.mean(epoch_sob_losses[-self.args.log_every:])
                        log_str += f" | Sob: {avg_sob:.6f}"
                    if 'loss_hand' in losses:
                        log_str += f" | Hand: {losses['loss_hand']:.4f}"
                    log_str += f" | LR: {losses['lr']:.8f}"
                    print(log_str)
                    
                    # Tensorboard
                    self.logger.add_scalar('train/loss_total', avg_loss, self.global_step)
                    self.logger.add_scalar('train/lr', losses['lr'], self.global_step)
                    if epoch_kl_losses:
                        self.logger.add_scalar('train/kl_loss', avg_kl, self.global_step)
                    if epoch_sob_losses:
                        self.logger.add_scalar('train/sobolev_loss', avg_sob, self.global_step)
                    if 'loss_hand' in losses:
                        self.logger.add_scalar('train/loss_hand', losses['loss_hand'], self.global_step)
                
                # Save periodic checkpoint
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint('latest.pt')

            # Validation
            if self.val_loader is not None and epoch % self.args.eval_every == 0:
                val_results = self.validate()
                val_loss = val_results['diffusion_loss']
                self.logger.add_scalar('val/diffusion_loss', val_loss, self.global_step)
                
                log_str = f"Epoch: {epoch:03d} | Val Loss: {val_loss:.4f}"
                if 'kl_loss' in val_results:
                    self.logger.add_scalar('val/kl_loss', val_results['kl_loss'], self.global_step)
                    log_str += f" | Val KL: {val_results['kl_loss']:.6f}"
                print(log_str)

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"[*] New best model saved (Loss: {best_val_loss:.4f})")

            # Save epoch checkpoint
            if epoch % self.args.save_epoch_every == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')
        
        print("="*70)
        print(f"Training completed! Best Val Loss: {best_val_loss:.4f}")
        print("="*70)
    
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
    
    parser = argparse.ArgumentParser(description='Train VQ-KL Latent Diffusion')
    
    # Dataset
    parser.add_argument('--dataset_name', type=str, default='beat', choices=['t2m', 'kit', 'beat'])
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy')
    parser.add_argument('--data_id', type=str, default='./datasets/BEAT_indices')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--times', type=int, default=1)
    
    # VQ-KL Autoencoder
    parser.add_argument('--vqkl_name', type=str, default='VQKL_BEAT')
    parser.add_argument('--freeze_vqkl', action='store_true', default=True)
    parser.add_argument('--scale_factor', type=float, default=None)
    parser.add_argument('--use_kl_posterior', action='store_true', default=True,
                        help='Use KL posterior sampling instead of VQ codes')
    
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
    parser.add_argument('--cond_drop_prob', type=float, default=0.1)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    
    # Loss weights
    parser.add_argument('--loss_type', type=str, default='l1', choices=['mse', 'l1', 'rescaled_mse', 'rescaled_l1'], 
                    help='Loại hàm mất mát (L1 thường tốt hơn cho motion)')
    parser.add_argument('--hand_loss_weight', type=float, default=10.0)
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='Weight for KL divergence loss')
    parser.add_argument('--hand_boost_factor', type=float, default=1.0,
                        help='Signal boost factor for hand joints during encoding')
    
    # Sobolev Loss Args
    parser.add_argument('--sobolev_loss_weight', type=float, default=0.0,
                        help='Weight for Sobolev (derivative) loss. 0.0 to disable.')
    parser.add_argument('--sobolev_depth', type=int, default=2,
                        help='Depth of Sobolev loss (1 for velocity, 2 for acceleration)')
    
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
    parser.add_argument('--save_epoch_every', type=int, default=25)
    parser.add_argument('--eval_every', type=int, default=1)
    
    # Paths
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='vqkl_diffusion')
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
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")
    
    # Load stats
    stats_file = pjoin(ROOT, 'global_pipeline.pkl')
    pipeline = joblib.load(stats_file)
    scaler = pipeline.named_steps['stdscale']
    args.mean = scaler.data_mean_
    args.std = scaler.data_std_

    # Load hand & body indices
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
        print("[WARNING] Hand/body indices not found")
    
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
    model = create_vqkl_latent_diffusion(
        dataset_name=args.dataset_name,
        vqkl_name=args.vqkl_name,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        freeze_vqkl=args.freeze_vqkl,
        scale_factor=args.scale_factor,
        use_kl_posterior=args.use_kl_posterior,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        no_eff=args.no_eff
    )
    
    # Create diffusion
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    loss_type_map = {
        'mse': LossType.MSE,
        'l1': LossType.MSE,
        'rescaled_mse': LossType.RESCALED_MSE,
        'rescaled_l1': LossType.RESCALED_MSE
    }

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=loss_type_map[args.loss_type],
        rescale_timesteps=False,
        vq_model=model, 
        hand_indices=hand_indices,
        body_indices=body_indices,
        hand_loss_weight=args.hand_loss_weight
    )
    
    # Create trainer
    trainer = VQKLDiffusionTrainer(
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