"""
Training script for VQ-VAE Latent Diffusion with MotionDiffuse
Integrates with GaussianDiffusion from document 7
UPDATED: Added Validation Loop and Best Model Saving
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
# from data.t2m_dataset import MotionDataset
from utils.fixseed import fixseed
from datasets.pymo import *

def create_split_files(motion_dir, train_file, val_file, val_ratio=0.1):
    """
    Auto-create train.txt and val.txt by listing all .npy files.
    Splits the data randomly if val.txt doesn't exist.
    """
    # Nếu cả 2 file đều tồn tại thì không làm gì cả
    if os.path.exists(train_file) and os.path.exists(val_file):
        return

    print(f"[INFO] Creating split files from {motion_dir}...")
    os.makedirs(os.path.dirname(train_file), exist_ok=True)
    
    # Walk recursively to pick up .npy files
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
    
    # Shuffle and split
    motion_files = sorted(motion_files)
    random.shuffle(motion_files)
    
    val_size = int(len(motion_files) * val_ratio)
    val_files = motion_files[:val_size]
    train_files = motion_files[val_size:]
    
    # Write files
    with open(train_file, 'w', encoding='utf-8') as f:
        for name in train_files:
            f.write(f"{name}\n")
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for name in val_files:
            f.write(f"{name}\n")
            
    print(f"[INFO] Created {train_file} ({len(train_files)} samples)")
    print(f"[INFO] Created {val_file} ({len(val_files)} samples)")


class VQDiffusionTrainer:
    """Trainer for VQ-VAE Latent Diffusion"""
    
    def __init__(self, args, model, diffusion, data_loader, val_loader=None):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.val_loader = val_loader # Store val_loader
        self.device = args.device
        
        # Wrap model for GaussianDiffusion compatibility
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
        
        # Schedule sampler for timestep sampling
        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler,
            diffusion
        )
        
        # Logging
        self.logger = SummaryWriter(args.log_dir)
        self.global_step = 0
        self.epoch = 0
        
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        # Unpack batch
        text, motion, m_lens = batch
        motion = motion.to(self.device).float()
        
        # Encode to latent space (frozen VQ-VAE)
        with torch.no_grad():
            latent, _ = self.model.encode_to_latent(motion)
        
        # CRITICAL FIX: Ensure latent is in (B, T, D) format
        B = latent.shape[0]
        expected_T = self.model.num_frames
        expected_D = self.model.vqvae.code_dim
        
        if latent.shape[1] == expected_D and latent.shape[2] == expected_T:
            # Shape is (B, code_dim, T_latent) - need to transpose
            latent = latent.permute(0, 2, 1)
        elif latent.shape[1] == expected_T and latent.shape[2] == expected_D:
            pass
        else:
            raise ValueError(f"Unexpected latent shape: {latent.shape}")
        
        # Sample timesteps
        t, weights = self.schedule_sampler.sample(B, self.device)
        
        # Prepare model kwargs
        model_kwargs = {
            'y': {
                'text': text,
                'length': m_lens,
            }
        }
        
        # Compute loss
        compute_losses = self.diffusion.training_losses(
            self.wrapped_model,
            latent,  # Use latent instead of raw motion
            t,
            model_kwargs=model_kwargs
        )
        
        # Extract losses
        loss_mse = compute_losses['mse'].mean()
        
        # Backward
        self.optimizer.zero_grad()
        loss_mse.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.args.grad_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Return losses for logging
        losses = {
            'loss': loss_mse.item(),
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        return losses
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []
        
        # print(f"[INFO] Running validation on {len(self.val_loader)} batches...")
        
        for batch in self.val_loader:
            text, motion, m_lens = batch
            motion = motion.to(self.device).float()
            
            # Encode to latent
            latent, _ = self.model.encode_to_latent(motion)

            # Handle Dimensions (same as train_step)
            expected_T = self.model.num_frames
            expected_D = self.model.vqvae.code_dim
            if latent.shape[1] == expected_D and latent.shape[2] == expected_T:
                latent = latent.permute(0, 2, 1)

            B = latent.shape[0]
            # For validation, we usually fix t or sample random t
            t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=self.device)
            
            model_kwargs = {
                'y': {
                    'text': text,
                    'length': m_lens,
                }
            }
            
            compute_losses = self.diffusion.training_losses(
                self.wrapped_model,
                latent,
                t,
                model_kwargs=model_kwargs
            )
            
            loss = compute_losses['mse'].mean()
            val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        return avg_val_loss
    
    @torch.no_grad()
    def sample(self, text, lengths, num_samples=1):
        """Generate samples from text prompts"""
        self.model.eval()
        
        B = len(text)
        T_latent = self.model.num_frames
        code_dim = self.model.latent_dim
        
        shape = (B * num_samples, T_latent, code_dim)
        
        model_kwargs = {
            'y': {
                'text': text * num_samples,
                'length': lengths * num_samples,
            }
        }
        
        print(f"Sampling with {self.args.sampler} (eta={self.args.ddim_eta})...")

        if self.args.sampler == 'ddpm':
            latent_samples = self.diffusion.p_sample_loop(
                self.wrapped_model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=True
            )
        elif self.args.sampler == 'ddim':
            latent_samples = self.diffusion.ddim_sample_loop(
                self.wrapped_model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=True,
                eta=self.args.ddim_eta 
            )
        else:
            raise ValueError(f"Unknown sampler: {self.args.sampler}")
        
        motion_samples = self.model.decode_from_latent(latent=latent_samples)
        return motion_samples
    
    def train(self):
        """Main training loop"""
        print("="*50)
        print("Starting VQ Latent Diffusion Training")
        print("="*50)
        print(f"Epochs: {self.args.max_epoch}")
        print(f"Train batches: {len(self.data_loader)}")
        if self.val_loader:
            print(f"Val batches:   {len(self.val_loader)}")
        print("="*50)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.max_epoch):
            self.epoch = epoch
            epoch_losses = []
            
            # --- Training ---
            for batch_idx, batch in enumerate(self.data_loader):
                losses = self.train_step(batch)
                epoch_losses.append(losses['loss'])
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.log_every == 0:
                    avg_loss = np.mean(epoch_losses[-self.args.log_every:])
                    print(f"Epoch: {epoch:03d} | Step: {self.global_step:06d} | "
                          f"Loss: {avg_loss:.4f} | LR: {losses['lr']:.8f}")
                    
                    self.logger.add_scalar('train/loss', avg_loss, self.global_step)
                    self.logger.add_scalar('train/lr', losses['lr'], self.global_step)
                
                # Save periodic checkpoint (based on steps)
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint('latest.pt')

            # --- Validation & Saving ---
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
        # print(f"Checkpoint saved to {save_path}")
    
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
    parser.add_argument('--dataset_name', type=str, default='t2m', choices=['t2m', 'kit', 'beat'])
    parser.add_argument('--data_root', type=str, default='./dataset/HumanML3D/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--times', type=int, default=1, help='Dataset repeat times (for BEAT)')
    
    # VQ-VAE
    parser.add_argument('--vqvae_name', type=str, default='VQVAE_BEAT')
    parser.add_argument('--freeze_vqvae', action='store_true', default=True)
    
    # Diffusion model
    parser.add_argument('--latent_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--ff_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--no_eff', action='store_true', help='Use standard attention instead of efficient')
    
    # Diffusion
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--noise_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    parser.add_argument('--schedule_sampler', type=str, default='uniform')
    
    # --- SAMPLER OPTIONS ---
    parser.add_argument('--sampler', type=str, default='ddpm', choices=['ddpm', 'ddim'], 
                        help='Sampler to use during evaluation/generation')
    parser.add_argument('--ddim_eta', type=float, default=0.0, 
                        help='DDIM eta (0.0 for deterministic, 1.0 for DDPM)')
    
    # Training
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Logging
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--save_epoch_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--sample_every', type=int, default=10)
    
    # Paths
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='vq_diffusion')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    
    # Device
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    
    # Set seed
    fixseed(args.seed)
    
    # Setup directories
    args.save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    args.model_dir = pjoin(args.save_root, 'model')
    args.log_dir = pjoin(args.save_root, 'logs')
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Dataset configuration
    if args.dataset_name == 't2m':
        args.data_root = './dataset/HumanML3D/'
        args.motion_dir = pjoin(args.data_root, 'new_joint_vecs')
        args.text_dir = pjoin(args.data_root, 'texts')
        args.joints_num = 22
        args.max_motion_length = 196
        dim_pose = 263

    elif args.dataset_name == 'kit':
        args.data_root = './dataset/KIT-ML/'
        args.motion_dir = pjoin(args.data_root, 'new_joint_vecs')
        args.text_dir = pjoin(args.data_root, 'texts')
        args.joints_num = 21
        args.max_motion_length = 196
        dim_pose = 251

    elif args.dataset_name == 'beat':
        args.data_root = './datasets/BEAT_numpy'
        args.motion_dir = pjoin(args.data_root, 'npy')
        args.text_dir = pjoin(args.data_root, 'txt')
        args.joints_num = 55
        args.max_motion_length = 360 
        dim_pose = 264 
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Load data stats
    stats_file_path = "/home/serverai/ltdoanh/Motion_Diffusion/global_pipeline.pkl"
    try:
        pipeline = joblib.load(stats_file_path)
        scaler = pipeline.named_steps['stdscale']
        mean = scaler.data_mean_
        std = scaler.data_std_
        print("[INFO] Mean and Std loaded successfully.")
    except Exception as e:
        print(f"Error loading stats: {e}")
        raise e

    args.mean = np.array(mean)
    args.std = np.array(std)
    
    train_split_file = pjoin(args.data_root, 'train.txt')
    val_split_file = pjoin(args.data_root, 'val.txt')

    # Create datasets options
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
    
    train_opt = DummyOpt(args, is_train=True)
    val_opt = DummyOpt(args, is_train=False)
    
    # Select dataset class
    if args.dataset_name in ['t2m', 'kit']:
        from data.t2m_dataset import Text2MotionDataset as DatasetClass
        # Ensure dataset supports 'val' split loading if implementing validation for T2M/KIT
    elif args.dataset_name == 'beat':
        from datasets.dataset import Beat2MotionDataset as DatasetClass
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Auto-create splits for BEAT if needed
    if args.dataset_name == 'beat':
        create_split_files(args.motion_dir, train_split_file, val_split_file)
    
    # Train Dataset
    train_dataset = DatasetClass(train_opt, mean, std, train_split_file, getattr(args, 'times', 1))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    # Val Dataset
    val_dataset = DatasetClass(val_opt, mean, std, val_split_file, times=1)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset:   {len(val_dataset)} samples")
    
    # Create model
    model = create_vq_latent_diffusion(
        dataset_name=args.dataset_name,
        vqvae_name=args.vqvae_name,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        freeze_vqvae=args.freeze_vqvae,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        no_eff=args.no_eff
    )
    
    # Create diffusion process
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=False
    )
    
    # Create trainer
    trainer = VQDiffusionTrainer(
        args=args,
        model=model,
        diffusion=diffusion,
        data_loader=train_loader,
        val_loader=val_loader # Pass the val_loader here
    )
    
    # Resume
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # Start
    trainer.train()


if __name__ == "__main__":
    main()