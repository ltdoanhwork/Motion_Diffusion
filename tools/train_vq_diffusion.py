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
import matplotlib.pyplot as plt

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
    """Calculate Sobolev Loss (loss on high-order derivatives)"""
    loss = 0.0
    curr_pred = pred
    curr_target = target
    
    for d in range(depth):
        if curr_pred.shape[1] < 2:
            break
        curr_pred = curr_pred[:, 1:] - curr_pred[:, :-1]
        curr_target = curr_target[:, 1:] - curr_target[:, :-1]
        loss += F.mse_loss(curr_pred, curr_target)
    return loss


def compute_bone_lengths(motion, bone_pairs):
    """Compute bone lengths from motion data"""
    if motion.dim() == 3:
        B, T, D = motion.shape
        J = D // 3
        motion = motion.reshape(B, T, J, 3)
    
    bone_lengths = []
    for parent_idx, child_idx in bone_pairs:
        parent_pos = motion[:, :, parent_idx, :]
        child_pos = motion[:, :, child_idx, :]
        bone_vec = child_pos - parent_pos
        bone_len = torch.norm(bone_vec, dim=-1)
        bone_lengths.append(bone_len)
    
    return torch.stack(bone_lengths, dim=-1)


def bone_length_loss(pred_motion, target_motion, bone_pairs):
    """Bone length consistency loss"""
    pred_lengths = compute_bone_lengths(pred_motion, bone_pairs)
    target_lengths = compute_bone_lengths(target_motion, bone_pairs)
    return F.l1_loss(pred_lengths, target_lengths)


def compute_fft_loss(pred_motion, target_motion):
    """Frequency domain loss"""
    pred_fft = torch.fft.rfft(pred_motion, dim=1)
    target_fft = torch.fft.rfft(target_motion, dim=1)
    
    loss_amp = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
    loss_real = F.l1_loss(pred_fft.real, target_fft.real)
    loss_imag = F.l1_loss(pred_fft.imag, target_fft.imag)
    
    return loss_amp + 0.5 * (loss_real + loss_imag)


def compute_detailed_hand_loss(pred_motion, target_motion, hand_indices, hand_bone_pairs):
    """Detailed hand loss - FIXED VERSION"""
    # Ensure (B, T, J, 3) format
    if pred_motion.dim() == 3 and pred_motion.shape[-1] != 3:
        B, T, D = pred_motion.shape
        J = D // 3
        pred_motion = pred_motion.reshape(B, T, J, 3)
        target_motion = target_motion.reshape(B, T, J, 3)
    
    # Extract hand joints
    pred_hand = pred_motion[:, :, hand_indices]
    target_hand = target_motion[:, :, hand_indices]
    
    # Position loss
    loss_pos = F.l1_loss(pred_hand, target_hand)
    
    # Velocity loss
    loss_vel = 0.0
    if pred_hand.shape[1] > 1:
        pred_vel = pred_hand[:, 1:] - pred_hand[:, :-1]
        target_vel = target_hand[:, 1:] - target_hand[:, :-1]
        loss_vel = F.l1_loss(pred_vel, target_vel)
    
    # Bone structure loss
    loss_bone_struct = 0.0
    if hand_bone_pairs and len(hand_bone_pairs) > 0:
        pred_vecs = []
        target_vecs = []
        
        for p_idx, c_idx in hand_bone_pairs:
            pv = pred_motion[:, :, c_idx] - pred_motion[:, :, p_idx]
            tv = target_motion[:, :, c_idx] - target_motion[:, :, p_idx]
            pred_vecs.append(pv)
            target_vecs.append(tv)
        
        pred_vecs = torch.stack(pred_vecs, dim=2)
        target_vecs = torch.stack(target_vecs, dim=2)
        loss_bone_struct = F.l1_loss(pred_vecs, target_vecs)
    
    total_hand_loss = 1.0 * loss_pos + 1.5 * loss_vel + 2.0 * loss_bone_struct
    
    return total_hand_loss, {
        'h_pos': loss_pos.item() if isinstance(loss_pos, torch.Tensor) else loss_pos,
        'h_vel': loss_vel.item() if isinstance(loss_vel, torch.Tensor) else loss_vel,
        'h_bone': loss_bone_struct.item() if isinstance(loss_bone_struct, torch.Tensor) else loss_bone_struct
    }


def save_loss_plot(history, save_path):
    """Save loss plot"""
    valid_keys = [k for k in history.keys() if len(history[k]) > 0]
    if not valid_keys:
        return
    
    plt.figure(figsize=(20, 15))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Subplot 1: Total Loss & LR
    plt.subplot(2, 2, 1)
    if 'loss' in history:
        plt.plot(history['loss'], label='Total Loss', color='black', linewidth=1.5)
    plt.title("Total Loss Process")
    plt.xlabel("Log Steps")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(True)
    
    if 'lr' in history:
        ax2 = plt.gca().twinx()
        ax2.plot(history['lr'], label='Learning Rate', color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel("Learning Rate", color='red')
    
    # Subplot 2: Physical Losses
    plt.subplot(2, 2, 2)
    phys_keys = ['phys_rec_loss', 'phys_vel_loss', 'bone_loss', 'foot_slide_loss']
    for k in phys_keys:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)
    plt.title("Physical Body Consistency Losses")
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Hand & Detail Losses
    plt.subplot(2, 2, 3)
    detail_keys = ['hand_loss', 'hand_bone_loss', 'fft_loss']
    for k in detail_keys:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)
    plt.title("Detailed Hand & Frequency Losses")
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Latent Space Losses
    plt.subplot(2, 2, 4)
    latent_keys = ['kl_loss', 'sobolev_loss']
    for k in latent_keys:
        if k in history and len(history[k]) > 0:
            plt.plot(history[k], label=k)
    plt.title("Latent Space Losses (VQ/KL & Smoothness)")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Loss plot saved to {save_path}")


class VQKLDiffusionTrainer:
    """Trainer for VQ-KL Latent Diffusion - COMPLETE FIX"""
    
    def __init__(self, args, model, diffusion, data_loader, val_loader=None):
        self.args = args
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.device = args.device
        
        self.wrapped_model = VQKLLatentDiffusionWrapper(model)
        
        self.optimizer = optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.max_epoch * len(data_loader),
            eta_min=1e-6
        )
        
        self.schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler,
            diffusion
        )
        
        self.logger = SummaryWriter(args.log_dir)
        self.global_step = 0
        self.epoch = 0
        
        self.loss_history = {
            'loss': [], 'lr': [],
            'kl_loss': [], 'sobolev_loss': [],
            'phys_rec_loss': [], 'phys_vel_loss': [],
            'foot_slide_loss': [], 'bone_loss': [],
            'fft_loss': [], 'hand_loss': [], 'hand_bone_loss': []
        }
        
        self.hand_indices = diffusion.hand_indices
        self.body_indices = diffusion.body_indices
        
        print(f"[INFO] VQ-KL Latent Diffusion Training Setup:")
        print(f"  - Hand indices: {self.hand_indices.shape if self.hand_indices is not None else 'None'}")
        print(f"  - Hand loss weight: {args.hand_loss_weight}")
        print(f"  - Physical loss weight: {args.physical_loss_weight}")
        print(f"  - Sobolev weight: {args.sobolev_loss_weight}")
    
    def train_step(self, batch):
        """Single training step - COMPLETE FIX"""
        self.model.train()
        
        text, motion, m_lens = batch
        motion = motion.to(self.device).float()
        B = motion.shape[0]
        
        # Hand boosting
        motion_for_encode = motion.clone()
        if hasattr(self.args, 'hand_boost_factor') and self.args.hand_boost_factor > 1.0:
            if self.hand_indices is not None:
                motion_for_encode[:, :, self.hand_indices] *= self.args.hand_boost_factor
        
        # CFG dropout
        if self.args.cond_drop_prob > 0:
            uncond_mask = torch.rand(B) < self.args.cond_drop_prob
            text = ["" if uncond_mask[i] else txt for i, txt in enumerate(text)]
        
        # Encode to latent
        with torch.no_grad():
            latent, encode_info = self.model.encode_to_latent(
                motion_for_encode,
                sample_posterior=self.model.use_kl_posterior
            )
        
        # Sample timesteps
        t, weights = self.schedule_sampler.sample(B, self.device)
        
        model_kwargs = {
            'y': {
                'text': text,
                'length': m_lens,
                'raw_motion': motion
            }
        }
        
        # Compute diffusion loss
        compute_losses = self.diffusion.training_losses(
            self.wrapped_model,
            latent,
            t,
            model_kwargs=model_kwargs
        )
        
        # FIX: FORCE compute pred_xstart if not available
        pred_xstart = compute_losses.get('pred_xstart')
        if pred_xstart is None:
            # Get model output (epsilon prediction)
            model_output = compute_losses.get('model_output')
            if model_output is None:
                # Run model directly to get output
                model_output = self.wrapped_model(latent, t, **model_kwargs)
            
            # Manually compute pred_xstart from epsilon
            # Use the diffusion's own method
            pred_xstart = self.diffusion._predict_xstart_from_eps(
                x_t=latent,
                t=t,
                eps=model_output
            )
            # print(f"[DEBUG] Manually computed pred_xstart: {pred_xstart.shape}")
        
        # Base diffusion loss
        if 'loss' in compute_losses:
            loss_total = compute_losses['loss'].mean()
        elif 'l1' in compute_losses:
            loss_total = compute_losses['l1'].mean()
        elif 'mse' in compute_losses:
            loss_total = compute_losses['mse'].mean()
        else:
            loss_total = list(compute_losses.values())[0].mean()
        
        # Initialize all loss tracking
        result = {
            'loss': 0.0,
            'kl_loss': 0.0,
            'sobolev_loss': 0.0,
            'phys_rec_loss': 0.0,
            'phys_vel_loss': 0.0,
            'foot_slide_loss': 0.0,
            'bone_loss': 0.0,
            'fft_loss': 0.0,
            'hand_loss': 0.0,
            'hand_bone_loss': 0.0,
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # KL Loss
        if encode_info.get('kl_loss') is not None:
            kl_loss = encode_info['kl_loss'].mean()
            loss_total = loss_total + self.args.kl_weight * kl_loss
            result['kl_loss'] = kl_loss.item()
        
        # Sobolev Loss
        if self.args.sobolev_loss_weight > 0 and pred_xstart is not None:
            sob_loss = sobolev_loss(pred_xstart, latent, depth=self.args.sobolev_depth)
            loss_total = loss_total + self.args.sobolev_loss_weight * sob_loss
            result['sobolev_loss'] = sob_loss.item()
        
        # ==================================================================
        # PHYSICAL SPACE LOSSES - GUARANTEED EXECUTION
        # ==================================================================
        if pred_xstart is not None and self.args.physical_loss_weight > 0:
            # print(f"[DEBUG] Computing physical losses with pred_xstart: {pred_xstart.shape}")
            
            # Decode to physical motion
            recon_motion = self.model.decode_from_latent(pred_xstart)
            # print(f"[DEBUG] Decoded recon_motion: {recon_motion.shape}")
            
            # Create mask
            mask = torch.zeros_like(motion)
            for i, length in enumerate(m_lens):
                mask[i, :length, :] = 1.0
            
            # A. Physical Reconstruction Loss
            rec_loss = F.l1_loss(recon_motion * mask, motion * mask)
            loss_total += self.args.physical_loss_weight * rec_loss
            result['phys_rec_loss'] = rec_loss.item()
            # print(f"[DEBUG] phys_rec_loss: {rec_loss.item():.6f}")
            
            # B. Physical Velocity Loss
            if recon_motion.shape[1] > 1:
                vel_recon = recon_motion[:, 1:] - recon_motion[:, :-1]
                vel_target = motion[:, 1:] - motion[:, :-1]
                vel_loss = F.l1_loss(vel_recon * mask[:, 1:], vel_target * mask[:, 1:])
                loss_total += (self.args.physical_loss_weight * 1.5) * vel_loss
                result['phys_vel_loss'] = vel_loss.item()
                # print(f"[DEBUG] phys_vel_loss: {vel_loss.item():.6f}")
            
            # C. Foot Contact Loss
            if hasattr(self.args, 'foot_indices') and self.args.foot_indices is not None:
                foot_indices = self.args.foot_indices
                
                if recon_motion.shape[1] > 1:
                    D = motion.shape[-1]
                    J = D // 3
                    
                    recon_reshaped = recon_motion.reshape(B, -1, J, 3)
                    motion_reshaped = motion.reshape(B, -1, J, 3)
                    
                    foot_vel_recon = recon_reshaped[:, 1:, foot_indices] - recon_reshaped[:, :-1, foot_indices]
                    foot_vel_target = motion_reshaped[:, 1:, foot_indices] - motion_reshaped[:, :-1, foot_indices]
                    
                    contact_threshold = self.args.foot_contact_threshold
                    contact_mask = (torch.norm(foot_vel_target, dim=-1) < contact_threshold).float()
                    
                    foot_sliding_loss = (torch.norm(foot_vel_recon, dim=-1) * contact_mask).mean()
                    loss_total += (self.args.physical_loss_weight * 2.0) * foot_sliding_loss
                    result['foot_slide_loss'] = foot_sliding_loss.item()
                    # print(f"[DEBUG] foot_slide_loss: {foot_sliding_loss.item():.6f}")
            
            # D. Bone Length Loss
            if self.args.use_bone_loss and hasattr(self.args, 'bone_pairs'):
                bone_loss = bone_length_loss(recon_motion, motion, self.args.bone_pairs)
                loss_total += self.args.lambda_bone * bone_loss
                result['bone_loss'] = bone_loss.item()
                # print(f"[DEBUG] bone_loss: {bone_loss.item():.6f}")
            
            # E. FFT Loss
            if hasattr(self.args, 'fft_loss_weight') and self.args.fft_loss_weight > 0:
                fft_loss = compute_fft_loss(recon_motion, motion)
                loss_total += self.args.fft_loss_weight * fft_loss
                result['fft_loss'] = fft_loss.item()
                # print(f"[DEBUG] fft_loss: {fft_loss.item():.6f}")
            
            # F. HAND LOSS - GUARANTEED EXECUTION
            if self.hand_indices is not None and self.args.hand_loss_weight > 0:
                hand_idx_list = self.hand_indices.cpu().tolist() if torch.is_tensor(self.hand_indices) else list(self.hand_indices)
                h_bone_pairs = getattr(self.args, 'hand_bone_pairs', [])
                
                D = recon_motion.shape[-1]
                J = D // 3
                recon_reshaped = recon_motion.reshape(B, -1, J, 3)
                motion_reshaped = motion.reshape(B, -1, J, 3)
                
                h_loss, h_logs = compute_detailed_hand_loss(
                    recon_reshaped, motion_reshaped, hand_idx_list, h_bone_pairs
                )
                
                loss_total += self.args.hand_loss_weight * h_loss
                result['hand_loss'] = h_loss.item()
                result['hand_bone_loss'] = h_logs['h_bone']
                # print(f"[DEBUG] hand_loss: {h_loss.item():.6f}, hand_bone: {h_logs['h_bone']:.6f}")
        
        else:
            print(f"[WARNING] Physical losses SKIPPED! pred_xstart={pred_xstart is not None}, weight={self.args.physical_loss_weight}")
        
        # Backward
        result['loss'] = loss_total.item()
        
        self.optimizer.zero_grad()
        loss_total.backward()
        
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.args.grad_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return result
    
    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        val_losses = []
        val_kl_losses = []
        val_phys_losses = []
        
        for batch in self.val_loader:
            text, motion, m_lens = batch
            motion = motion.to(self.device).float()
            
            latent, encode_info = self.model.encode_to_latent(
                motion,
                sample_posterior=False
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
            
            pred_xstart = compute_losses.get('pred_xstart')
            if pred_xstart is not None:
                recon_motion = self.model.decode_from_latent(pred_xstart)
                phys_loss = F.l1_loss(recon_motion, motion)
                val_phys_losses.append(phys_loss.item())
        
        result = {'diffusion_loss': np.mean(val_losses)}
        if val_kl_losses:
            result['kl_loss'] = np.mean(val_kl_losses)
        if val_phys_losses:
            result['phys_rec_loss'] = np.mean(val_phys_losses)
        
        return result
    
    def train(self):
        """Main training loop"""
        print("="*70)
        print("Starting VQ-KL Latent Diffusion Training")
        print("="*70)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.max_epoch):
            self.epoch = epoch
            epoch_losses = []
            
            for batch_idx, batch in enumerate(self.data_loader):
                losses = self.train_step(batch)
                epoch_losses.append(losses['loss'])
                self.global_step += 1
                
                # Logging
                if self.global_step % self.args.log_every == 0:
                    log_str = f"Epoch: {epoch:03d} | Step: {self.global_step:06d}"
                    log_str += f" | Loss: {losses['loss']:.4f}"
                    log_str += f" | Phys: {losses['phys_rec_loss']:.4f}"
                    log_str += f" | Hand: {losses['hand_loss']:.4f}"
                    log_str += f" | LR: {losses['lr']:.8f}"
                    print(log_str)
                    
                    # Tensorboard
                    for k, v in losses.items():
                        if v > 0:
                            self.logger.add_scalar(f'train/{k}', v, self.global_step)
                    
                    # History
                    for k, v in losses.items():
                        if k in self.loss_history:
                            self.loss_history[k].append(v)
                    
                    if self.global_step % (self.args.log_every * 10) == 0:
                        plot_path = os.path.join(self.args.log_dir, f'loss_step{self.global_step}.png')
                        save_loss_plot(self.loss_history, plot_path)
                
                if self.global_step % self.args.save_every == 0:
                    self.save_checkpoint('latest.pt')
            
            # Validation
            if self.val_loader is not None and epoch % self.args.eval_every == 0:
                val_results = self.validate()
                val_loss = val_results['diffusion_loss']
                self.logger.add_scalar('val/diffusion_loss', val_loss, self.global_step)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"[*] Best model saved (Loss: {best_val_loss:.4f})")
            
            if epoch % self.args.save_epoch_every == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')
        
        print(f"Training completed! Best Val Loss: {best_val_loss:.4f}")
        final_plot_path = os.path.join(self.args.log_dir, 'final_loss_curve.png')
        save_loss_plot(self.loss_history, final_plot_path)
    
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
    
    # Loss weights - Latent Space
    parser.add_argument('--loss_type', type=str, default='l1', choices=['mse', 'l1', 'rescaled_mse', 'rescaled_l1'])
    parser.add_argument('--hand_loss_weight', type=float, default=20.0, help='Weight for detailed hand loss')
    parser.add_argument('--kl_weight', type=float, default=1e-6)
    parser.add_argument('--hand_boost_factor', type=float, default=1.0)
    
    # Sobolev Loss (Latent Space Smoothness)
    parser.add_argument('--sobolev_loss_weight', type=float, default=0.1,
                        help='Weight for Sobolev loss in latent space. 0.0 to disable.')
    parser.add_argument('--sobolev_depth', type=int, default=2,
                        help='Depth of derivatives (1=velocity, 2=acceleration)')
    
    # Physical Space Losses
    parser.add_argument('--physical_loss_weight', type=float, default=0.5,
                        help='Base weight for physical space losses (rec, vel, foot). 0.0 to disable.')
    parser.add_argument('--foot_contact_threshold', type=float, default=0.005,
                        help='Velocity threshold for foot contact detection')
    
    # Bone Length Loss
    parser.add_argument('--use_bone_loss', action='store_true', default=False,
                        help='Enable bone length consistency loss')
    parser.add_argument('--lambda_bone', type=float, default=0.1,
                        help='Weight for bone length loss')
    
    # Frequency Loss
    parser.add_argument('--fft_loss_weight', type=float, default=0.0, 
                        help='Weight for FFT/Frequency domain loss. Try 0.1 or 0.05')
    
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
    # Dataset config
    if args.dataset_name == 'beat':
        args.motion_dir = pjoin(args.data_root, 'npy')
        args.text_dir = pjoin(args.data_root, 'txt')
        
        args.joints_num = 27 
        args.max_motion_length = 360
        
        # Cấu trúc chuẩn BEAT Body
        args.foot_indices = [10, 11, 24, 25] 
        args.bone_pairs = [
            (0, 1), (0, 2), (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 26),
            (1, 4), (4, 7), (7, 10), (10, 24),
            (2, 5), (5, 8), (8, 11), (11, 25),
            (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
            (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
        ]
        
        LH_START = 27  # Bắt đầu tay trái (sau 27 khớp body)
        RH_START = 42  # Bắt đầu tay phải (sau 27 body + 15 tay trái)
                
        args.hand_bone_pairs = [            
            # Ngón cái
            (22, LH_START+0), (LH_START+0, LH_START+1), (LH_START+1, LH_START+2),
            # Ngón trỏ
            (22, LH_START+3), (LH_START+3, LH_START+4), (LH_START+4, LH_START+5),
            # Ngón giữa
            (22, LH_START+6), (LH_START+6, LH_START+7), (LH_START+7, LH_START+8),
            # Ngón áp út
            (22, LH_START+9), (LH_START+9, LH_START+10), (LH_START+10, LH_START+11),
            # Ngón út
            (22, LH_START+12), (LH_START+12, LH_START+13), (LH_START+13, LH_START+14),

            # Ngón cái
            (23, RH_START+0), (RH_START+0, RH_START+1), (RH_START+1, RH_START+2),
            # Ngón trỏ
            (23, RH_START+3), (RH_START+3, RH_START+4), (RH_START+4, RH_START+5),
            # Ngón giữa
            (23, RH_START+6), (RH_START+6, RH_START+7), (RH_START+7, RH_START+8),
            # Ngón áp út
            (23, RH_START+9), (RH_START+9, RH_START+10), (RH_START+10, RH_START+11),
            # Ngón út
            (23, RH_START+12), (RH_START+12, RH_START+13), (RH_START+13, RH_START+14),
        ]
        
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")
    
    # Load stats
    stats_file = pjoin(ROOT, 'global_pipeline.pkl')
    pipeline = joblib.load(stats_file)
    scaler = pipeline.named_steps['stdscale']
    args.mean = scaler.data_mean_
    args.std = scaler.data_std_

    # Load hand & body indices - ADD DEBUG PRINT
    hand_indices_path = pjoin(args.data_id, 'hand_indices.npy')
    body_indices_path = pjoin(args.data_id, 'body_indices.npy')

    if os.path.exists(hand_indices_path) and os.path.exists(body_indices_path):
        hand_indices = np.load(hand_indices_path)
        body_indices = np.load(body_indices_path)
        
        # FIX: Convert from feature-level to joint-level indices
        # Feature format: 264-D = 88 joints * 3 coords
        # If hand_indices contains feature indices (0-263), convert to joint indices (0-87)
        if hand_indices.max() >= 88:
            print(f"[INFO] Converting hand_indices from feature-level to joint-level...")
            print(f"  Before: min={hand_indices.min()}, max={hand_indices.max()}")
            hand_indices = np.unique(hand_indices // 3)  # Convert and remove duplicates
            print(f"  After: min={hand_indices.min()}, max={hand_indices.max()}")
        
        if body_indices.max() >= 88:
            print(f"[INFO] Converting body_indices from feature-level to joint-level...")
            body_indices = np.unique(body_indices // 3)
    
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

    diffusion.hand_indices = hand_indices
    diffusion.body_indices = body_indices
    
    trainer = VQKLDiffusionTrainer(
        args=args,
        model=model,
        diffusion=diffusion,
        data_loader=train_loader,
        val_loader=val_loader
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()