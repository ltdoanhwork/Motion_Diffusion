"""
Enhanced Trainer for VQ-KL Autoencoder with:
- GAN + Hierarchical Weighted Loss
- Sobolev Regularization (Velocity + Acceleration)
- Spectral Loss (Frequency Domain)
- Geometric Consistency (Bone Length Preservation)
- Cosine Annealing LR Scheduler

Compatible with train_vq_with_discriminator.py
"""
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
import os

try:
    from trainers.motion_vqperceptual import VQLPIPSWithDiscriminator
except ImportError:
    print("[WARNING] Using taming LPIPS - may cause dimension mismatch for motion data")
    from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator


def def_value():
    return 0.0


class RVQTokenizerTrainerGAN:
    """
    State-of-the-art VQ-KL Trainer with:
    - Hierarchical Weighted Loss (Hand vs Body)
    - Sobolev Regularization (Velocity + Acceleration)
    - Spectral Loss (Frequency Domain via FFT)
    - Geometric Consistency (Bone Length Preservation)
    - GAN + Perceptual Loss
    - Cosine Annealing LR Scheduling
    """
    
    def __init__(self, args, vq_model, 
                 kl_weight=0.0, 
                 use_posterior_sample=True,
                 # GAN parameters
                 disc_start=10000,
                 disc_weight=0.75,
                 disc_factor=1.0,
                 perceptual_weight=0.1,
                 disc_loss="hinge",
                 disc_num_layers=2,
                 disc_in_channels=264,
                 # Hierarchical Loss parameters
                 hand_indices=None,
                 body_indices=None,
                 hand_loss_weight=2.0,
                 # Sobolev Loss parameters
                 lambda_vel=0.5,
                 lambda_acc=0.5,
                 # Spectral Loss parameters
                 lambda_spectral=0.1,
                 # Geometric Loss parameters
                 bone_pairs=None,
                 lambda_bone=0.1):
        """
        Args:
            hand_indices: torch.Tensor of shape (N_hand,) containing indices of hand features
            body_indices: torch.Tensor of shape (N_body,) containing indices of body features
            hand_loss_weight: α weight for hand reconstruction (recommended 2-10)
            lambda_vel: Weight for velocity loss (L_vel)
            lambda_acc: Weight for acceleration loss (L_acc)
            lambda_spectral: Weight for spectral/frequency loss
            bone_pairs: List of tuples [(start_idx, end_idx), ...] defining bone connections
            lambda_bone: Weight for bone length consistency loss
        """
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        
        # VQ-KL parameters
        self.kl_weight = kl_weight
        self.use_posterior_sample = use_posterior_sample
        self.is_vqkl = kl_weight > 0.0
        
        # GAN parameters
        self.disc_start = disc_start
        self.use_gan = disc_weight > 0.0
        
        # ==================== Hierarchical Loss Setup ====================
        self.hand_indices = hand_indices
        self.body_indices = body_indices
        self.hand_loss_weight = hand_loss_weight
        self.spatial_mask = None
        
        if hand_indices is not None:
            print(f"\n[INFO] Hierarchical Loss Configuration:")
            print(f"  - Hand indices: {len(hand_indices)} features")
            print(f"  - Hand weight (α): {hand_loss_weight}")
            print(f"  - Body weight: 1.0")
        
        # ==================== Sobolev Loss Setup ====================
        self.lambda_vel = lambda_vel
        self.lambda_acc = lambda_acc
        print(f"\n[INFO] Sobolev Regularization:")
        print(f"  - Velocity weight (λ_vel): {lambda_vel}")
        print(f"  - Acceleration weight (λ_acc): {lambda_acc}")
        
        # ==================== Spectral Loss Setup ====================
        self.lambda_spectral = lambda_spectral
        print(f"\n[INFO] Spectral Regularization:")
        print(f"  - Spectral weight (λ_spec): {lambda_spectral}")
        
        # ==================== Geometric Loss Setup ====================
        self.bone_pairs = bone_pairs
        self.lambda_bone = lambda_bone
        if bone_pairs is not None:
            print(f"\n[INFO] Geometric Consistency:")
            print(f"  - Number of bones: {len(bone_pairs)}")
            print(f"  - Bone length weight (λ_bone): {lambda_bone}")
        
        print(f"\n[INFO] Trainer Configuration:")
        print(f"  - VQ-KL mode: {self.is_vqkl}")
        if self.is_vqkl:
            print(f"    • KL weight: {kl_weight}")
            print(f"    • Posterior sampling: {use_posterior_sample}")
        print(f"  - GAN mode: {self.use_gan}")
        if self.use_gan:
            print(f"    • Disc starts at iter: {disc_start}")
            print(f"    • Disc weight: {disc_weight}")

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            
            # Base reconstruction loss
            if args.recons_loss == 'l1':
                self.base_recons_criterion = torch.nn.L1Loss(reduction='none')
            elif args.recons_loss == 'l1_smooth':
                self.base_recons_criterion = torch.nn.SmoothL1Loss(reduction='none')
            elif args.recons_loss == 'l2':
                self.base_recons_criterion = torch.nn.MSELoss(reduction='none')
            else:
                raise ValueError(f"Unknown recons_loss: {args.recons_loss}")
            
            # Initialize GAN loss
            if self.use_gan:
                self.loss_fn_gan = VQLPIPSWithDiscriminator(
                    disc_start=disc_start,
                    codebook_weight=1.0,
                    pixelloss_weight=1.0,
                    disc_num_layers=disc_num_layers,
                    disc_in_channels=disc_in_channels,
                    disc_factor=disc_factor,
                    disc_weight=disc_weight,
                    perceptual_weight=perceptual_weight,
                    use_actnorm=False,
                    disc_conditional=False,
                    disc_ndf=64,
                    disc_loss=disc_loss
                ).to(self.device)
                
                print(f"[INFO] ✓ VQLPIPSWithDiscriminator initialized")

    def create_spatial_mask(self, motion_shape, device):
        """
        Creates spatial weight mask M where M_hand = α, M_body = 1
        
        Args:
            motion_shape: Shape of motion tensor (B, T, D)
            device: torch device
            
        Returns:
            mask: Tensor of shape (1, 1, D) for broadcasting
        """
        if self.spatial_mask is not None:
            return self.spatial_mask
        
        D = motion_shape[-1]
        mask = torch.ones(D, device=device, dtype=torch.float32)
        
        if self.hand_indices is not None:
            mask[self.hand_indices] = self.hand_loss_weight
        
        self.spatial_mask = mask.view(1, 1, -1)
        return self.spatial_mask

    def hierarchical_reconstruction_loss(self, pred_motion, gt_motion):
        """
        Computes masked MSE loss with hierarchical weights
        L_rec = || M ⊙ (x_pred - x_gt) ||²
        """
        mask = self.create_spatial_mask(gt_motion.shape, gt_motion.device)
        diff = pred_motion - gt_motion
        weighted_diff = mask * diff
        loss = torch.mean(weighted_diff ** 2)
        return loss

    def sobolev_regularization(self, pred_motion, gt_motion):
        """
        Computes Sobolev regularization: velocity + acceleration penalties
        
        L_vel = || ∂x_pred/∂t - ∂x_gt/∂t ||²
        L_acc = || ∂²x_pred/∂t² - ∂²x_gt/∂t² ||²
        """
        time_dim = 1
        
        # Velocity (First-order derivative)
        vel_pred = torch.diff(pred_motion, dim=time_dim)
        vel_gt = torch.diff(gt_motion, dim=time_dim)
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        
        # Acceleration (Second-order derivative)
        acc_pred = torch.diff(vel_pred, dim=time_dim)
        acc_gt = torch.diff(vel_gt, dim=time_dim)
        loss_acc = F.mse_loss(acc_pred, acc_gt)
        
        return loss_vel, loss_acc

    def spectral_loss(self, pred_motion, gt_motion):
        """
        Computes spectral loss in frequency domain using FFT
        
        This captures global frequency characteristics and penalizes
        high-frequency jitter that Sobolev might miss.
        
        Args:
            pred_motion: (B, T, D) predicted motion
            gt_motion: (B, T, D) ground truth motion
            
        Returns:
            loss_spectral: Spectral loss value
        """
        # Apply FFT along time dimension (dim=1)
        # rfft returns complex numbers for real input
        fft_pred = torch.fft.rfft(pred_motion, dim=1)
        fft_gt = torch.fft.rfft(gt_motion, dim=1)
        
        # Compute magnitude (amplitude) spectrum
        mag_pred = torch.abs(fft_pred)
        mag_gt = torch.abs(fft_gt)
        
        # L1 loss on magnitudes (more robust than L2 for frequency domain)
        loss_spectral = F.l1_loss(mag_pred, mag_gt)
        
        return loss_spectral

    def bone_length_loss(self, pred_motion, gt_motion):
        """
        Computes bone length consistency loss
        
        Ensures that the distance between connected joints remains constant,
        preventing unnatural bone stretching/shrinking.
        
        Args:
            pred_motion: (B, T, D) predicted motion (D should be divisible by 3 for xyz)
            gt_motion: (B, T, D) ground truth motion
            
        Returns:
            loss_bone: Bone length consistency loss
        """
        if self.bone_pairs is None:
            return torch.tensor(0.0, device=pred_motion.device)
        
        B, T, D = pred_motion.shape
        
        # Assume D contains XYZ coordinates: reshape to (B, T, J, 3)
        # where J = D // 3 (number of joints)
        if D % 3 != 0:
            # If not divisible by 3, skip bone loss
            return torch.tensor(0.0, device=pred_motion.device)
        
        J = D // 3
        pred_joints = pred_motion.reshape(B, T, J, 3)
        gt_joints = gt_motion.reshape(B, T, J, 3)
        
        bone_losses = []
        
        for start_idx, end_idx in self.bone_pairs:
            if start_idx >= J or end_idx >= J:
                continue
                
            # Compute bone vectors
            pred_bone = pred_joints[:, :, end_idx, :] - pred_joints[:, :, start_idx, :]
            gt_bone = gt_joints[:, :, end_idx, :] - gt_joints[:, :, start_idx, :]
            
            # Compute bone lengths
            pred_length = torch.norm(pred_bone, dim=-1)  # (B, T)
            gt_length = torch.norm(gt_bone, dim=-1)      # (B, T)
            
            # Penalize length difference
            bone_loss = F.mse_loss(pred_length, gt_length)
            bone_losses.append(bone_loss)
        
        if len(bone_losses) == 0:
            return torch.tensor(0.0, device=pred_motion.device)
        
        loss_bone = torch.stack(bone_losses).mean()
        return loss_bone

    def compute_total_reconstruction_loss(self, pred_motion, gt_motion):
        """
        Computes comprehensive reconstruction loss combining:
        1. Hierarchical weighted MSE
        2. Sobolev regularization (velocity + acceleration)
        3. Spectral loss (frequency domain)
        4. Bone length consistency
        
        L_total = L_rec + λ_vel*L_vel + λ_acc*L_acc + λ_spec*L_spec + λ_bone*L_bone
        """
        # 1. Hierarchical Reconstruction Loss
        loss_rec = self.hierarchical_reconstruction_loss(pred_motion, gt_motion)
        
        # 2. Sobolev Regularization
        loss_vel, loss_acc = self.sobolev_regularization(pred_motion, gt_motion)
        
        # 3. Spectral Loss
        loss_spectral = self.spectral_loss(pred_motion, gt_motion)
        
        # 4. Bone Length Loss
        loss_bone = self.bone_length_loss(pred_motion, gt_motion)
        
        # 5. Combine losses
        loss_total = (loss_rec + 
                     self.lambda_vel * loss_vel + 
                     self.lambda_acc * loss_acc + 
                     self.lambda_spectral * loss_spectral +
                     self.lambda_bone * loss_bone)
        
        return loss_total, loss_rec, loss_vel, loss_acc, loss_spectral, loss_bone

    def forward(self, batch_data, global_step=0, optimizer_idx=0):
        """
        Forward pass with all regularizations
        """
        motions = batch_data[1].detach().to(self.device).float()
        
        # ==================== Forward Pass ====================
        if self.is_vqkl and hasattr(self.vq_model, 'encode'):
            # VQ-KL Mode
            posterior = self.vq_model.encode(motions)
            z = posterior.sample() if self.use_posterior_sample else posterior.mode()
            pred_motion = self.vq_model.decode(z)
            
            # VQ loss
            if hasattr(self.vq_model, 'quantize'):
                _, loss_vq, perplexity_info = self.vq_model.quantize(z)
                if isinstance(perplexity_info, tuple):
                    perplexity = perplexity_info[2]
                    perplexity = torch.tensor(len(torch.unique(perplexity)),
                                            device=self.device, dtype=torch.float32)
                else:
                    perplexity = torch.tensor(0.0, device=self.device)
            else:
                loss_vq = torch.tensor(0.0, device=self.device)
                perplexity = torch.tensor(0.0, device=self.device)
            
            # KL loss
            if hasattr(posterior, 'kl'):
                loss_kl = posterior.kl().mean()
            else:
                mean = posterior.mean if hasattr(posterior, 'mean') else posterior.mode()
                logvar = posterior.logvar if hasattr(posterior, 'logvar') else torch.zeros_like(mean)
                loss_kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / mean.shape[0]
        else:
            # Standard VQ-VAE Mode
            pred_motion, loss_vq, perplexity = self.vq_model(motions)
            loss_kl = torch.tensor(0.0, device=self.device)
        
        self.motions = motions
        self.pred_motion = pred_motion
        
        # ==================== Enhanced Loss Computation ====================
        
        # Compute all regularization losses
        (loss_total_rec, loss_rec, loss_vel, loss_acc, 
         loss_spectral, loss_bone) = self.compute_total_reconstruction_loss(
            pred_motion, motions
        )
        
        # VQ Commitment Loss
        loss_commit = loss_vq
        
        # ==================== GAN Loss ====================
        if self.use_gan:
            motions_disc = motions.permute(0, 2, 1)
            pred_motion_disc = pred_motion.permute(0, 2, 1)
            
            try:
                last_layer = self.vq_model.decoder.conv_out.weight
            except:
                last_layer = None
            
            gan_loss, gan_log = self.loss_fn_gan(
                codebook_loss=loss_commit,
                inputs=motions_disc,
                reconstructions=pred_motion_disc,
                optimizer_idx=optimizer_idx,
                global_step=global_step,
                last_layer=last_layer,
                cond=None,
                split="train"
            )
            
            if optimizer_idx == 0:
                # Generator update
                loss_gan_g = gan_log.get('train/g_loss', torch.tensor(0.0))
                loss_perceptual = gan_log.get('train/p_loss', torch.tensor(0.0))
                disc_weight = gan_log.get('train/d_weight', torch.tensor(0.0))
                disc_factor = gan_log.get('train/disc_factor', torch.tensor(0.0))
                
                loss = (loss_total_rec + 
                       disc_weight * disc_factor * loss_gan_g + 
                       self.opt.commit * loss_commit)
                
                if self.is_vqkl:
                    loss = loss + self.kl_weight * loss_kl
                
                return (loss, loss_rec, loss_vel, loss_acc, loss_spectral, loss_bone,
                       loss_commit, perplexity, loss_kl, loss_gan_g, 
                       loss_perceptual, disc_weight)
            
            else:
                # Discriminator update
                return (gan_loss, torch.tensor(0.0), torch.tensor(0.0), 
                       torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),
                       torch.tensor(0.0), perplexity, torch.tensor(0.0),
                       torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        
        else:
            # No GAN
            loss = (loss_total_rec + self.opt.commit * loss_commit)
            
            if self.is_vqkl:
                loss = loss + self.kl_weight * loss_kl
            
            return (loss, loss_rec, loss_vel, loss_acc, loss_spectral, loss_bone,
                   loss_commit, perplexity, loss_kl, torch.tensor(0.0), 
                   torch.tensor(0.0), torch.tensor(0.0))

    def save(self, file_name, ep, total_it):
        """Save checkpoint"""
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler_g": self.scheduler_g.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        
        if self.use_gan:
            state["discriminator"] = self.loss_fn_gan.discriminator.state_dict()
            state["opt_disc"] = self.opt_disc.state_dict()
            state["scheduler_d"] = self.scheduler_d.state_dict()
        
        torch.save(state, file_name)

    def resume(self, model_dir):
        """Resume from checkpoint"""
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        
        if self.use_gan and 'discriminator' in checkpoint:
            self.loss_fn_gan.discriminator.load_state_dict(checkpoint['discriminator'])
            self.opt_disc.load_state_dict(checkpoint['opt_disc'])
            self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        self.vq_model.to(self.device)

        # Generator optimizer
        self.opt_vq_model = optim.AdamW(
            self.vq_model.parameters(),
            lr=self.opt.lr,
            betas=(0.9, 0.99),
            weight_decay=self.opt.weight_decay
        )
        
        # Cosine Annealing LR Scheduler for Generator
        self.scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_vq_model,
            T_max=self.opt.max_epoch,
            eta_min=self.opt.lr * 0.01  # Minimum LR = 1% of initial
        )
        
        # Discriminator optimizer
        if self.use_gan:
            lr_disc = getattr(self.opt, 'lr_disc', self.opt.lr)
            self.opt_disc = optim.AdamW(
                self.loss_fn_gan.discriminator.parameters(),
                lr=lr_disc,
                betas=(0.9, 0.99),
                weight_decay=self.opt.weight_decay
            )
            self.scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.opt_disc,
                T_max=self.opt.max_epoch,
                eta_min=lr_disc * 0.01
            )

        epoch = 0
        it = 0
        min_val_loss = np.inf

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print(f"[INFO] Resumed from epoch {epoch}, iteration {it}")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'\n{"="*80}')
        print(f'🚀 Enhanced VQ-KL Trainer Started')
        print(f'{"="*80}')
        print(f'Total Epochs: {self.opt.max_epoch}')
        print(f'Total Iterations: {total_iters}')
        print(f'Loss Components:')
        print(f'  - Hierarchical Rec (hand α={self.hand_loss_weight})')
        print(f'  - Sobolev (λ_vel={self.lambda_vel}, λ_acc={self.lambda_acc})')
        print(f'  - Spectral (λ_spec={self.lambda_spectral})')
        print(f'  - Bone Length (λ_bone={self.lambda_bone})')
        if self.use_gan:
            print(f'  - GAN (starts at iter {self.disc_start})')
        print(f'{"="*80}\n')
        
        logs = defaultdict(def_value, OrderedDict())

        # ==================== Training Loop ====================
        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            if self.use_gan:
                self.loss_fn_gan.discriminator.train()
            
            for i, batch_data in enumerate(train_loader):
                it += 1
                
                # ========== Generator Update ==========
                results = self.forward(batch_data, global_step=it, optimizer_idx=0)
                (loss, loss_rec, loss_vel, loss_acc, loss_spectral, loss_bone,
                 loss_commit, perplexity, loss_kl, loss_gan_g, 
                 loss_perceptual, disc_weight) = results
                
                self.opt_vq_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vq_model.parameters(), max_norm=1.0)
                self.opt_vq_model.step()
                
                # ========== Discriminator Update ==========
                if self.use_gan and it >= self.disc_start:
                    disc_results = self.forward(batch_data, global_step=it, optimizer_idx=1)
                    loss_disc = disc_results[0]
                    
                    self.opt_disc.zero_grad()
                    loss_disc.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.loss_fn_gan.discriminator.parameters(), max_norm=1.0
                    )
                    self.opt_disc.step()
                    
                    logs['loss_disc'] += loss_disc.item()
                
                # ==================== Logging ====================
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_acc'] += loss_acc.item()
                logs['loss_spectral'] += loss_spectral.item()
                logs['loss_bone'] += loss_bone.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
                
                if self.is_vqkl:
                    logs['loss_kl'] += loss_kl.item()
                
                if self.use_gan:
                    logs['loss_gan_g'] += loss_gan_g.item()
                    logs['loss_perceptual'] += loss_perceptual.item()
                    logs['disc_weight'] += disc_weight.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    
                    for tag, value in logs.items():
                        self.logger.add_scalar(
                            'Train/%s' % tag,
                            value / self.opt.log_every,
                            it
                        )
                        mean_loss[tag] = value / self.opt.log_every
                    
                    logs = defaultdict(def_value, OrderedDict())
                    
                    # Console output
                    log_str = (f"Ep: {epoch:03d} | It: {it:06d} | "
                              f"Loss: {mean_loss['loss']:.4f} | "
                              f"Rec: {mean_loss['loss_rec']:.4f} | "
                              f"Vel: {mean_loss['loss_vel']:.4f} | "
                              f"Acc: {mean_loss['loss_acc']:.4f} | "
                              f"Spec: {mean_loss['loss_spectral']:.4f} | "
                              f"Bone: {mean_loss['loss_bone']:.4f}")
                    
                    if self.is_vqkl:
                        log_str += f" | KL: {mean_loss['loss_kl']:.6f}"
                    
                    if self.use_gan and it >= self.disc_start:
                        log_str += (f" | G: {mean_loss['loss_gan_g']:.4f} | "
                                   f"D: {mean_loss.get('loss_disc', 0.0):.4f}")
                    
                    print(log_str)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            
            # Step LR schedulers at epoch end
            self.scheduler_g.step()
            if self.use_gan:
                self.scheduler_d.step()
            
            epoch += 1

            # ==================== Validation ====================
            if val_loader is not None:
                print(f'\n{"="*80}')
                print(f'Validation - Epoch {epoch}')
                print(f'{"="*80}')
                
                self.vq_model.eval()
                if self.use_gan:
                    self.loss_fn_gan.discriminator.eval()
                
                val_metrics = defaultdict(list)

                with torch.no_grad():
                    for batch_data in val_loader:
                        results = self.forward(batch_data, global_step=it, optimizer_idx=0)
                        (loss, loss_rec, loss_vel, loss_acc, loss_spectral, loss_bone,
                         loss_commit, perplexity, loss_kl, _, _, _) = results
                        
                        val_metrics['loss'].append(loss.item())
                        val_metrics['loss_rec'].append(loss_rec.item())
                        val_metrics['loss_vel'].append(loss_vel.item())
                        val_metrics['loss_acc'].append(loss_acc.item())
                        val_metrics['loss_spectral'].append(loss_spectral.item())
                        val_metrics['loss_bone'].append(loss_bone.item())
                        val_metrics['loss_commit'].append(loss_commit.item())
                        val_metrics['perplexity'].append(perplexity.item())
                        
                        if self.is_vqkl:
                            val_metrics['loss_kl'].append(loss_kl.item())

                # Log validation metrics
                for key, values in val_metrics.items():
                    avg = np.mean(values)
                    self.logger.add_scalar(f'Val/{key}', avg, epoch)
                
                avg_loss = np.mean(val_metrics['loss'])
                
                val_log = (f'Val Loss: {avg_loss:.5f} | '
                          f'Rec: {np.mean(val_metrics["loss_rec"]):.5f} | '
                          f'Vel: {np.mean(val_metrics["loss_vel"]):.5f} | '
                          f'Acc: {np.mean(val_metrics["loss_acc"]):.5f} | '
                          f'Spec: {np.mean(val_metrics["loss_spectral"]):.5f} | '
                          f'Bone: {np.mean(val_metrics["loss_bone"]):.5f}')
                
                if self.is_vqkl:
                    val_log += f' | KL: {np.mean(val_metrics["loss_kl"]):.6f}'
                
                print(val_log)

                # Save best model
                if avg_loss < min_val_loss:
                    print(f'\n🎉 NEW BEST: {min_val_loss:.5f} → {avg_loss:.5f}\n')
                    min_val_loss = avg_loss
                    self.save(pjoin(self.opt.model_dir, 'best_model.tar'), epoch, it)
                
                print(f'{"="*80}\n')

        # ==================== Training Complete ====================
        elapsed = time.time() - start_time
        print(f'\n{"="*80}')
        print(f'✅ Training Complete!')
        print(f'{"="*80}')
        print(f'Total Time: {elapsed/3600:.2f} hours')
        print(f'Final Epoch: {epoch}')
        print(f'Total Iterations: {it}')
        print(f'Best Val Loss: {min_val_loss:.5f}')
        print(f'{"="*80}\n')


# ==================== Utility Functions ====================

def get_smpl_bone_pairs():
    """
    Returns standard SMPL skeleton bone pairs for geometric consistency
    
    Returns:
        List of tuples [(parent_joint_idx, child_joint_idx), ...]
    """
    # SMPL 24 joint skeleton topology
    # Joint indices based on standard SMPL model
    bone_pairs = [
        # Spine
        (0, 1),   # Pelvis -> L_Hip
        (0, 2),   # Pelvis -> R_Hip
        (0, 3),   # Pelvis -> Spine1
        (3, 6),   # Spine1 -> Spine2
        (6, 9),   # Spine2 -> Spine3
        (9, 12),  # Spine3 -> Neck
        (12, 15), # Neck -> Head
        
        # Left Leg
        (1, 4),   # L_Hip -> L_Knee
        (4, 7),   # L_Knee -> L_Ankle
        (7, 10),  # L_Ankle -> L_Foot
        
        # Right Leg
        (2, 5),   # R_Hip -> R_Knee
        (5, 8),   # R_Knee -> R_Ankle
        (8, 11),  # R_Ankle -> R_Foot
        
        # Left Arm
        (9, 13),  # Spine3 -> L_Collar
        (13, 16), # L_Collar -> L_Shoulder
        (16, 18), # L_Shoulder -> L_Elbow
        (18, 20), # L_Elbow -> L_Wrist
        (20, 22), # L_Wrist -> L_Hand
        
        # Right Arm
        (9, 14),  # Spine3 -> R_Collar
        (14, 17), # R_Collar -> R_Shoulder
        (17, 19), # R_Shoulder -> R_Elbow
        (19, 21), # R_Elbow -> R_Wrist
        (21, 23), # R_Wrist -> R_Hand
    ]
    
    return bone_pairs


def get_hand_joint_indices(skeleton_type='smpl'):
    """
    Returns indices of hand-related joints for hierarchical weighting
    
    Args:
        skeleton_type: Type of skeleton ('smpl', 'cmu', etc.)
    
    Returns:
        torch.Tensor: Indices of hand joints (for xyz coordinates, multiply by 3)
    """
    if skeleton_type == 'smpl':
        # SMPL hand joints (wrists + hands)
        hand_joints = [20, 21, 22, 23]  # L_Wrist, R_Wrist, L_Hand, R_Hand
    elif skeleton_type == 'cmu':
        # CMU MoCap format (approximate)
        hand_joints = [9, 10, 11, 12, 24, 25, 26, 27]
    else:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}")
    
    # Convert to feature indices (each joint has x, y, z)
    hand_indices = []
    for joint_idx in hand_joints:
        hand_indices.extend([joint_idx * 3, joint_idx * 3 + 1, joint_idx * 3 + 2])
    
    return torch.tensor(hand_indices, dtype=torch.long)


def create_trainer_from_config(config, vq_model):
    """
    Factory function to create trainer from configuration
    
    Args:
        config: Configuration object/dict with training parameters
        vq_model: The VQ model to train
    
    Returns:
        RVQTokenizerTrainerGAN instance
    """
    # Get bone pairs for geometric consistency
    bone_pairs = None
    if hasattr(config, 'use_bone_loss') and config.use_bone_loss:
        skeleton_type = getattr(config, 'skeleton_type', 'smpl')
        if skeleton_type == 'smpl':
            bone_pairs = get_smpl_bone_pairs()
        print(f"[INFO] Using {len(bone_pairs)} bone pairs for geometric consistency")
    
    # Get hand indices for hierarchical loss
    hand_indices = None
    if hasattr(config, 'use_hierarchical_loss') and config.use_hierarchical_loss:
        skeleton_type = getattr(config, 'skeleton_type', 'smpl')
        hand_indices = get_hand_joint_indices(skeleton_type)
        print(f"[INFO] Using {len(hand_indices)} hand features for hierarchical weighting")
    
    # Create trainer
    trainer = RVQTokenizerTrainerGAN(
        args=config,
        vq_model=vq_model,
        kl_weight=getattr(config, 'kl_weight', 0.0),
        use_posterior_sample=getattr(config, 'use_posterior_sample', True),
        # GAN parameters
        disc_start=getattr(config, 'disc_start', 10000),
        disc_weight=getattr(config, 'disc_weight', 0.75),
        disc_factor=getattr(config, 'disc_factor', 1.0),
        perceptual_weight=getattr(config, 'perceptual_weight', 0.1),
        disc_loss=getattr(config, 'disc_loss', 'hinge'),
        disc_num_layers=getattr(config, 'disc_num_layers', 2),
        disc_in_channels=getattr(config, 'disc_in_channels', 264),
        # Hierarchical loss
        hand_indices=hand_indices,
        hand_loss_weight=getattr(config, 'hand_loss_weight', 2.0),
        # Sobolev regularization
        lambda_vel=getattr(config, 'lambda_vel', 0.5),
        lambda_acc=getattr(config, 'lambda_acc', 0.5),
        # Spectral loss
        lambda_spectral=getattr(config, 'lambda_spectral', 0.1),
        # Geometric consistency
        bone_pairs=bone_pairs,
        lambda_bone=getattr(config, 'lambda_bone', 0.1),
    )
    
    return trainer