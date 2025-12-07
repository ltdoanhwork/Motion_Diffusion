"""
Trainer for VQ-KL Autoencoder with GAN + Hierarchical Weighted Loss + Sobolev Regularization
Fully implements spatial masking for hands and temporal smoothness constraints
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
import sys

try:
    from trainers.motion_vqperceptual import VQLPIPSWithDiscriminator
except ImportError:
    print("[WARNING] Using taming LPIPS - may cause dimension mismatch for motion data")
    from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator


def def_value():
    return 0.0


class RVQTokenizerTrainerGAN:
    """
    Enhanced VQ-KL Trainer with:
    - Hierarchical Weighted Loss (Hand vs Body)
    - Sobolev Regularization (Velocity + Acceleration)
    - GAN + Perceptual Loss
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
                 lambda_acc=0.5):
        """
        Args:
            hand_indices: torch.Tensor of shape (N_hand,) containing indices of hand features
            body_indices: torch.Tensor of shape (N_body,) containing indices of body features
            hand_loss_weight: α weight for hand reconstruction (recommended 2-10)
            lambda_vel: Weight for velocity loss (L_vel)
            lambda_acc: Weight for acceleration loss (L_acc)
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
        
        # Create spatial weight mask M
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
        
        print(f"\n[INFO] Trainer Configuration:")
        print(f"  - VQ-KL mode: {self.is_vqkl}")
        if self.is_vqkl:
            print(f"    • KL weight: {kl_weight}")
            print(f"    • Posterior sampling: {use_posterior_sample}")
        print(f"  - GAN mode: {self.use_gan}")
        if self.use_gan:
            print(f"    • Disc starts at iter: {disc_start}")
            print(f"    • Disc weight: {disc_weight}")
            print(f"    • Perceptual weight: {perceptual_weight}")
            print(f"    • Disc loss: {disc_loss}")

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            
            # Base reconstruction loss (used in GAN)
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
        
        # motion_shape is (B, T, D)
        D = motion_shape[-1]
        
        # Initialize mask with ones
        mask = torch.ones(D, device=device, dtype=torch.float32)
        
        # Apply hand weight
        if self.hand_indices is not None:
            mask[self.hand_indices] = self.hand_loss_weight
        
        # Reshape for broadcasting: (1, 1, D)
        self.spatial_mask = mask.view(1, 1, -1)
        
        return self.spatial_mask

    def hierarchical_reconstruction_loss(self, pred_motion, gt_motion):
        """
        Computes masked MSE loss with hierarchical weights
        
        L_rec = || M ⊙ (x_pred - x_gt) ||²
        
        where M_hand = α, M_body = 1
        """
        # Create mask if not exists
        mask = self.create_spatial_mask(gt_motion.shape, gt_motion.device)
        
        # Compute weighted MSE
        diff = pred_motion - gt_motion
        weighted_diff = mask * diff
        loss = torch.mean(weighted_diff ** 2)
        
        return loss

    def sobolev_regularization(self, pred_motion, gt_motion):
        """
        Computes Sobolev regularization: velocity + acceleration penalties
        
        L_vel = || ∂x_pred/∂t - ∂x_gt/∂t ||²
        L_acc = || ∂²x_pred/∂t² - ∂²x_gt/∂t² ||²
        
        Args:
            pred_motion: (B, T, D) predicted motion
            gt_motion: (B, T, D) ground truth motion
            
        Returns:
            loss_vel: Velocity loss
            loss_acc: Acceleration loss
        """
        # Time dimension is axis 1: (B, T, D)
        time_dim = 1
        
        # 1. Velocity (First-order derivative)
        # ∂x/∂t ≈ x[t+1] - x[t]
        vel_pred = torch.diff(pred_motion, dim=time_dim)
        vel_gt = torch.diff(gt_motion, dim=time_dim)
        loss_vel = F.mse_loss(vel_pred, vel_gt)
        
        # 2. Acceleration (Second-order derivative)
        # ∂²x/∂t² ≈ vel[t+1] - vel[t]
        acc_pred = torch.diff(vel_pred, dim=time_dim)
        acc_gt = torch.diff(vel_gt, dim=time_dim)
        loss_acc = F.mse_loss(acc_pred, acc_gt)
        
        return loss_vel, loss_acc

    def compute_total_reconstruction_loss(self, pred_motion, gt_motion):
        """
        Computes total reconstruction loss combining:
        1. Hierarchical weighted MSE
        2. Sobolev regularization (velocity + acceleration)
        
        L_total = L_rec + λ_vel * L_vel + λ_acc * L_acc
        """
        # 1. Hierarchical Reconstruction Loss
        loss_rec = self.hierarchical_reconstruction_loss(pred_motion, gt_motion)
        
        # 2. Sobolev Regularization
        loss_vel, loss_acc = self.sobolev_regularization(pred_motion, gt_motion)
        
        # 3. Combine losses
        loss_total = loss_rec + self.lambda_vel * loss_vel + self.lambda_acc * loss_acc
        
        return loss_total, loss_rec, loss_vel, loss_acc

    def forward(self, batch_data, global_step=0, optimizer_idx=0):
        """
        Forward pass with hierarchical + Sobolev losses
        """
        motions = batch_data[1].detach().to(self.device).float()
        
        # ==================== Forward Pass ====================
        if self.is_vqkl and hasattr(self.vq_model, 'encode'):
            # VQ-KL Mode
            posterior = self.vq_model.encode(motions)
            
            if self.use_posterior_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
            
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
        
        # 1. Hierarchical Reconstruction + Sobolev Loss
        loss_total_rec, loss_rec, loss_vel, loss_acc = self.compute_total_reconstruction_loss(
            pred_motion, motions
        )
        
        # 2. VQ Commitment Loss
        loss_commit = loss_vq
        
        # 3. Combined reconstruction loss for GAN
        nll_loss = loss_total_rec
        
        # ==================== GAN Loss ====================
        if self.use_gan:
            # Reshape for discriminator: (B, T, D) -> (B, D, T)
            motions_disc = motions.permute(0, 2, 1)
            pred_motion_disc = pred_motion.permute(0, 2, 1)
            
            # Get last layer for adaptive weighting
            try:
                last_layer = self.vq_model.decoder.conv_out.weight
            except:
                last_layer = None
            
            # Calculate GAN loss
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
                # Note: GAN loss already includes basic rec + perceptual + disc + codebook
                # We replace the basic rec with our hierarchical + Sobolev version
                
                # Extract GAN components
                loss_gan_g = gan_log.get('train/g_loss', torch.tensor(0.0))
                loss_perceptual = gan_log.get('train/p_loss', torch.tensor(0.0))
                disc_weight = gan_log.get('train/d_weight', torch.tensor(0.0))
                disc_factor = gan_log.get('train/disc_factor', torch.tensor(0.0))
                
                # Reconstruct total loss with our custom reconstruction loss
                # Original: loss = nll_loss + d_weight * disc_factor * g_loss + codebook_weight * codebook_loss
                # Our version: Replace nll_loss with loss_total_rec
                loss = (loss_total_rec + 
                       disc_weight * disc_factor * loss_gan_g + 
                       self.opt.commit * loss_commit)
                
                if self.is_vqkl:
                    loss = loss + self.kl_weight * loss_kl
                
                return (loss, loss_rec, loss_vel, loss_acc, loss_commit, perplexity, 
                       loss_kl, loss_gan_g, loss_perceptual, disc_weight)
            
            else:
                # Discriminator update
                return (gan_loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0),
                       torch.tensor(0.0), perplexity, torch.tensor(0.0),
                       torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        
        else:
            # No GAN - standard training with hierarchical + Sobolev
            loss = (loss_total_rec + self.opt.commit * loss_commit)
            
            if self.is_vqkl:
                loss = loss + self.kl_weight * loss_kl
            
            return (loss, loss_rec, loss_vel, loss_acc, loss_commit, perplexity, 
                   loss_kl, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        """Warm-up learning rate"""
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr
        if self.use_gan:
            for param_group in self.opt_disc.param_groups:
                param_group["lr"] = current_lr
        return current_lr

    def save(self, file_name, ep, total_it):
        """Save checkpoint"""
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        
        if self.use_gan:
            state["discriminator"] = self.loss_fn_gan.discriminator.state_dict()
            state["opt_disc"] = self.opt_disc.state_dict()
            state["scheduler_disc"] = self.scheduler_disc.state_dict()
        
        torch.save(state, file_name)

    def resume(self, model_dir):
        """Resume from checkpoint"""
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        if self.use_gan and 'discriminator' in checkpoint:
            self.loss_fn_gan.discriminator.load_state_dict(checkpoint['discriminator'])
            self.opt_disc.load_state_dict(checkpoint['opt_disc'])
            self.scheduler_disc.load_state_dict(checkpoint['scheduler_disc'])
        
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader=None, eval_val_loader=None,
              eval_wrapper=None, plot_eval=None):
        """Main training loop with hierarchical + Sobolev losses"""
        self.vq_model.to(self.device)

        # Generator optimizer
        self.opt_vq_model = optim.AdamW(
            self.vq_model.parameters(),
            lr=self.opt.lr,
            betas=(0.9, 0.99),
            weight_decay=self.opt.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt_vq_model,
            milestones=self.opt.milestones,
            gamma=self.opt.gamma
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
            self.scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(
                self.opt_disc,
                milestones=self.opt.milestones,
                gamma=self.opt.gamma
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
        print(f'\n{"="*70}')
        print(f'Total Epochs: {self.opt.max_epoch}')
        print(f'Total Iterations: {total_iters}')
        print(f'Discriminator starts at: {self.disc_start}')
        print(f'Hand weight (α): {self.hand_loss_weight}')
        print(f'Sobolev weights: λ_vel={self.lambda_vel}, λ_acc={self.lambda_acc}')
        print(f'{"="*70}\n')
        
        logs = defaultdict(def_value, OrderedDict())

        # ==================== Training Loop ====================
        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            if self.use_gan:
                self.loss_fn_gan.discriminator.train()
            
            for i, batch_data in enumerate(train_loader):
                it += 1
                
                # Warm-up
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(
                        it, self.opt.warm_up_iter, self.opt.lr
                    )
                
                # ========== Generator Update ==========
                results = self.forward(batch_data, global_step=it, optimizer_idx=0)
                (loss, loss_rec, loss_vel, loss_acc, loss_commit, perplexity,
                 loss_kl, loss_gan_g, loss_perceptual, disc_weight) = results
                
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
                
                # Update LR
                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                    if self.use_gan:
                        self.scheduler_disc.step()
                
                # ==================== Logging ====================
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_acc'] += loss_acc.item()
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
                              f"Acc: {mean_loss['loss_acc']:.4f}")
                    
                    if self.is_vqkl:
                        log_str += f" | KL: {mean_loss['loss_kl']:.6f}"
                    
                    if self.use_gan and it >= self.disc_start:
                        log_str += (f" | G: {mean_loss['loss_gan_g']:.4f} | "
                                   f"D: {mean_loss.get('loss_disc', 0.0):.4f} | "
                                   f"Perc: {mean_loss['loss_perceptual']:.4f}")
                    
                    print(log_str)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            # ==================== Validation ====================
            if val_loader is not None:
                print(f'\n{"="*70}')
                print(f'Validation - Epoch {epoch}')
                print(f'{"="*70}')
                
                self.vq_model.eval()
                if self.use_gan:
                    self.loss_fn_gan.discriminator.eval()
                
                val_metrics = defaultdict(list)

                with torch.no_grad():
                    for batch_data in val_loader:
                        results = self.forward(batch_data, global_step=it, optimizer_idx=0)
                        (loss, loss_rec, loss_vel, loss_acc, loss_commit, perplexity,
                         loss_kl, loss_gan_g, loss_perceptual, _) = results
                        
                        val_metrics['loss'].append(loss.item())
                        val_metrics['loss_rec'].append(loss_rec.item())
                        val_metrics['loss_vel'].append(loss_vel.item())
                        val_metrics['loss_acc'].append(loss_acc.item())
                        val_metrics['loss_commit'].append(loss_commit.item())
                        val_metrics['perplexity'].append(perplexity.item())
                        
                        if self.is_vqkl:
                            val_metrics['loss_kl'].append(loss_kl.item())
                        if self.use_gan:
                            val_metrics['loss_perceptual'].append(loss_perceptual.item())

                # Log averages
                for key, values in val_metrics.items():
                    avg = np.mean(values)
                    self.logger.add_scalar(f'Val/{key}', avg, epoch)
                
                avg_loss = np.mean(val_metrics['loss'])
                
                val_log = (f'Val Loss: {avg_loss:.5f} | '
                          f'Rec: {np.mean(val_metrics["loss_rec"]):.5f} | '
                          f'Vel: {np.mean(val_metrics["loss_vel"]):.5f} | '
                          f'Acc: {np.mean(val_metrics["loss_acc"]):.5f}')
                
                if self.is_vqkl:
                    val_log += f' | KL: {np.mean(val_metrics["loss_kl"]):.6f}'
                if self.use_gan:
                    val_log += f' | Perc: {np.mean(val_metrics["loss_perceptual"]):.5f}'
                
                print(val_log)

                if avg_loss < min_val_loss:
                    print(f'\n🎉 NEW BEST: {min_val_loss:.5f} → {avg_loss:.5f}\n')
                    min_val_loss = avg_loss
                    self.save(pjoin(self.opt.model_dir, 'best_model.tar'), epoch, it)
                
                print(f'{"="*70}\n')