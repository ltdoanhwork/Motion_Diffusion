"""
Trainer for VQ-KL Autoencoder with GAN + Perceptual Loss
Integrates VQLPIPSWithDiscriminator for better reconstruction quality
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

# Import Motion-specific VQLPIPSWithDiscriminator
# Use custom implementation for 1D motion sequences instead of 2D images
try:
    from trainers.motion_vqperceptual import VQLPIPSWithDiscriminator
except ImportError:
    # Fallback to taming if custom version not available
    print("[WARNING] Using taming LPIPS - may cause dimension mismatch for motion data")
    from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator


def def_value():
    return 0.0


class RVQTokenizerTrainerGAN:
    """
    Trainer for VQ-KL Autoencoder with GAN
    Supports:
    - VQ codes (discrete)
    - KL divergence (continuous posterior)
    - LPIPS perceptual loss
    - GAN discriminator loss
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
                 disc_in_channels=264):
        """
        Args:
            args: Training arguments
            vq_model: VQ-VAE or VQ-KL model
            kl_weight: KL divergence weight
            use_posterior_sample: Sample from posterior
            disc_start: Iteration to start discriminator training
            disc_weight: Discriminator loss weight
            disc_factor: Discriminator loss factor
            perceptual_weight: LPIPS perceptual loss weight
            disc_loss: Discriminator loss type (hinge/vanilla)
            disc_num_layers: Number of discriminator layers
            disc_in_channels: Input channels for discriminator
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
            
            # Base reconstruction loss
            if args.recons_loss == 'l1':
                self.recons_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.recons_criterion = torch.nn.SmoothL1Loss()
            elif args.recons_loss == 'l2':
                self.recons_criterion = torch.nn.MSELoss()
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

    def calculate_velocity(self, motion):
        """Calculate motion velocity (first-order derivative)"""
        return motion[:, 1:] - motion[:, :-1]

    def forward(self, batch_data, global_step=0, optimizer_idx=0):
        """
        Forward pass with GAN loss
        Args:
            batch_data: Input batch
            global_step: Current training iteration
            optimizer_idx: 0 for generator, 1 for discriminator
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
        
        # ==================== Loss Components ====================
        
        # 1. Basic Reconstruction Loss
        loss_rec = self.recons_criterion(pred_motion, motions)
        
        # 2. Velocity Loss
        gt_velocity = self.calculate_velocity(motions)
        pred_velocity = self.calculate_velocity(pred_motion)
        loss_vel = self.recons_criterion(pred_velocity, gt_velocity)
        
        # 3. VQ Commitment Loss
        loss_commit = loss_vq
        
        # 4. KL Loss
        # Already computed
        
        # ==================== GAN Loss ====================
        if self.use_gan:
            # Reshape for discriminator: (B, T, C) -> (B, C, T)
            # Note: motion discriminator might need temporal dimension
            motions_disc = motions.permute(0, 2, 1)  # (B, C, T)
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
                # GAN loss already includes rec + perceptual + disc + codebook
                # We need to add our custom losses
                weight_vel = getattr(self.opt, 'loss_vel', 0.1)
                
                loss = gan_loss + weight_vel * loss_vel
                
                if self.is_vqkl:
                    loss = loss + self.kl_weight * loss_kl
                
                # Extract individual losses from GAN log for monitoring
                loss_gan_g = gan_log.get('train/g_loss', torch.tensor(0.0))
                loss_perceptual = gan_log.get('train/p_loss', torch.tensor(0.0))
                disc_weight = gan_log.get('train/d_weight', torch.tensor(0.0))
                
                return (loss, loss_rec, loss_vel, loss_commit, perplexity, 
                       loss_kl, loss_gan_g, loss_perceptual, disc_weight)
            
            else:
                # Discriminator update
                return (gan_loss, torch.tensor(0.0), torch.tensor(0.0), 
                       torch.tensor(0.0), perplexity, torch.tensor(0.0),
                       torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))
        
        else:
            # No GAN - standard training
            weight_vel = getattr(self.opt, 'loss_vel', 0.1)
            
            loss = (loss_rec + 
                   weight_vel * loss_vel + 
                   self.opt.commit * loss_commit)
            
            if self.is_vqkl:
                loss = loss + self.kl_weight * loss_kl
            
            return (loss, loss_rec, loss_vel, loss_commit, perplexity, 
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
        """Main training loop with GAN"""
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
                (loss, loss_rec, loss_vel, loss_commit, perplexity,
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
                              f"Vel: {mean_loss['loss_vel']:.4f}")
                    
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
                        (loss, loss_rec, loss_vel, loss_commit, perplexity,
                         loss_kl, loss_gan_g, loss_perceptual, _) = results
                        
                        val_metrics['loss'].append(loss.item())
                        val_metrics['loss_rec'].append(loss_rec.item())
                        val_metrics['loss_vel'].append(loss_vel.item())
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
                
                val_log = f'Val Loss: {avg_loss:.5f}'
                for key in ['loss_rec', 'loss_vel', 'loss_commit']:
                    val_log += f' | {key.split("_")[-1].capitalize()}: {np.mean(val_metrics[key]):.5f}'
                
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