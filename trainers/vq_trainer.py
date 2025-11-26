"""
Trainer for VQ-KL Autoencoder (Hybrid VQ-VAE + KL Divergence)
Supports both discrete VQ codes and continuous KL posterior
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
from utils.utils import print_current_loss

import os
import sys


def def_value():
    return 0.0


class RVQTokenizerTrainer:
    """
    Trainer for VQ-KL Autoencoder
    Supports:
    - Standard VQ-VAE (discrete codes only)
    - VQ-KL (discrete codes + KL posterior)
    """
    
    def __init__(self, args, vq_model, kl_weight=0.0, use_posterior_sample=True):
        """
        Args:
            args: Training arguments
            vq_model: VQ-VAE or VQ-KL model
            kl_weight: Weight for KL divergence loss (0.0 for standard VQ-VAE)
            use_posterior_sample: If True, sample from posterior; else use mode
        """
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device
        
        # VQ-KL specific parameters
        self.kl_weight = kl_weight
        self.use_posterior_sample = use_posterior_sample
        self.is_vqkl = kl_weight > 0.0
        
        if self.is_vqkl:
            print(f"[INFO] VQ-KL mode enabled:")
            print(f"  - KL weight: {kl_weight}")
            print(f"  - Posterior sampling: {use_posterior_sample}")
        else:
            print(f"[INFO] Standard VQ-VAE mode")

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            
            # Initialize reconstruction loss
            if args.recons_loss == 'l1':
                self.recons_criterion = torch.nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.recons_criterion = torch.nn.SmoothL1Loss()
            elif args.recons_loss == 'l2':
                self.recons_criterion = torch.nn.MSELoss()
            else:
                raise ValueError(f"Unknown recons_loss: {args.recons_loss}")

    def calculate_velocity(self, motion):
        """
        Calculate motion velocity (first-order derivative)
        Args:
            motion: (Batch, Time, Dim)
        Returns:
            velocity: (Batch, Time-1, Dim)
        """
        return motion[:, 1:] - motion[:, :-1]

    def forward(self, batch_data):
        """
        Forward pass with loss calculation
        Supports both VQ-VAE and VQ-KL architectures
        """
        motions = batch_data[1].detach().to(self.device).float()
        
        # ==================== Forward Pass ====================
        if self.is_vqkl and hasattr(self.vq_model, 'encode'):
            # VQ-KL Mode: Use encode/decode with posterior
            posterior = self.vq_model.encode(motions)
            
            # Sample or use mode
            if self.use_posterior_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
            
            # Decode
            pred_motion = self.vq_model.decode(z)
            
            # Get VQ loss (if model has quantization)
            if hasattr(self.vq_model, 'quantize'):
                _, loss_vq, perplexity_info = self.vq_model.quantize(z)
                if isinstance(perplexity_info, tuple):
                    perplexity = perplexity_info[2]  # indices
                    perplexity = torch.tensor(len(torch.unique(perplexity)), 
                                            device=self.device, dtype=torch.float32)
                else:
                    perplexity = torch.tensor(0.0, device=self.device)
            else:
                loss_vq = torch.tensor(0.0, device=self.device)
                perplexity = torch.tensor(0.0, device=self.device)
            
            # KL divergence loss
            if hasattr(posterior, 'kl'):
                loss_kl = posterior.kl().mean()
            else:
                # Manual KL calculation for diagonal Gaussian
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
        
        # 1. Reconstruction Loss (Position)
        loss_rec = self.recons_criterion(pred_motion, motions)
        
        # 2. Velocity Loss (Smoothness & Motion Direction)
        # Helps reduce foot sliding and jittering
        gt_velocity = self.calculate_velocity(motions)
        pred_velocity = self.calculate_velocity(pred_motion)
        loss_vel = self.recons_criterion(pred_velocity, gt_velocity)
        
        # 3. Commitment Loss (from VQ)
        loss_commit = loss_vq
        
        # 4. KL Divergence Loss (for VQ-KL)
        # Already computed above
        
        # ==================== Total Loss ====================
        # Get velocity weight (default 0.5)
        weight_vel = getattr(self.opt, 'loss_vel', 0.5)
        
        # Total loss composition
        loss = (loss_rec + 
                weight_vel * loss_vel + 
                self.opt.commit * loss_commit)
        
        # Add KL loss if VQ-KL mode
        if self.is_vqkl:
            loss = loss + self.kl_weight * loss_kl
        
        # Return all components for logging
        return loss, loss_rec, loss_vel, loss_commit, perplexity, loss_kl

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):
        """Warm-up learning rate schedule"""
        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
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
        torch.save(state, file_name)

    def resume(self, model_dir):
        """Resume from checkpoint"""
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader=None, eval_val_loader=None, 
              eval_wrapper=None, plot_eval=None):
        """
        Main training loop
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (for loss monitoring)
            eval_val_loader: Evaluation loader (for metrics like FID)
            eval_wrapper: Wrapper for evaluation
            plot_eval: Evaluation plotting function
        """
        self.vq_model.to(self.device)

        # Initialize optimizer and scheduler
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

        epoch = 0
        it = 0
        min_val_loss = np.inf  # Track best validation loss

        # Resume from checkpoint if specified
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print(f"[INFO] Resumed from epoch {epoch}, iteration {it}")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'\n{"="*70}')
        print(f'Total Epochs: {self.opt.max_epoch}')
        print(f'Total Iterations: {total_iters}')
        print(f'Iterations per Epoch: {len(train_loader)}')
        if val_loader:
            print(f'Validation Batches: {len(val_loader)}')
        print(f'{"="*70}\n')
        
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        # ==================== Training Loop ====================
        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            
            for i, batch_data in enumerate(train_loader):
                it += 1
                
                # Warm-up learning rate
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(
                        it, self.opt.warm_up_iter, self.opt.lr
                    )
                
                # Forward pass
                if self.is_vqkl:
                    loss, loss_rec, loss_vel, loss_commit, perplexity, loss_kl = self.forward(batch_data)
                else:
                    loss, loss_rec, loss_vel, loss_commit, perplexity, loss_kl = self.forward(batch_data)
                
                # Backward pass
                self.opt_vq_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.vq_model.parameters(), max_norm=1.0
                )
                self.opt_vq_model.step()

                # Update learning rate after warm-up
                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                # ==================== Logging ====================
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
                
                if self.is_vqkl:
                    logs['loss_kl'] += loss_kl.item()

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    
                    # Calculate mean and log to tensorboard
                    for tag, value in logs.items():
                        self.logger.add_scalar(
                            'Train/%s' % tag, 
                            value / self.opt.log_every, 
                            it
                        )
                        mean_loss[tag] = value / self.opt.log_every
                    
                    logs = defaultdict(def_value, OrderedDict())
                    
                    # Console output
                    log_str = (f"Epoch: {epoch:03d} | Iter: {it:06d} | "
                              f"Loss: {mean_loss['loss']:.4f} | "
                              f"Rec: {mean_loss['loss_rec']:.4f} | "
                              f"Vel: {mean_loss['loss_vel']:.4f} | "
                              f"Commit: {mean_loss['loss_commit']:.4f}")
                    
                    if self.is_vqkl:
                        log_str += f" | KL: {mean_loss['loss_kl']:.6f}"
                    
                    log_str += f" | LR: {mean_loss['lr']:.6f}"
                    print(log_str)

                # Save latest checkpoint periodically
                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # Save at end of epoch
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            # ==================== Validation ====================
            if val_loader is not None:
                print(f'\n{"="*70}')
                print(f'Validation - Epoch {epoch}')
                print(f'{"="*70}')
                
                self.vq_model.eval()
                
                val_loss_list = []
                val_rec_list = []
                val_vel_list = []
                val_commit_list = []
                val_perp_list = []
                val_kl_list = []

                with torch.no_grad():
                    for i, batch_data in enumerate(val_loader):
                        # Forward pass on validation data
                        if self.is_vqkl:
                            loss, loss_rec, loss_vel, loss_commit, perplexity, loss_kl = \
                                self.forward(batch_data)
                            val_kl_list.append(loss_kl.item())
                        else:
                            loss, loss_rec, loss_vel, loss_commit, perplexity, _ = \
                                self.forward(batch_data)
                        
                        val_loss_list.append(loss.item())
                        val_rec_list.append(loss_rec.item())
                        val_vel_list.append(loss_vel.item())
                        val_commit_list.append(loss_commit.item())
                        val_perp_list.append(perplexity.item())

                # Calculate averages
                avg_loss = np.mean(val_loss_list)
                avg_rec = np.mean(val_rec_list)
                avg_vel = np.mean(val_vel_list)
                avg_commit = np.mean(val_commit_list)
                avg_perp = np.mean(val_perp_list)

                # Log to tensorboard
                self.logger.add_scalar('Val/loss', avg_loss, epoch)
                self.logger.add_scalar('Val/loss_rec', avg_rec, epoch)
                self.logger.add_scalar('Val/loss_vel', avg_vel, epoch)
                self.logger.add_scalar('Val/loss_commit', avg_commit, epoch)
                self.logger.add_scalar('Val/perplexity', avg_perp, epoch)
                
                if self.is_vqkl and val_kl_list:
                    avg_kl = np.mean(val_kl_list)
                    self.logger.add_scalar('Val/loss_kl', avg_kl, epoch)

                # Console output
                val_log = (f'Val Loss: {avg_loss:.5f} | '
                          f'Rec: {avg_rec:.5f} | '
                          f'Vel: {avg_vel:.5f} | '
                          f'Commit: {avg_commit:.5f}')
                
                if self.is_vqkl and val_kl_list:
                    val_log += f' | KL: {avg_kl:.6f}'
                
                print(val_log)

                # Save best model
                if avg_loss < min_val_loss:
                    print(f'\n{"="*70}')
                    print(f'🎉 NEW BEST MODEL!')
                    print(f'Previous: {min_val_loss:.5f} → Current: {avg_loss:.5f}')
                    print(f'{"="*70}\n')
                    min_val_loss = avg_loss
                    self.save(pjoin(self.opt.model_dir, 'best_model.tar'), epoch, it)
                
                print(f'{"="*70}\n')
            else:
                print('[INFO] Validation skipped (val_loader is None)\n')
            
            # ==================== Evaluation (FID/Diversity) ====================
            if eval_val_loader is not None and eval_wrapper is not None:
                print(f'\n{"="*70}')
                print(f'Evaluation - Epoch {epoch}')
                print(f'{"="*70}\n')
                
                # Placeholder for evaluation code
                # You can implement FID, diversity metrics here
                # Example:
                # best_fid = evaluation_vqvae(
                #     self.opt.model_dir, eval_val_loader, self.vq_model, 
                #     self.logger, epoch, eval_wrapper=eval_wrapper
                # )
                
                print('[INFO] Evaluation metrics not implemented yet')
                print(f'{"="*70}\n')


class LengthEstTrainer(object):
    """
    Trainer for motion length estimator
    (Keep original implementation for compatibility)
    """
    
    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = torch.nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()