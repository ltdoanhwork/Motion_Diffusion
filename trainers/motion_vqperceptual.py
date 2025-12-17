"""
Motion-specific VQ Perceptual Loss with Discriminator
Adapted for motion data (1D temporal sequences instead of 2D images)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def adopt_weight(weight, global_step, threshold=0, value=0.):
    """Gradually introduce a loss component after a threshold"""
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    """Hinge loss for discriminator"""
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    """Vanilla GAN loss for discriminator"""
    d_loss = 0.5 * (
        torch.mean(F.softplus(-logits_real)) +
        torch.mean(F.softplus(logits_fake))
    )
    return d_loss


class MotionPerceptualLoss(nn.Module):
    """
    Perceptual loss for motion data using learned features
    Instead of pre-trained VGG (for images), we use a simple feature extractor
    """
    def __init__(self, in_channels=264):
        super().__init__()
        
        # Simple 1D CNN feature extractor for motion
        # Extract multi-scale features similar to VGG layers
        self.features = nn.ModuleList([
            # Layer 1: Low-level features (local motion patterns)
            nn.Sequential(
                nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            # Layer 2: Mid-level features (motion phrases)
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
            # Layer 3: High-level features (motion semantics)
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Freeze feature extractor (we want fixed features like LPIPS)
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, input, target):
        """
        Args:
            input: (B, C, T) - predicted motion
            target: (B, C, T) - ground truth motion
        Returns:
            perceptual loss (scalar)
        """
        loss = 0.0
        x_input = input
        x_target = target
        
        # Extract features at multiple scales
        for layer in self.features:
            x_input = layer(x_input)
            x_target = layer(x_target)
            
            # L2 distance in feature space
            loss += F.mse_loss(x_input, x_target)
        
        return loss


class MotionDiscriminator(nn.Module):
    """
    1D Temporal Discriminator for motion sequences
    Adapted from NLayerDiscriminator for 1D temporal data
    """
    def __init__(self, input_nc=264, ndf=64, n_layers=3, use_actnorm=False):
        """
        Args:
            input_nc: Number of input channels (motion features)
            ndf: Number of discriminator filters
            n_layers: Number of convolutional layers
            use_actnorm: Use actnorm instead of batchnorm
        """
        super().__init__()
        
        if use_actnorm:
            norm_layer = lambda c: nn.GroupNorm(1, c, affine=True)
        else:
            norm_layer = nn.BatchNorm1d
        
        # Initial convolution
        sequence = [
            nn.Conv1d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Progressively increase filters and reduce temporal resolution
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Output layer: classify as real/fake
        sequence += [
            nn.Conv1d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        """
        Args:
            input: (B, C, T) - motion sequence
        Returns:
            logits: (B, 1, T') - real/fake classification for each temporal region
        """
        return self.model(input)


class MotionVQGANLoss(nn.Module):
    """
    Complete loss module for Motion VQ-GAN
    Combines:
    - Pixel-wise reconstruction loss
    - Motion perceptual loss
    - Adversarial discriminator loss
    - VQ codebook loss
    """
    def __init__(self, 
                 disc_start,
                 codebook_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=264,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_ndf=64,
                 disc_loss="hinge"):
        """
        Args:
            disc_start: Iteration to start discriminator training
            codebook_weight: Weight for VQ codebook loss
            pixelloss_weight: Weight for pixel-wise reconstruction
            disc_num_layers: Number of discriminator layers
            disc_in_channels: Input channels (motion dimension)
            disc_factor: Global discriminator loss factor
            disc_weight: Discriminator loss weight
            perceptual_weight: Weight for perceptual loss
            use_actnorm: Use actnorm in discriminator
            disc_conditional: Conditional discriminator (not used for motion)
            disc_ndf: Discriminator base filters
            disc_loss: Type of GAN loss (hinge/vanilla)
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        
        # Motion-specific perceptual loss
        self.perceptual_loss = MotionPerceptualLoss(
            in_channels=disc_in_channels
        ).eval()
        self.perceptual_weight = perceptual_weight
        
        # Motion discriminator
        self.discriminator = MotionDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        )
        
        self.discriminator_iter_start = disc_start
        
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        
        print(f"[INFO] MotionVQGANLoss initialized with {disc_loss} loss")
        
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Adaptive weight balancing reconstruction and adversarial loss
        Based on gradient magnitudes
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # Fallback: use equal weighting
            return torch.tensor(self.discriminator_weight)
        
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        """
        Forward pass for either generator or discriminator
        
        Args:
            codebook_loss: VQ codebook commitment loss
            inputs: (B, C, T) - ground truth motion
            reconstructions: (B, C, T) - reconstructed motion
            optimizer_idx: 0 for generator, 1 for discriminator
            global_step: Current training iteration
            last_layer: Last layer of decoder for adaptive weighting
            cond: Conditional input (not used)
            split: "train" or "val"
        
        Returns:
            loss: Total loss
            log: Dictionary of individual loss components
        """
        # ==================== Reconstruction Loss ====================
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        
        # ==================== Perceptual Loss ====================
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0], device=inputs.device)
        
        nll_loss = torch.mean(rec_loss)
        
        # ==================== Generator Update ====================
        if optimizer_idx == 0:
            # Generator tries to fool discriminator
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            
            g_loss = -torch.mean(logits_fake)
            
            # Adaptive weighting between reconstruction and adversarial
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)
            
            # Progressive GAN loss introduction
            disc_factor = adopt_weight(
                self.disc_factor, 
                global_step, 
                threshold=self.discriminator_iter_start
            )
            
            # Total generator loss
            loss = (nll_loss + 
                   d_weight * disc_factor * g_loss + 
                   self.codebook_weight * codebook_loss.mean())
            
            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean(),
                "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                "{}/nll_loss".format(split): nll_loss.detach().mean(),
                "{}/rec_loss".format(split): torch.mean(rec_loss.detach()),
                "{}/p_loss".format(split): p_loss.detach().mean() if isinstance(p_loss, torch.Tensor) else p_loss,
                "{}/d_weight".format(split): d_weight.detach(),
                "{}/disc_factor".format(split): torch.tensor(disc_factor),
                "{}/g_loss".format(split): g_loss.detach().mean(),
            }
            return loss, log
        
        # ==================== Discriminator Update ====================
        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )
            
            disc_factor = adopt_weight(
                self.disc_factor,
                global_step,
                threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            
            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean()
            }
            return d_loss, log


# Convenience function for backward compatibility
class VQLPIPSWithDiscriminator(MotionVQGANLoss):
    """Alias for backward compatibility with original code"""
    pass