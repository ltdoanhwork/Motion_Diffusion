"""
VQ-KL Latent Diffusion Model for Motion Generation
Updated to use VQ-KL autoencoder architecture (hybrid VQ-VAE + KL)
"""
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vq.model import RVQVAE
from models.transformer import MotionTransformer


class VQKLLatentDiffusion(nn.Module):
    """
    Latent Diffusion Model with VQ-KL autoencoder
    Combines discrete VQ codes with continuous KL-regularized latent space
    """
    def __init__(
        self,
        # VQ-KL Autoencoder parameters
        vqkl_config,
        vqkl_checkpoint=None,
        freeze_vqkl=True,
        # Latent scaling
        scale_factor=1.0,
        use_kl_posterior=True,  # Use KL posterior sampling vs VQ codes
        # Diffusion transformer parameters
        latent_dim=512,
        num_frames=60,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        activation="gelu",
        # Text encoder parameters
        num_text_layers=4,
        text_latent_dim=256,
        text_ff_size=2048,
        text_num_heads=4,
        no_clip=False,
        no_eff=False,
        **kwargs
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.freeze_vqkl = freeze_vqkl
        self.use_kl_posterior = use_kl_posterior
        
        # Scale factor for latent normalization
        self.scale_factor = scale_factor
        print(f"[INFO] Using scale_factor={scale_factor:.6f} for latent normalization")
        print(f"[INFO] KL posterior mode: {use_kl_posterior}")
        
        # Initialize VQ-KL Autoencoder
        self.vqkl = RVQVAE(**vqkl_config)
        
        # Load checkpoint if provided
        if vqkl_checkpoint is not None:
            print(f"Loading VQ-KL from {vqkl_checkpoint}")
            checkpoint = torch.load(vqkl_checkpoint, map_location='cpu')
            if 'vq_model' in checkpoint:
                self.vqkl.load_state_dict(checkpoint['vq_model'])
            elif 'state_dict' in checkpoint:
                self.vqkl.load_state_dict(checkpoint['state_dict'])
            else:
                self.vqkl.load_state_dict(checkpoint)
            print("VQ-KL loaded successfully")
        
        # Freeze if required
        if freeze_vqkl:
            for param in self.vqkl.parameters():
                param.requires_grad = False
            self.vqkl.eval()
            print("VQ-KL frozen")
        
        # Get latent dimension from config
        input_feats = vqkl_config.get('embed_dim', vqkl_config.get('code_dim', 512))
        
        # Initialize Motion Transformer
        self.transformer = MotionTransformer(
            input_feats=input_feats,
            num_frames=num_frames,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            num_text_layers=num_text_layers,
            text_latent_dim=text_latent_dim,
            text_ff_size=text_ff_size,
            text_num_heads=text_num_heads,
            no_clip=no_clip,
            no_eff=no_eff
        )
        
        print(f"VQKLLatentDiffusion initialized:")
        print(f"  - Input features: {input_feats}")
        print(f"  - Latent seq length: {num_frames}")
        print(f"  - Transformer latent dim: {latent_dim}")
        print(f"  - Scale factor: {scale_factor:.6f}")
    
    @torch.no_grad()
    def encode_to_latent(self, motion, sample_posterior=True):
        """
        Encode motion to normalized latent space using VQ-KL
        Args:
            motion: (B, T, D) raw motion data
            sample_posterior: If True, sample from KL posterior; else use VQ codes
        Returns:
            latent: (B, T_latent, embed_dim) normalized latent
            info: dict with encoding information (posterior, codes, etc.)
        """
        if self.freeze_vqkl:
            self.vqkl.eval()
        
        # Encode through VQ-KL encoder
        if hasattr(self.vqkl, 'encode'):
            # VQ-KL style: returns posterior distribution
            posterior = self.vqkl.encode(motion)
            
            if self.use_kl_posterior:
                # Use KL posterior (continuous)
                if sample_posterior:
                    latent = posterior.sample()
                else:
                    latent = posterior.mode()
            else:
                # Fallback to VQ codes if configured
                # Note: This branch might need adjustment depending on exact VQ-KL implementation
                latent = posterior.mode() 
            
            info = {
                'posterior': posterior,
                'kl_loss': posterior.kl() if hasattr(posterior, 'kl') else None
            }
            
        else:
            # Fallback to VQ-VAE style
            code_idx, all_codes = self.vqkl.encode(motion)
            latent = all_codes.sum(dim=0)  # (B, T_latent, code_dim)
            latent = latent.permute(0, 2, 1)
            info = {'code_idx': code_idx}
        
        # Ensure correct shape (B, T, C)
        if latent.shape[1] == self.vqkl.embed_dim and latent.shape[2] != self.vqkl.embed_dim:
             # (B, C, T) -> (B, T, C)
             latent = latent.permute(0, 2, 1)

        # Normalize latent
        latent_normalized = latent * self.scale_factor
        
        return latent_normalized, info
    
    # @torch.no_grad()
    def decode_from_latent(self, latent):
        """
        Decode normalized latent back to motion space
        Args:
            latent: (B, T_latent, embed_dim) normalized latent from diffusion
        Returns:
            motion: (B, T, D) reconstructed motion
        """
        if self.freeze_vqkl:
            self.vqkl.eval()
        
        # Rescale latent back to original magnitude
        latent_rescaled = latent / self.scale_factor
        
        # Handle shape for decoder: Decoder expects (B, C, T) usually
        if latent_rescaled.shape[-1] == self.vqkl.embed_dim:
             # (B, T, C) -> (B, C, T)
             latent_rescaled = latent_rescaled.permute(0, 2, 1)
            
        motion = self.vqkl.decode(latent_rescaled)
        
        return motion
    
    def forward(self, x, timesteps, y=None, **kwargs): 
        """
        Forward pass for training
        x should already be in normalized latent space
        """
        # Extract conditioning
        text = None
        length = None
        xf_proj = None
        xf_out = None

        if y is not None:
            text = y.get('text', None)
            length = y.get('length', None)
            xf_proj = y.get('xf_proj', None)
            xf_out = y.get('xf_out', None)
        
        if text is None:
            text = kwargs.get('text', None)
        if length is None:
            length = kwargs.get('length', None)
        if xf_proj is None:
            xf_proj = kwargs.get('xf_proj', None)
        if xf_out is None:
            xf_out = kwargs.get('xf_out', None)
        
        # Check if input is already in latent space
        latent = x
        
        # Predict noise/x_0 in latent space
        output = self.transformer(
            latent, 
            timesteps, 
            length=length, 
            text=text,
            xf_proj=xf_proj,
            xf_out=xf_out
        )
        
        return output


class VQKLLatentDiffusionWrapper(nn.Module):
    """
    Wrapper for training VQ-KL Latent Diffusion with GaussianDiffusion
    """
    def __init__(self, vqkl_diffusion_model):
        super().__init__()
        self.model = vqkl_diffusion_model
        
    def forward(self, x, timesteps, **kwargs):
        """
        Args:
            x: (B, T_latent, embed_dim) - normalized noisy latent
            timesteps: (B,) diffusion timesteps  
            **kwargs: Contains 'y' dict with text, length, etc.
        Returns:
            output: (B, T_latent, embed_dim) - predicted in latent space
        """
        y_dict = kwargs.get('y', {})
        text = y_dict.get('text', None)
        length = y_dict.get('length', None)
        xf_proj = y_dict.get('xf_proj', None)
        xf_out = y_dict.get('xf_out', None)
        
        # Validate input shape
        assert x.dim() == 3, f"Expected 3D input (B, T, D), got {x.shape}"
        
        # Apply transformer
        output = self.model.transformer(
            x,
            timesteps,
            length=length,
            text=text,
            xf_proj=xf_proj,
            xf_out=xf_out
        )
        
        return output


def create_vqkl_latent_diffusion(
    dataset_name='beat',
    vqkl_name='VQKL_BEAT',
    checkpoints_dir='./checkpoints',
    device='cuda',
    freeze_vqkl=True,
    scale_factor=None,
    use_kl_posterior=True,
    **diffusion_kwargs
):
    """
    Factory function to create VQKLLatentDiffusion model
    """
    import os
    from os.path import join as pjoin
    
    # Dataset-specific configurations
    if dataset_name == 't2m':
        dim_pose = 263
        down_t = 2
        num_frames_original = 196
    elif dataset_name == 'kit':
        dim_pose = 251
        down_t = 2
        num_frames_original = 196
    elif dataset_name == 'beat':
        dim_pose = 264
        down_t = 3
        num_frames_original = 360
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Calculate latent sequence length
    num_frames_latent = num_frames_original // (2 ** down_t)
    
    # VQ-KL configuration (hybrid VQ + KL)
    vqkl_config = {
        'args': type('Args', (), {
            'num_quantizers': 10,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.0,
            'mu': 0.99,
        })(),
        'input_width': dim_pose,
        # FIX: Changed nb_code from 512 to 1024 to match checkpoint
        'nb_code': 1024, 
        'code_dim': 512,
        'embed_dim': 512,  # For KL posterior
        'output_emb_width': 512,
        'down_t': down_t,
        'stride_t': 2,
        'width': 512,
        'depth': 3,
        'dilation_growth_rate': 3,
        'activation': 'relu',
        'norm': None,
        'double_z': True,  # For KL: encode to mean and logvar
    }
    
    # VQ-KL checkpoint path
    vqkl_checkpoint = pjoin(
        checkpoints_dir, 
        dataset_name, 
        vqkl_name, 
        'model', 
        'best_model.tar'
    )
    
    if not os.path.exists(vqkl_checkpoint):
        print(f"Warning: VQ-KL checkpoint not found at {vqkl_checkpoint}")
        # Try latest.tar as fallback
        vqkl_checkpoint_fallback = pjoin(checkpoints_dir, dataset_name, vqkl_name, 'model', 'latest.tar')
        if os.path.exists(vqkl_checkpoint_fallback):
             print(f"Using fallback checkpoint: {vqkl_checkpoint_fallback}")
             vqkl_checkpoint = vqkl_checkpoint_fallback
        else:
             vqkl_checkpoint = None
    
    # Load scale_factor if not provided
    if scale_factor is None:
        scale_factor_file = pjoin(checkpoints_dir, dataset_name, vqkl_name, 'scale_factor.txt')
        if os.path.exists(scale_factor_file):
            with open(scale_factor_file, 'r') as f:
                for line in f:
                    if line.startswith('scale_factor='):
                        scale_factor = float(line.split('=')[1].strip())
                        print(f"[INFO] Loaded scale_factor={scale_factor:.6f}")
                        break
        
        if scale_factor is None:
            print("="*70)
            print("WARNING: scale_factor not found! Using default=1.0")
            print("="*70)
            scale_factor = 1.0
    
    # Default diffusion transformer parameters
    default_kwargs = {
        'num_frames': num_frames_latent,
        'latent_dim': 512,
        'ff_size': 1024,
        'num_layers': 8,
        'num_heads': 8,
        'dropout': 0.1,
        'activation': 'gelu',
        'num_text_layers': 4,
        'text_latent_dim': 256,
        'text_ff_size': 2048,
        'text_num_heads': 4,
        'no_clip': False,
        'no_eff': False,
    }
    default_kwargs.update(diffusion_kwargs)
    
    # Create model
    model = VQKLLatentDiffusion(
        vqkl_config=vqkl_config,
        vqkl_checkpoint=vqkl_checkpoint,
        freeze_vqkl=freeze_vqkl,
        scale_factor=scale_factor,
        use_kl_posterior=use_kl_posterior,
        **default_kwargs
    )
    
    return model.to(device)