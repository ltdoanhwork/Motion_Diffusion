"""
VQ-VAE Latent Diffusion Model for Motion Generation
FIXED: Added proper latent scaling for Gaussian diffusion compatibility
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


class VQLatentDiffusion(nn.Module):
    """
    Latent Diffusion Model with proper scaling for VQ-VAE latent space
    """
    def __init__(
        self,
        # VQ-VAE parameters
        vqvae_config,
        vqvae_checkpoint=None,
        freeze_vqvae=True,
        # Latent scaling (CRITICAL FIX)
        scale_factor=1.0,
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
        self.freeze_vqvae = freeze_vqvae
        
        # CRITICAL: Scale factor to normalize latent distribution
        self.scale_factor = scale_factor
        print(f"[INFO] Using scale_factor={scale_factor:.6f} for latent normalization")
        
        # Initialize VQ-VAE
        self.vqvae = RVQVAE(**vqvae_config)
        
        # Load VQ-VAE checkpoint if provided
        if vqvae_checkpoint is not None:
            print(f"Loading VQ-VAE from {vqvae_checkpoint}")
            checkpoint = torch.load(vqvae_checkpoint, map_location='cpu')
            if 'vq_model' in checkpoint:
                self.vqvae.load_state_dict(checkpoint['vq_model'])
            else:
                self.vqvae.load_state_dict(checkpoint)
            print("VQ-VAE loaded successfully")
        
        # Freeze VQ-VAE if required
        if freeze_vqvae:
            for param in self.vqvae.parameters():
                param.requires_grad = False
            self.vqvae.eval()
            print("VQ-VAE frozen")
        
        # Input features for transformer
        input_feats = vqvae_config['code_dim']
        
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
        
        print(f"VQLatentDiffusion initialized:")
        print(f"  - Input features: {input_feats}")
        print(f"  - Latent seq length: {num_frames}")
        print(f"  - Transformer latent dim: {latent_dim}")
        print(f"  - Scale factor: {scale_factor:.6f}")
    
    @torch.no_grad()
    def encode_to_latent(self, motion):
        """
        Encode motion to NORMALIZED latent space
        Args:
            motion: (B, T, D) raw motion data
        Returns:
            latent: (B, T_latent, code_dim) normalized latent (mean~0, std~1)
            code_idx: (B, T_latent, num_quantizers) discrete code indices
        """
        if self.freeze_vqvae:
            self.vqvae.eval()
        
        # Encode motion
        code_idx, all_codes = self.vqvae.encode(motion)
        # all_codes: (Q, B, T_latent, code_dim)
        
        # Sum over quantizers
        latent = all_codes.sum(dim=0)  # (B, T_latent, code_dim)
        
        # Transpose to (B, T, D)
        latent = latent.permute(0, 2, 1)  # (B, T_latent, code_dim)
        
        # CRITICAL FIX: Scale latent to have std ~ 1.0
        # This makes it compatible with Gaussian diffusion noise
        latent_normalized = latent * self.scale_factor
        
        # Verify shape
        B = latent_normalized.shape[0]
        assert latent_normalized.shape == (B, self.num_frames, self.vqvae.code_dim), \
            f"Latent shape mismatch: {latent_normalized.shape} vs ({B}, {self.num_frames}, {self.vqvae.code_dim})"
        
        return latent_normalized, code_idx
    
    # @torch.no_grad()
    def decode_from_latent(self, latent=None, code_idx=None):
        """
        Decode NORMALIZED latent back to motion space
        Args:
            latent: (B, T_latent, code_dim) normalized latent from diffusion
            code_idx: (B, T_latent, Q) discrete code indices
        Returns:
            motion: (B, T, D) reconstructed motion
        """
        if self.freeze_vqvae:
            self.vqvae.eval()
        
        if code_idx is not None:
            # Use discrete codes
            motion = self.vqvae.forward_decoder(code_idx)
        else:
            # CRITICAL FIX: Rescale latent back to original magnitude
            latent_rescaled = latent / self.scale_factor
            
            # Transpose for decoder: (B, T_latent, code_dim) -> (B, code_dim, T_latent)
            latent_transposed = latent_rescaled.permute(0, 2, 1)
            motion = self.vqvae.decoder(latent_transposed)
        
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
        if x.shape[-1] == self.vqvae.code_dim and x.shape[1] == self.num_frames:
            latent = x
        else:
            # Encode if raw motion (shouldn't happen during training)
            with torch.no_grad():
                latent, _ = self.encode_to_latent(x)

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


class VQLatentDiffusionWrapper(nn.Module):
    """
    Wrapper for training VQ Latent Diffusion with GaussianDiffusion
    """
    def __init__(self, vq_diffusion_model):
        super().__init__()
        self.model = vq_diffusion_model
        
    def forward(self, x, timesteps, **kwargs):
        """
        Args:
            x: (B, T_latent, code_dim) - normalized noisy latent from GaussianDiffusion
            timesteps: (B,) diffusion timesteps  
            **kwargs: Contains 'y' dict with text, length, etc.
        Returns:
            output: (B, T_latent, code_dim) - predicted in latent space
        """
        # Extract conditioning
        y_dict = kwargs.get('y', {})
        text = y_dict.get('text', None)
        length = y_dict.get('length', None)
        xf_proj = y_dict.get('xf_proj', None)
        xf_out = y_dict.get('xf_out', None)
        
        # Validate input shape
        assert x.dim() == 3, f"Expected 3D input (B, T, D), got {x.shape}"
        B, T, D = x.shape
        
        # Verify dimensions
        assert T == self.model.num_frames, \
            f"Sequence length mismatch: {T} vs {self.model.num_frames}"
        assert D == self.model.vqvae.code_dim, \
            f"Feature dim mismatch: {D} vs {self.model.vqvae.code_dim}"
        
        # x is already normalized latent, apply transformer
        output = self.model.transformer(
            x,
            timesteps,
            length=length,
            text=text,
            xf_proj=xf_proj,
            xf_out=xf_out
        )
        
        return output


def create_vq_latent_diffusion(
    dataset_name='beat',
    vqvae_name='VQVAE_BEAT',
    checkpoints_dir='./checkpoints',
    device='cuda',
    freeze_vqvae=True,
    scale_factor=None,  # NEW: Allow passing scale_factor
    **diffusion_kwargs
):
    """
    Factory function to create VQLatentDiffusion model with proper scaling
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
    
    # VQ-VAE configuration
    vqvae_config = {
        'args': type('Args', (), {
            'num_quantizers': 10,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.0,
            'mu': 0.99,
        })(),
        'input_width': dim_pose,
        'nb_code': 512,
        'code_dim': 512,
        'output_emb_width': 512,
        'down_t': down_t,
        'stride_t': 2,
        'width': 512,
        'depth': 3,
        'dilation_growth_rate': 3,
        'activation': 'relu',
        'norm': None
    }
    
    # VQ-VAE checkpoint path
    vqvae_checkpoint = pjoin(
        checkpoints_dir, 
        dataset_name, 
        vqvae_name, 
        'model', 
        'best_model.tar'
    )
    
    if not os.path.exists(vqvae_checkpoint):
        print(f"Warning: VQ-VAE checkpoint not found at {vqvae_checkpoint}")
        vqvae_checkpoint = None
    
    # CRITICAL: Load scale_factor if not provided
    if scale_factor is None:
        scale_factor_file = pjoin(checkpoints_dir, dataset_name, vqvae_name, 'scale_factor.txt')
        if os.path.exists(scale_factor_file):
            with open(scale_factor_file, 'r') as f:
                for line in f:
                    if line.startswith('scale_factor='):
                        scale_factor = float(line.split('=')[1].strip())
                        print(f"[INFO] Loaded scale_factor={scale_factor:.6f} from {scale_factor_file}")
                        break
        
        if scale_factor is None:
            print("="*70)
            print("WARNING: scale_factor not found!")
            print("Please run: python calculate_scale_factor.py")
            print("Using default scale_factor=1.0 (NOT RECOMMENDED)")
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
    
    # Create model with scale_factor
    model = VQLatentDiffusion(
        vqvae_config=vqvae_config,
        vqvae_checkpoint=vqvae_checkpoint,
        freeze_vqvae=freeze_vqvae,
        scale_factor=scale_factor,  # CRITICAL PARAMETER
        **default_kwargs
    )
    
    return model.to(device)