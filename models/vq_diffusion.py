"""
VQ-VAE Latent Diffusion Model for Motion Generation
Integrates VQ-VAE encoder/decoder with MotionTransformer
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
    Latent Diffusion Model that works in VQ-VAE latent space
    """
    def __init__(
        self,
        # VQ-VAE parameters
        vqvae_config,
        vqvae_checkpoint=None,
        freeze_vqvae=True,
        # Diffusion transformer parameters
        latent_dim=512,
        num_frames=60,  # This is latent sequence length (original_frames // down_t^2)
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
        # Training mode
        use_continuous_latent=True,  # If True, use continuous latent; else use discrete codes
        **kwargs
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.use_continuous_latent = use_continuous_latent
        self.freeze_vqvae = freeze_vqvae
        
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
        
        # Calculate input features for transformer
        # For continuous latent: code_dim
        # For discrete codes: code_dim (after embedding lookup)
        input_feats = vqvae_config['code_dim']
        
        # Initialize Motion Transformer to work on latent space
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
        print(f"  - Input features (latent): {input_feats}")
        print(f"  - Latent sequence length: {num_frames}")
        print(f"  - Transformer latent dim: {latent_dim}")
        print(f"  - Use continuous latent: {use_continuous_latent}")
    
    @torch.no_grad()
    def encode_to_latent(self, motion):
        """
        Encode motion to latent space using VQ-VAE
        Args:
            motion: (B, T, D) raw motion data
        Returns:
            latent: (B, T_latent, code_dim) continuous latent representation
            code_idx: (B, T_latent, num_quantizers) discrete code indices
        """
        if self.freeze_vqvae:
            self.vqvae.eval()
        
        # Encode motion
        code_idx, all_codes = self.vqvae.encode(motion)
        # code_idx: (B, T_latent, Q)
        # all_codes: (Q, B, T_latent, code_dim)
        
        # Sum over quantizers to get continuous representation
        latent = all_codes.sum(dim=0)  # (B, T_latent, code_dim)
        
        return latent, code_idx
    
    @torch.no_grad()
    def decode_from_latent(self, latent=None, code_idx=None):
        """
        Decode latent back to motion space using VQ-VAE decoder
        Args:
            latent: (B, T_latent, code_dim) continuous latent
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
            # Use continuous latent
            # Reshape for decoder: (B, T_latent, code_dim) -> (B, code_dim, T_latent)
            latent_transposed = latent.permute(0, 2, 1)
            motion = self.vqvae.decoder(latent_transposed)
        
        return motion
    
    def forward(self, x, timesteps, y=None, **kwargs): # <--- SỬA CHỮ KÝ HÀM
        """
        Forward pass for training ...
        """
        # --- THÊM KHỐI GIẢI NÉN 'y' NÀY ---
        text = y.get('text', None)
        length = y.get('length', None)
        xf_proj = y.get('xf_proj', None)
        xf_out = y.get('xf_out', None)
        # ---------------------------------
        
        # Check if input is already in latent space
        if x.shape[-1] == self.vqvae.code_dim and x.shape[1] < 200:
            latent = x
        else:
            with torch.no_grad():
                latent, _ = self.encode_to_latent(x)

        output = self.transformer(
            latent, 
            timesteps, 
            length=length, 
            text=text,
            xf_proj=xf_proj,
            xf_out=xf_out
            )
        
        return output
    
    def forward_with_reconstruction(self, x, timesteps, length=None, text=None):
        """
        Forward pass with full reconstruction (for evaluation)
        Args:
            x: (B, T_original, D) raw motion
            timesteps: (B,) diffusion timesteps
            length: List of sequence lengths
            text: List of text descriptions
        Returns:
            pred_latent: (B, T_latent, code_dim) predicted latent
            recon_motion: (B, T_original, D) reconstructed motion
            original_latent: (B, T_latent, code_dim) original latent
        """
        # Encode to latent
        with torch.no_grad():
            original_latent, code_idx = self.encode_to_latent(x)
        
        # Predict in latent space
        pred_latent = self.forward(
            original_latent, 
            timesteps, 
            length=length, 
            text=text
        )
        
        # Decode back to motion space
        with torch.no_grad():
            recon_motion = self.decode_from_latent(latent=pred_latent)
        
        return pred_latent, recon_motion, original_latent


class VQLatentDiffusionWrapper(nn.Module):
    """
    Wrapper that handles the full pipeline: raw motion -> latent -> diffusion -> reconstruction
    This is useful for training with the GaussianDiffusion class
    """
    def __init__(self, vq_diffusion_model):
        super().__init__()
        self.model = vq_diffusion_model
        
    def forward(self, x, timesteps, **kwargs):
        """
        Args:
            x: (B, T_original, D) raw motion data
            timesteps: (B,) diffusion timesteps
            **kwargs: Additional arguments (text, length, etc.)
        Returns:
            output: (B, T_latent, code_dim) predicted in latent space
        """
        # Encode to latent space
        with torch.no_grad():
            if self.model.freeze_vqvae:
                self.model.vqvae.eval()
            latent, _ = self.model.encode_to_latent(x)
        
        # Apply diffusion model in latent space
        output = self.model.transformer(
            latent,
            timesteps,
            length=kwargs.get('y', {}).get('length', None),
            text=kwargs.get('y', {}).get('text', None),
            xf_proj=kwargs.get('y', {}).get('xf_proj', None),
            xf_out=kwargs.get('y', {}).get('xf_out', None)
        )
        
        return output


def create_vq_latent_diffusion(
    dataset_name='t2m',
    vqvae_name='VQVAE_t2m',
    checkpoints_dir='./checkpoints',
    device='cuda',
    freeze_vqvae=True,
    **diffusion_kwargs
):
    """
    Factory function to create VQLatentDiffusion model
    
    Args:
        dataset_name: 't2m', 'kit', or 'beat'
        vqvae_name: Name of trained VQ-VAE checkpoint
        checkpoints_dir: Directory containing checkpoints
        device: Device to load model
        freeze_vqvae: Whether to freeze VQ-VAE weights
        **diffusion_kwargs: Additional arguments for transformer
    
    Returns:
        model: VQLatentDiffusion model
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
        dim_pose = 264  # axis-angle for 55 joints
        down_t = 3
        num_frames_original = 360  # ~6 seconds at 60fps
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Calculate latent sequence length
    # With down_t=2 and stride_t=2, we downsample by 2^down_t = 4
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
        'finest.tar'
    )
    
    if not os.path.exists(vqvae_checkpoint):
        print(f"Warning: VQ-VAE checkpoint not found at {vqvae_checkpoint}")
        vqvae_checkpoint = None
    
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
    model = VQLatentDiffusion(
        vqvae_config=vqvae_config,
        vqvae_checkpoint=vqvae_checkpoint,
        freeze_vqvae=freeze_vqvae,
        **default_kwargs
    )
    
    return model.to(device)


# Example usage
if __name__ == "__main__":
    # Test VQLatentDiffusion
    print("="*50)
    print("Testing VQLatentDiffusion")
    print("="*50)
    
    # Create model
    model = create_vq_latent_diffusion(
        dataset_name='t2m',
        vqvae_name='VQVAE_t2m',
        device='cuda',
        freeze_vqvae=True,
        num_layers=6,  # Smaller for testing
    )
    
    print(f"\nModel created:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    
    # Test forward pass with raw motion
    B, T_original, D = 4, 196, 263
    motion = torch.randn(B, T_original, D).cuda()
    timesteps = torch.randint(0, 1000, (B,)).cuda()
    text = ["a person walks forward"] * B
    length = [196, 180, 160, 150]
    
    print(f"\nTesting forward pass:")
    print(f"  Input motion: {motion.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    
    # Encode to latent
    with torch.no_grad():
        latent, code_idx = model.encode_to_latent(motion)
    print(f"  Encoded latent: {latent.shape}")
    print(f"  Code indices: {code_idx.shape}")
    
    # Forward pass
    output = model(latent, timesteps, length=length, text=text)
    print(f"  Output (predicted): {output.shape}")
    
    # Decode back
    with torch.no_grad():
        recon = model.decode_from_latent(latent=output)
    print(f"  Reconstructed motion: {recon.shape}")
    
    # Test with wrapper
    print(f"\nTesting with wrapper:")
    wrapper = VQLatentDiffusionWrapper(model)
    y_dict = {
        'text': text,
        'length': length
    }
    output_wrapper = wrapper(motion, timesteps, y=y_dict)
    print(f"  Wrapper output: {output_wrapper.shape}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)