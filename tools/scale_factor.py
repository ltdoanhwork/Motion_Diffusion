"""
Script to calculate the scale factor for VQ-VAE latent space
This ensures the latent distribution matches Gaussian diffusion assumptions
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
import joblib
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vq.model import RVQVAE
from datasets.dataset import Beat2MotionDataset


def calculate_scale_factor(args):
    """Calculate scale factor from training data latents"""
    
    # Load VQ-VAE model
    print("Loading VQ-VAE model...")
    vqvae_config = {
        'args': type('Args', (), {
            'num_quantizers': 10,
            'shared_codebook': False,
            'quantize_dropout_prob': 0.0,
            'mu': 0.99,
        })(),
        'input_width': 264,
        'nb_code': 512,
        'code_dim': 512,
        'output_emb_width': 512,
        'down_t': 3,
        'stride_t': 2,
        'width': 512,
        'depth': 3,
        'dilation_growth_rate': 3,
        'activation': 'relu',
        'norm': None
    }
    
    vqvae = RVQVAE(**vqvae_config)
    
    # Load checkpoint
    checkpoint_path = pjoin(args.checkpoints_dir, 'beat', args.vqvae_name, 'model', 'best_model.tar')
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'vq_model' in checkpoint:
        vqvae.load_state_dict(checkpoint['vq_model'])
    else:
        vqvae.load_state_dict(checkpoint)
    
    vqvae.to(args.device)
    vqvae.eval()
    
    # Load dataset
    print("Loading dataset...")
    stats_file = pjoin(ROOT, 'global_pipeline.pkl')
    pipeline = joblib.load(stats_file)
    scaler = pipeline.named_steps['stdscale']
    mean = scaler.data_mean_
    std = scaler.data_std_
    
    class DummyOpt:
        def __init__(self):
            self.motion_dir = pjoin(args.data_root, 'npy')
            self.text_dir = pjoin(args.data_root, 'txt')
            self.max_motion_length = 360
            self.dataset_name = 'beat'
            self.motion_rep = 'position'
            self.is_train = False
    
    opt = DummyOpt()
    split_file = pjoin(args.data_root, 'train.txt')
    
    dataset = Beat2MotionDataset(opt, mean, std, split_file, times=1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )
    
    print(f"Processing {len(dataset)} samples...")
    
    # Collect all latents
    all_latents = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            _, motion, _ = batch
            motion = motion.to(args.device).float()
            
            # Encode to latent
            code_idx, all_codes = vqvae.encode(motion)
            # Sum over quantizers
            latent = all_codes.sum(dim=0)  # (B, T_latent, code_dim)
            
            # Transpose to (B, T, D)
            latent = latent.permute(0, 2, 1)
            
            all_latents.append(latent.cpu().numpy())
            
            # Limit samples for speed (optional)
            if args.max_samples > 0 and (batch_idx + 1) * args.batch_size >= args.max_samples:
                break
    
    # Concatenate all latents
    all_latents = np.concatenate(all_latents, axis=0)  # (N, T, D)
    print(f"Collected latents shape: {all_latents.shape}")
    
    # Calculate statistics
    latent_mean = all_latents.mean()
    latent_std = all_latents.std()
    latent_min = all_latents.min()
    latent_max = all_latents.max()
    
    print("\n" + "="*60)
    print("LATENT SPACE STATISTICS")
    print("="*60)
    print(f"Mean:  {latent_mean:.6f}")
    print(f"Std:   {latent_std:.6f}")
    print(f"Min:   {latent_min:.6f}")
    print(f"Max:   {latent_max:.6f}")
    print(f"Range: [{latent_min:.6f}, {latent_max:.6f}]")
    print("="*60)
    
    # Calculate scale factor
    scale_factor = 1.0 / latent_std
    print(f"\nRecommended SCALE_FACTOR: {scale_factor:.8f}")
    print("="*60)
    
    # Test reconstruction quality
    print("\nTesting VQ-VAE reconstruction quality...")
    test_batch_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            _, motion, _ = batch
            motion = motion[:4].to(args.device).float()  # Take 4 samples
            
            # Encode
            code_idx, all_codes = vqvae.encode(motion)
            latent = all_codes.sum(dim=0).permute(0, 2, 1)
            
            # Decode
            latent_transposed = latent.permute(0, 2, 1)
            recon_motion = vqvae.decoder(latent_transposed)
            
            # Calculate MSE
            mse = torch.nn.functional.mse_loss(motion, recon_motion)
            print(f"VQ-VAE Reconstruction MSE: {mse.item():.6f}")
            
            if mse.item() > 1.0:
                print("⚠️  WARNING: High reconstruction error! Check VQ-VAE checkpoint.")
            else:
                print("✓ VQ-VAE reconstruction looks good.")
            break
    
    # Save scale factor
    save_path = pjoin(args.checkpoints_dir, 'beat', args.vqvae_name, 'scale_factor.txt')
    with open(save_path, 'w') as f:
        f.write(f"scale_factor={scale_factor:.8f}\n")
        f.write(f"latent_mean={latent_mean:.6f}\n")
        f.write(f"latent_std={latent_std:.6f}\n")
    
    print(f"\n✓ Scale factor saved to: {save_path}")
    print("\nNext steps:")
    print("1. Add this scale_factor to your VQLatentDiffusion model")
    print("2. Multiply latent by scale_factor in encode_to_latent()")
    print("3. Divide latent by scale_factor in decode_from_latent()")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--vqvae_name', type=str, default='VQVAE_BEAT')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=5000, 
                        help='Max samples to process (0 = all)')
    parser.add_argument('--gpu_id', type=int, default=0)
    
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    calculate_scale_factor(args)