"""
Training script for VQ-VAE with Self-Attention and Skip Connections
Improved reconstruction quality, especially for fine details (hands)
"""
import os
import sys
from os.path import join as pjoin
import argparse
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PYMO_DIR = os.path.join(ROOT, 'datasets', 'pymo')
if PYMO_DIR not in sys.path:
    sys.path.insert(0, PYMO_DIR)

# Import models and trainers
from models.vq.model import RVQVAEWithAttention
from trainers.vq_trainer import RVQTokenizerTrainer
from datasets.dataset import Beat2MotionDataset as DatasetClass
from utils.fixseed import fixseed


def create_train_split(motion_dir, split_file):
    """Auto-create train.txt from all .npy files"""
    if os.path.exists(split_file):
        print(f"[INFO] {split_file} already exists")
        return
        
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    
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
        
    print(f"[INFO] Found {len(motion_files)} motion files")
    motion_files_sorted = sorted(motion_files)
    
    with open(split_file, 'w', encoding='utf-8') as f:
        for name in motion_files_sorted:
            f.write(f"{name}\n")
    
    print(f"[INFO] Created {split_file}")


def create_val_split(train_file, val_file, val_ratio=0.1):
    """Auto-create val.txt by splitting from train.txt"""
    if os.path.exists(val_file):
        print(f"[INFO] {val_file} already exists")
        return
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"{train_file} not found")

    print(f"[INFO] Creating validation split from {train_file}...")
    with open(train_file, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    np.random.shuffle(all_lines)
    val_size = int(len(all_lines) * val_ratio)
    if val_size == 0 and len(all_lines) > 0:
        val_size = 1
        
    val_lines = all_lines[:val_size]
    train_lines = all_lines[val_size:]

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines) + '\n')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines) + '\n')
    
    print(f"[INFO] Created {val_file} with {len(val_lines)} samples")
    print(f"[INFO] Updated {train_file} with {len(train_lines)} samples")


class DummyOpt:
    """Mock options for Beat2MotionDataset"""
    def __init__(self, args, is_train):
        self.motion_dir = args.motion_dir
        self.text_dir = args.text_dir
        self.max_motion_length = args.max_motion_length
        self.dataset_name = args.dataset_name
        self.motion_rep = args.motion_rep
        self.is_train = is_train
        self.joints_num = args.joints_num


def main():
    parser = argparse.ArgumentParser(
        description='Train VQ-VAE with Self-Attention and Skip Connections'
    )
    
    # ==================== Paths ====================
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
    parser.add_argument('--name', type=str, default='VQKL_Attention_BEAT',
                        help='Experiment name')
    parser.add_argument('--dataset_name', type=str, default='beat',
                        choices=['t2m', 'kit', 'beat'])
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy')
    
    # ==================== VQ-KL Model ====================
    parser.add_argument('--dim_pose', type=int, default=264)
    parser.add_argument('--codebook_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--num_codebooks', type=int, default=10)
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--joints_num', type=int, default=55)
    parser.add_argument('--double_z', action='store_true', default=True)
    parser.add_argument('--use_posterior_sample', action='store_true', default=True)
    
    # ==================== Attention Parameters ====================
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Enable self-attention in encoder/decoder')
    parser.add_argument('--attention_type', type=str, default='self',
                        choices=['self', 'nonlocal'],
                        help='Type of attention: self (multi-head) or nonlocal')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--encoder_attention_layers', type=int, nargs='+', 
                        default=[2],
                        help='Which encoder layers to add attention (0-indexed)')
    parser.add_argument('--decoder_attention_layers', type=int, nargs='+',
                        default=[1],
                        help='Which decoder layers to add attention (0-indexed)')
    
    # ==================== Skip Connection Parameters ====================
    parser.add_argument('--use_skip_connections', action='store_true', default=True,
                        help='Enable U-Net style skip connections')
    parser.add_argument('--skip_connection_type', type=str, default='concat',
                        choices=['concat', 'cross_attn'],
                        help='Skip connection fusion: concat (simple) or cross_attn (sophisticated)')
    
    # ==================== Training ====================
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--is_continue', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # ==================== Loss Weights ====================
    parser.add_argument('--recons_loss', type=str, default='l1',
                        choices=['l1', 'l1_smooth', 'l2'])
    parser.add_argument('--loss_vel', type=float, default=0.1)
    parser.add_argument('--commit', type=float, default=0.02)
    parser.add_argument('--kl_weight', type=float, default=1e-6)
    
    # ==================== Training Schedule ====================
    parser.add_argument('--warm_up_iter', type=int, default=1000)
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 40, 60])
    parser.add_argument('--gamma', type=float, default=0.5)
    
    # ==================== Logging ====================
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--save_latest', type=int, default=500)
    parser.add_argument('--eval_every_e', type=int, default=5)
    
    # ==================== Device ====================
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3407)

    args = parser.parse_args()

    # ==================== Setup ====================
    fixseed(args.seed)
    args.is_train = True
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("VQ-VAE WITH SELF-ATTENTION AND SKIP CONNECTIONS")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Architecture: VQ + KL + Attention + Skip Connections")
    print(f"Dataset: {args.dataset_name.upper()}")
    print("="*70 + "\n")

    # ==================== Directories ====================
    args.save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    args.model_dir = pjoin(args.save_root, 'model')
    args.log_dir = pjoin(args.save_root, 'logs')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"[INFO] Checkpoints: {args.save_root}")
    
    # ==================== Dataset ====================
    args.motion_dir = pjoin(args.data_root, 'npy')
    args.text_dir = pjoin(args.data_root, 'txt')
    args.max_motion_length = 360
    args.motion_rep = 'position'
    
    # Load mean/std
    stats_file_path = pjoin(ROOT, 'global_pipeline.pkl')
    print(f"[INFO] Loading stats from {stats_file_path}")
    try:
        pipeline = joblib.load(stats_file_path)
        scaler = pipeline.named_steps['stdscale']
        mean = scaler.data_mean_
        std = scaler.data_std_
        print("[INFO] ✓ Stats loaded")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Create splits
    train_split_file = pjoin(args.data_root, 'train.txt')
    val_split_file = pjoin(args.data_root, 'val.txt')
    create_train_split(args.motion_dir, train_split_file)
    create_val_split(train_split_file, val_split_file, val_ratio=0.1)

    # Dataloaders
    print("\n[INFO] Setting up dataloaders...")
    train_opt = DummyOpt(args, is_train=True)
    train_dataset = DatasetClass(train_opt, mean, std, train_split_file)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_opt = DummyOpt(args, is_train=False)
    val_dataset = DatasetClass(val_opt, mean, std, val_split_file)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"[INFO] ✓ Train: {len(train_dataset)} samples")
    print(f"[INFO] ✓ Val: {len(val_dataset)} samples")

    # ==================== Model ====================
    print("\n[INFO] Initializing VQ-VAE with Attention...")
    
    args.num_quantizers = args.num_codebooks
    args.shared_codebook = False
    args.quantize_dropout_prob = 0.0
    args.mu = 0.99

    print(f"[INFO] Architecture:")
    print(f"  - Input dim: {args.dim_pose}")
    print(f"  - Codebook size: {args.codebook_size}")
    print(f"  - Codebook dim: {args.codebook_dim}")
    print(f"  - Embed dim (KL): {args.embed_dim}")
    print(f"  - Num quantizers: {args.num_codebooks}")
    print(f"\n[INFO] Attention:")
    print(f"  - Enabled: {args.use_attention}")
    if args.use_attention:
        print(f"  - Type: {args.attention_type}")
        print(f"  - Num heads: {args.num_attention_heads}")
        print(f"  - Encoder layers: {args.encoder_attention_layers}")
        print(f"  - Decoder layers: {args.decoder_attention_layers}")
    print(f"\n[INFO] Skip Connections:")
    print(f"  - Enabled: {args.use_skip_connections}")
    if args.use_skip_connections:
        print(f"  - Type: {args.skip_connection_type}")
    print(f"\n[INFO] Loss weights:")
    print(f"  - Reconstruction: 1.0 ({args.recons_loss})")
    print(f"  - Velocity: {args.loss_vel}")
    print(f"  - VQ Commitment: {args.commit}")
    print(f"  - KL: {args.kl_weight}")

    model = RVQVAEWithAttention(
        args,
        input_width=args.dim_pose,
        nb_code=args.codebook_size,
        code_dim=args.codebook_dim,
        embed_dim=args.embed_dim,
        output_emb_width=args.codebook_dim,
        down_t=3,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        double_z=args.double_z,
        # Attention parameters
        use_attention=args.use_attention,
        attention_type=args.attention_type,
        num_attention_heads=args.num_attention_heads,
        encoder_attention_layers=args.encoder_attention_layers,
        decoder_attention_layers=args.decoder_attention_layers,
        # Skip connection parameters
        use_skip_connections=args.use_skip_connections,
        skip_connection_type=args.skip_connection_type
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] ✓ Model: {total_params/1e6:.2f}M parameters")

    # ==================== Trainer ====================
    print("\n[INFO] Initializing Trainer...")
    
    trainer = RVQTokenizerTrainer(
        args,
        model,
        kl_weight=args.kl_weight,
        use_posterior_sample=args.use_posterior_sample
    )
    
    print("[INFO] ✓ Trainer ready")

    # ==================== Training ====================
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Max epochs: {args.max_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"Mode: {'Continue' if args.is_continue else 'From scratch'}")
    print("="*70 + "\n")
    
    trainer.train(
        train_loader,
        val_loader=val_loader,
        eval_val_loader=None,
        eval_wrapper=None,
        plot_eval=None
    )
    
    # ==================== Complete ====================
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Model: {args.model_dir}")
    print(f"Logs: {args.log_dir}")
    print("\nNext steps:")
    print(f"1. Calculate scale_factor:")
    print(f"   python tools/calculate_scale_factor.py --vqkl_name {args.name}")
    print(f"2. Compare with baseline (without attention):")
    print(f"   Check reconstruction quality improvement")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()