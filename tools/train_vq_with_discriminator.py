"""
Training script for Enhanced VQ-KL Autoencoder with:
- GAN + Perceptual Loss (LPIPS)
- Hierarchical Weighted Loss
- Sobolev Regularization
- Spectral Loss
- Geometric Consistency
"""
import os
import sys
from os.path import join as pjoin
import argparse
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

# --- 1. THIẾT LẬP ĐƯỜNG DẪN ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PYMO_DIR = os.path.join(ROOT, 'datasets', 'pymo')
if PYMO_DIR not in sys.path:
    sys.path.insert(0, PYMO_DIR)

# --- 2. IMPORT TỪ DỰ ÁN ---
from models.vq.model import RVQVAE
from trainers.vq_trainer_gan import (
    RVQTokenizerTrainerGAN,
    create_trainer_from_config,
    get_smpl_bone_pairs,
    get_hand_joint_indices
)
from datasets.dataset import Beat2MotionDataset as DatasetClass
from utils.fixseed import fixseed


def create_train_split(motion_dir, split_file):
    """Tự động tạo train.txt từ tất cả file .npy"""
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
    """Tự động tạo val.txt bằng cách tách từ train.txt"""
    if os.path.exists(val_file):
        print(f"[INFO] {val_file} already exists")
        return
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"{train_file} not found. Cannot create val split.")

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
    """Giả lập các options mà Beat2MotionDataset cần"""
    def __init__(self, args, is_train):
        self.motion_dir = args.motion_dir
        self.text_dir = args.text_dir
        self.max_motion_length = args.max_motion_length
        self.dataset_name = args.dataset_name
        self.motion_rep = args.motion_rep
        self.is_train = is_train
        self.joints_num = args.joints_num


def main():
    parser = argparse.ArgumentParser(description='Train Enhanced VQ-KL with GAN')
    
    # ==================== Paths ====================
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Thư mục gốc lưu checkpoints')
    parser.add_argument('--name', type=str, default='VQKL_Enhanced_BEAT',
                        help='Tên thử nghiệm')
    parser.add_argument('--dataset_name', type=str, default='beat', 
                        choices=['t2m', 'kit', 'beat'])
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy',
                        help='Đường dẫn data BEAT')
    
    # ==================== VQ-KL Model ====================
    parser.add_argument('--dim_pose', type=int, default=264)
    parser.add_argument('--codebook_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--num_codebooks', type=int, default=10)
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--joints_num', type=int, default=55)
    parser.add_argument('--double_z', action='store_true', default=True)
    parser.add_argument('--use_posterior_sample', action='store_true', default=True)
    
    # ==================== Training ====================
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='Tăng epochs vì có discriminator')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Giảm batch size do GAN tốn memory')
    parser.add_argument('--lr', type=float, default=4.5e-6,
                        help='LR cho generator')
    parser.add_argument('--lr_disc', type=float, default=4.5e-6,
                        help='LR cho discriminator')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--is_continue', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # ==================== Base Loss Weights ====================
    parser.add_argument('--recons_loss', type=str, default='l1', 
                        choices=['l1', 'l1_smooth', 'l2'])
    parser.add_argument('--commit', type=float, default=0.02,
                        help='VQ commitment loss weight')
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='KL divergence weight')
    
    # ==================== GAN Loss Weights ====================
    parser.add_argument('--disc_start', type=int, default=10000,
                        help='Iteration để bắt đầu train discriminator')
    parser.add_argument('--disc_weight', type=float, default=0.75,
                        help='Discriminator loss weight')
    parser.add_argument('--disc_factor', type=float, default=1.0)
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                        help='LPIPS perceptual loss weight')
    parser.add_argument('--disc_loss', type=str, default='hinge',
                        choices=['hinge', 'vanilla'])
    parser.add_argument('--disc_num_layers', type=int, default=2,
                        help='Số layers trong discriminator')
    parser.add_argument('--disc_in_channels', type=int, default=264,
                        help='Input channels cho discriminator (= dim_pose)')
    
    # ==================== Enhanced Loss Parameters ====================
    # Hierarchical Loss
    parser.add_argument('--use_hierarchical_loss', action='store_true', default=True,
                        help='Enable hierarchical weighting for hand joints')
    parser.add_argument('--hand_loss_weight', type=float, default=3.0,
                        help='Weight multiplier for hand reconstruction loss (α)')
    parser.add_argument('--skeleton_type', type=str, default='smpl',
                        choices=['smpl', 'cmu'],
                        help='Skeleton type for joint mapping')
    
    # Sobolev Regularization
    parser.add_argument('--lambda_vel', type=float, default=0.5,
                        help='Weight for velocity loss (first derivative)')
    parser.add_argument('--lambda_acc', type=float, default=0.3,
                        help='Weight for acceleration loss (second derivative)')
    
    # Spectral Loss
    parser.add_argument('--lambda_spectral', type=float, default=0.1,
                        help='Weight for spectral/frequency domain loss')
    
    # Geometric Consistency
    parser.add_argument('--use_bone_loss', action='store_true', default=True,
                        help='Enable bone length consistency loss')
    parser.add_argument('--lambda_bone', type=float, default=0.15,
                        help='Weight for bone length preservation loss')
    
    # ==================== Training Schedule ====================
    parser.add_argument('--warm_up_iter', type=int, default=1000)
    parser.add_argument('--milestones', type=int, nargs='+', default=[30, 40])
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
    
    print("\n" + "="*80)
    print("🚀 ENHANCED VQ-KL + GAN TRAINING")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Architecture: VQ + KL + LPIPS + Discriminator")
    print(f"Enhanced with:")
    print(f"  ✓ Hierarchical Loss (hands)")
    print(f"  ✓ Sobolev Regularization (velocity + acceleration)")
    print(f"  ✓ Spectral Loss (frequency domain)")
    if args.use_bone_loss:
        print(f"  ✓ Geometric Consistency (bone lengths)")
    print(f"Dataset: {args.dataset_name.upper()}")
    print("="*80 + "\n")

    # ==================== Directories ====================
    args.save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    args.model_dir = pjoin(args.save_root, 'model')
    args.log_dir = pjoin(args.save_root, 'logs')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
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
    print("\n[INFO] Initializing VQ-KL Model...")
    args.num_quantizers = args.num_codebooks
    args.shared_codebook = False
    args.quantize_dropout_prob = 0.0
    args.mu = 0.99

    print(f"[INFO] Architecture:")
    print(f"  - VQ codebook size: {args.codebook_size}")
    print(f"  - VQ codebook dim: {args.codebook_dim}")
    print(f"  - KL embed dim: {args.embed_dim}")
    print(f"  - Num quantizers: {args.num_codebooks}")

    model = RVQVAE(
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
        double_z=args.double_z
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] ✓ Model: {total_params/1e6:.2f}M parameters")

    # ==================== Enhanced Trainer ====================
    print("\n[INFO] Initializing Enhanced Trainer with GAN...")
    print(f"\n[INFO] Loss Configuration:")
    print(f"  Base Losses:")
    print(f"    - Reconstruction: 1.0 ({args.recons_loss})")
    print(f"    - VQ Commitment: {args.commit}")
    print(f"    - KL Divergence: {args.kl_weight}")
    print(f"  GAN Losses:")
    print(f"    - Discriminator: {args.disc_weight} (starts at iter {args.disc_start})")
    print(f"    - Perceptual (LPIPS): {args.perceptual_weight}")
    print(f"  Enhanced Losses:")
    if args.use_hierarchical_loss:
        print(f"    - Hierarchical (hand α): {args.hand_loss_weight}")
    print(f"    - Sobolev Velocity: {args.lambda_vel}")
    print(f"    - Sobolev Acceleration: {args.lambda_acc}")
    print(f"    - Spectral (FFT): {args.lambda_spectral}")
    if args.use_bone_loss:
        bone_pairs = get_smpl_bone_pairs() if args.skeleton_type == 'smpl' else []
        print(f"    - Geometric ({len(bone_pairs)} bones): {args.lambda_bone}")
    
    # Use factory function to create trainer
    trainer = create_trainer_from_config(args, model)
    
    print("\n[INFO] ✓ Enhanced Trainer ready!")

    # ==================== Training ====================
    print("\n" + "="*80)
    print("📊 STARTING TRAINING")
    print("="*80)
    print(f"Max epochs: {args.max_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Generator LR: {args.lr} (Cosine Annealing)")
    print(f"Discriminator LR: {args.lr_disc} (Cosine Annealing)")
    print(f"Mode: {'Continue' if args.is_continue else 'From scratch'}")
    print("="*80 + "\n")
    
    trainer.train(
        train_loader,
        val_loader=val_loader
    )
    
    # ==================== Complete ====================
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED!")
    print("="*80)
    print(f"Model saved: {args.model_dir}")
    print(f"Logs saved: {args.log_dir}")
    print("\n📝 Next steps:")
    print(f"1. Calculate scale_factor:")
    print(f"   python tools/calculate_scale_factor.py --vqkl_name {args.name}")
    print(f"2. Train diffusion:")
    print(f"   python tools/train_vqkl_diffusion.py --vqkl_name {args.name}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()