"""
Training script for VQ-KL Autoencoder (Hybrid VQ-VAE + KL Divergence)
Combines discrete vector quantization with continuous KL regularization
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
from trainers.vq_trainer import RVQTokenizerTrainer
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

    # Ghi đè lại file train
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines) + '\n')
    # Ghi file val mới
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
    parser = argparse.ArgumentParser(description='Train VQ-KL Autoencoder (Hybrid VQ + KL)')
    
    # ==================== Paths ====================
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                        help='Thư mục gốc lưu checkpoints')
    parser.add_argument('--name', type=str, default='VQKL_BEAT',
                        help='Tên thử nghiệm (VQ-KL thay vì VQ-VAE)')
    parser.add_argument('--dataset_name', type=str, default='beat', 
                        choices=['t2m', 'kit', 'beat'],
                        help='Tên dataset')
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy',
                        help='Đường dẫn data BEAT (chứa npy/ và txt/)')
    
    # ==================== VQ-KL Model Params ====================
    parser.add_argument('--dim_pose', type=int, default=264,
                        help='Kích thước vector motion (264 cho BEAT)')
    parser.add_argument('--codebook_dim', type=int, default=512,
                        help='Kích thước codebook embedding (cho VQ)')
    parser.add_argument('--embed_dim', type=int, default=512,
                        help='Kích thước embedding cho KL posterior (continuous latent)')
    parser.add_argument('--num_codebooks', type=int, default=10,
                        help='Số lượng codebook (cho RVQ)')
    parser.add_argument('--codebook_size', type=int, default=1024,
                        help='Số code trong mỗi codebook')
    parser.add_argument('--joints_num', type=int, default=55,
                        help='Số khớp (55 cho BEAT)')
    
    # ==================== VQ-KL Specific ====================
    parser.add_argument('--double_z', action='store_true', default=True,
                        help='Encode thành mean + logvar cho KL (bắt buộc cho VQ-KL)')
    parser.add_argument('--use_posterior_sample', action='store_true', default=True,
                        help='Sample từ posterior thay vì dùng mean')
    
    # ==================== Training Params ====================
    parser.add_argument('--max_epoch', type=int, default=25,
                        help='Số epoch tối đa')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--is_continue', action='store_true',
                        help='Tiếp tục từ checkpoint latest.tar')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Số workers cho dataloader')
    
    # ==================== Loss Weights (VQ-KL) ====================
    parser.add_argument('--recons_loss', type=str, default='l1', 
                        choices=['l1', 'l1_smooth', 'l2'],
                        help='Loại reconstruction loss')
    parser.add_argument('--loss_vel', type=float, default=0.1,
                        help='Trọng số velocity loss (deprecated, có thể bỏ qua)')
    parser.add_argument('--commit', type=float, default=0.02,
                        help='Trọng số commitment loss (cho VQ)')
    parser.add_argument('--kl_weight', type=float, default=1e-6,
                        help='Trọng số KL divergence loss (CRITICAL cho VQ-KL)')
    parser.add_argument('--perceptual_weight', type=float, default=0.0,
                        help='Trọng số perceptual loss (optional)')
    
    # ==================== Training Schedule ====================
    parser.add_argument('--warm_up_iter', type=int, default=1000,
                        help='Số iter warm up cho learning rate')
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 40, 60],
                        help='Epochs để giảm LR')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Hệ số giảm LR tại milestones')
    
    # ==================== Logging ====================
    parser.add_argument('--log_every', type=int, default=50,
                        help='Log loss mỗi N iterations')
    parser.add_argument('--save_latest', type=int, default=500,
                        help='Lưu latest.tar mỗi N iterations')
    parser.add_argument('--eval_every_e', type=int, default=5,
                        help='Evaluate mỗi N epochs')
    
    # ==================== Device ====================
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--seed', type=int, default=3407,
                        help='Random seed')

    args = parser.parse_args()

    # ==================== Setup System ====================
    fixseed(args.seed)
    args.is_train = True
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*70)
    print("VQ-KL AUTOENCODER TRAINING")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Architecture: Hybrid VQ + KL Divergence")
    print(f"Dataset: {args.dataset_name.upper()}")
    print("="*70 + "\n")

    # ==================== Setup Directories ====================
    args.save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    args.model_dir = pjoin(args.save_root, 'model')
    args.log_dir = pjoin(args.save_root, 'logs')

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"[INFO] Checkpoints: {args.save_root}")
    
    # ==================== Dataset Config ====================
    args.motion_dir = pjoin(args.data_root, 'npy')
    args.text_dir = pjoin(args.data_root, 'txt')
    args.max_motion_length = 360  # ~6 seconds @ 60fps
    args.motion_rep = 'position'
    
    # ==================== Load Mean/Std ====================
    stats_file_path = pjoin(ROOT, 'global_pipeline.pkl')
    print(f"[INFO] Loading normalization stats from {stats_file_path}")
    try:
        pipeline = joblib.load(stats_file_path)
        scaler = pipeline.named_steps['stdscale']
        mean = scaler.data_mean_
        std = scaler.data_std_
        print("[INFO] ✓ Mean and Std loaded successfully")
    except Exception as e:
        print(f"[ERROR] Could not load mean/std: {e}")
        print("Exiting.")
        sys.exit(1)

    # ==================== Create Data Splits ====================
    train_split_file = pjoin(args.data_root, 'train.txt')
    val_split_file = pjoin(args.data_root, 'val.txt')

    create_train_split(args.motion_dir, train_split_file)
    create_val_split(train_split_file, val_split_file, val_ratio=0.1)

    # ==================== Setup Dataloaders ====================
    print("\n[INFO] Setting up dataloaders...")
    
    # Train loader
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

    # Validation loader
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
    
    print(f"[INFO] ✓ Train dataset: {len(train_dataset)} samples")
    print(f"[INFO] ✓ Validation dataset: {len(val_dataset)} samples")

    # ==================== Initialize VQ-KL Model ====================
    print("\n[INFO] Initializing VQ-KL Autoencoder...")
    
    # Setup quantizer args
    args.num_quantizers = args.num_codebooks 
    args.shared_codebook = False 
    args.quantize_dropout_prob = 0.0 
    args.mu = 0.99

    # Print architecture details
    print(f"[INFO] Architecture configuration:")
    print(f"  - Input dim: {args.dim_pose}")
    print(f"  - Codebook size: {args.codebook_size}")
    print(f"  - Codebook dim (VQ): {args.codebook_dim}")
    print(f"  - Embed dim (KL): {args.embed_dim}")
    print(f"  - Num quantizers: {args.num_codebooks}")
    print(f"  - Double z (mean+logvar): {args.double_z}")
    print(f"  - Temporal downsampling: 8x (down_t=3)")
    print(f"\n[INFO] Loss weights:")
    print(f"  - Reconstruction: 1.0 ({args.recons_loss})")
    print(f"  - Commitment: {args.commit}")
    print(f"  - KL divergence: {args.kl_weight}")
    if args.perceptual_weight > 0:
        print(f"  - Perceptual: {args.perceptual_weight}")

    # Create VQ-KL model
    model = RVQVAE(
        args, 
        input_width=args.dim_pose, 
        nb_code=args.codebook_size, 
        code_dim=args.codebook_dim,
        embed_dim=args.embed_dim,  # For KL posterior
        output_emb_width=args.codebook_dim, 
        down_t=3,  # 360 -> 45 frames (8x downsampling)
        stride_t=2,
        width=512,
        depth=3, 
        dilation_growth_rate=3,
        activation='relu',
        norm=None,
        double_z=args.double_z  # Enable KL: encode to mean + logvar
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[INFO] ✓ Model initialized: {total_params/1e6:.2f}M parameters")

    # ==================== Initialize Trainer ====================
    print("\n[INFO] Initializing VQ-KL Trainer...")
    
    # Pass KL-specific parameters to trainer
    trainer = RVQTokenizerTrainer(
        args, 
        model,
        kl_weight=args.kl_weight,  # Critical for VQ-KL
        use_posterior_sample=args.use_posterior_sample
    )
    
    print("[INFO] ✓ Trainer ready")

    # ==================== Start Training ====================
    print("\n" + "="*70)
    print("STARTING VQ-KL TRAINING")
    print("="*70)
    print(f"Max epochs: {args.max_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Initial LR: {args.lr}")
    print(f"LR milestones: {args.milestones}")
    print(f"Mode: {'Continue' if args.is_continue else 'From scratch'}")
    print("="*70 + "\n")
    
    trainer.train(
        train_loader, 
        val_loader=val_loader,
        eval_val_loader=None, 
        eval_wrapper=None, 
        plot_eval=None
    )
    
    # ==================== Training Complete ====================
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Model saved to: {args.model_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print("\nNext steps:")
    print("1. Calculate scale_factor:")
    print(f"   python tools/calculate_scale_factor.py --vqkl_name {args.name}")
    print("2. Train diffusion model:")
    print(f"   python tools/train_vqkl_diffusion.py --vqkl_name {args.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()