import os
import sys
from os.path import join as pjoin
import argparse
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

# --- 1. THIẾT LẬP ĐƯỜNG DẪN ---
# Đảm bảo ROOT là thư mục cha của 'tools/' (ví dụ: /.../Motion_Diffusion)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Thêm pymo vào path để joblib có thể load pipeline
PYMO_DIR = os.path.join(ROOT, 'datasets', 'pymo')
if PYMO_DIR not in sys.path:
    sys.path.insert(0, PYMO_DIR)

# --- 2. IMPORT TỪ DỰ ÁN ---
from models.vq.model import RVQVAE
from trainers.vq_trainer import RVQTokenizerTrainer
from datasets.dataset import Beat2MotionDataset as DatasetClass
from utils.fixseed import fixseed


# --- 3. HÀM HỖ TRỢ TẠO SPLIT (Tương tự như trong train_vq_diffusion.py) ---

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
    """Tự động tạo val.txt bằng cách tách 10% từ train.txt"""
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
        val_size = 1 # Đảm bảo có ít nhất 1 sample
        
    val_lines = all_lines[:val_size]
    train_lines = all_lines[val_size:]

    # Ghi đè lại file train
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines) + '\n')
    # Ghi file val mới
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines) + '\n')
    
    # print(f"[INFO] Created {val_file} with {len(val_lines)} samples.")
    print(f"[INFO] {train_file} updated, now has {len(train_lines)} samples.")


# --- 4. LỚP OPTION GIẢ LẬP (Cho Dataset) ---

class DummyOpt:
    """Giả lập các options mà Beat2MotionDataset cần."""
    def __init__(self, args, is_train):
        self.motion_dir = args.motion_dir
        self.text_dir = args.text_dir
        self.max_motion_length = args.max_motion_length
        self.dataset_name = args.dataset_name
        self.motion_rep = args.motion_rep
        self.is_train = is_train
        self.meta_dir = args.meta_dir
        self.joints_num = args.joints_num # Cần cho vq_trainer

# --- 5. HÀM MAIN ---

def main():
    parser = argparse.ArgumentParser()
    
    # --- Paths ---
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Thư mục gốc lưu checkpoints')
    parser.add_argument('--name', type=str, default='VQVAE_BEAT', help='Tên của thử nghiệm này (để tạo thư mục checkpoint)')
    parser.add_argument('--dataset_name', type=str, default='beat', choices=['t2m', 'kit', 'beat'], help='Tên dataset')
    parser.add_argument('--data_root', type=str, default='./datasets/BEAT_numpy', help='Đường dẫn đến data BEAT đã xử lý (chứa npy/ và txt/)')
    
    # --- Model Params (VQ-VAE) ---
    parser.add_argument('--dim_pose', type=int, default=264, help='Kích thước vector motion (264 cho BEAT)')
    parser.add_argument('--codebook_dim', type=int, default=512, help='Kích thước của codebook embedding (latent_dim)')
    parser.add_argument('--num_codebooks', type=int, default=10, help='Số lượng codebook (cho RVQ)')
    parser.add_argument('--codebook_size', type=int, default=1024, help='Số lượng code (vector) trong mỗi codebook')
    parser.add_argument('--joints_num', type=int, default=55, help='Số khớp (55 cho BEAT)')

    # --- Training Params ---
    parser.add_argument('--max_epoch', type=int, default=100, help='Số epoch tối đa')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--is_continue', action='store_true', help='Tiếp tục training từ checkpoint "latest.tar"')
    parser.add_argument('--num_workers', type=int, default=4, help='Số luồng tải data')
    
    # --- Trainer Params (từ vq_trainer.py) ---
    parser.add_argument('--recons_loss', type=str, default='l1', choices=['l1', 'l1_smooth'], help='Loại loss tái tạo')
    parser.add_argument('--loss_vel', type=float, default=0.1, help='Trọng số cho loss_explicit (sẽ bị bỏ qua nếu bạn sửa vq_trainer)')
    parser.add_argument('--commit', type=float, default=0.02, help='Trọng số cho commitment loss')
    parser.add_argument('--warm_up_iter', type=int, default=1000, help='Số iter warm up cho learning rate')
    parser.add_argument('--milestones', type=int, nargs='+', default=[20, 40, 60], help='Epochs để giảm LR')
    parser.add_argument('--gamma', type=float, default=0.5, help='Hệ số giảm LR')
    
    # --- Logging ---
    parser.add_argument('--log_every', type=int, default=50, help='Log loss mỗi N iteration')
    parser.add_argument('--save_latest', type=int, default=500, help='Lưu checkpoint latest.tar mỗi N iteration')
    parser.add_argument('--eval_every_e', type=int, default=5, help='Chạy evaluation mỗi N epoch (nếu có)')
    
    # --- Device ---
    parser.add_argument('--gpu_id', type=int, default=0, help='ID của GPU')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')

    args = parser.parse_args()

    # --- 6. THIẾT LẬP HỆ THỐNG ---
    fixseed(args.seed)
    args.is_train = True
    args.device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")

    # --- 7. THIẾT LẬP THƯ MỤC ---
    args.save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    args.model_dir = pjoin(args.save_root, 'model')
    args.log_dir = pjoin(args.save_root, 'logs')
    args.eval_dir = pjoin(args.save_root, 'eval')
    args.meta_dir = pjoin(args.save_root, 'meta') # Để trainer lưu mean/std (nếu cần)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(args.meta_dir, exist_ok=True)
    
    # --- 8. CẤU HÌNH DATASET (BEAT) ---
    args.motion_dir = pjoin(args.data_root, 'npy')
    args.text_dir = pjoin(args.data_root, 'txt')
    args.max_motion_length = 360  # ~6 giây @ 60fps
    args.motion_rep = 'position'  # Phải khớp với pipeline tiền xử lý
    
    # --- 9. TẢI MEAN/STD TỪ PIPELINE ---
    stats_file_path = pjoin(ROOT, 'global_pipeline.pkl')
    print(f"[INFO] Loading stats from {stats_file_path}")
    try:
        pipeline = joblib.load(stats_file_path)
        scaler = pipeline.named_steps['stdscale']
        mean = scaler.data_mean_
        std = scaler.data_std_
        print("[INFO] Mean and Std loaded successfully from pipeline.")
    except Exception as e:
        print(f"[ERROR] Could not load mean/std from {stats_file_path}: {e}")
        print("Exiting.")
        sys.exit(1)

    # --- 10. TẠO DATA LOADER ---
    train_split_file = pjoin(args.data_root, 'train.txt')
    val_split_file = pjoin(args.data_root, 'val.txt')

    # Tự động tạo split nếu chưa có
    create_train_split(args.motion_dir, train_split_file)
    create_val_split(train_split_file, val_split_file, val_ratio=0.1)

    # Tạo các "opt" giả lập cho dataset
    train_opt = DummyOpt(args, is_train=True)
    val_opt = DummyOpt(args, is_train=False)

    train_dataset = DatasetClass(train_opt, mean, std, train_split_file)
    val_dataset = DatasetClass(val_opt, mean, std, val_split_file)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    # print(f"Validation dataset: {len(val_dataset)} samples")

    # --- 11. KHỞI TẠO MÔ HÌNH VQ-VAE ---
    # --- 11. KHỞI TẠO MÔ HÌNH VQ-VAE ---
    # Cập nhật các tham số cho RVQVAE từ 'args'
    # (Vì RVQVAE cần 'args' để cấu hình ResidualVQ)
    args.num_quantizers = args.num_codebooks  # Đổi tên cho nhất quán
    args.shared_codebook = False # Giả định (bạn có thể thêm vào argparse nếu cần)
    args.quantize_dropout_prob = 0.0 # Giả định
    args.mu = 0.99  # <--- THÊM DÒNG NÀY (0.99 là giá trị EMA mặc định phổ biến)

    model = RVQVAE(
        args,  # <--- TRUYỀN TOÀN BỘ 'args' VÀO ĐÂY
        input_width=args.dim_pose,  # <--- Sửa tên
        nb_code=args.codebook_size,  # <--- Sửa tên
        code_dim=args.codebook_dim,  # <--- Sửa tên
        output_emb_width=args.codebook_dim, # Đảm bảo khớp với code_dim
        down_t=3,
        stride_t=2,
        width=512,
        depth=3, 
        dilation_growth_rate=3,
        activation='relu',
        norm=None
        )
    
    print(f"VQ-VAE Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    # --- 12. KHỞI TẠO TRAINER ---
    # args (từ argparse) được truyền vào, trainer sẽ tự lấy các giá trị
    trainer = RVQTokenizerTrainer(args, model)

    # --- 13. HUẤN LUYỆN ---
    print("Starting VQ-VAE Training...")
    trainer.train(
        train_loader, 
        val_loader,
        # ***LƯU Ý QUAN TRỌNG VỀ EVALUATION***
        # Truyền None cho các tham số eval của t2m
        eval_val_loader=None, 
        eval_wrapper=None, 
        plot_eval=None
    )

if __name__ == "__main__":
    main()