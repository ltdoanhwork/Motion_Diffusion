# import os
# import torch
# import numpy as np
# import argparse
# from os.path import join as pjoin
# import sys
# sys.path.append('/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/text2motion')
# import utils.paramUtil as paramUtil
# from torch.utils.data import DataLoader
# from utils.plot_script import *
# from utils.get_opt import get_opt
# from datasets.evaluator_models import MotionLenEstimatorBiGRU

# from trainers import DDPMTrainer
# from models import MotionTransformer
# from utils.word_vectorizer import WordVectorizer, POS_enumerator
# from utils.utils import *
# from utils.motion_process import recover_from_ric


# def plot_t2m(data, result_path, npy_path, caption):
#     joint = recover_from_ric(torch.from_numpy(data).float(), opt.joints_num).numpy()
#     print(joint.shape)
#     joint = motion_temporal_filter(joint, sigma=1)
#     plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=20)
#     if npy_path != "":
#         np.save(npy_path, joint)


# def build_models(opt):
#     encoder = MotionTransformer(
#         input_feats=opt.dim_pose,
#         num_frames=opt.max_motion_length,
#         num_layers=opt.num_layers,
#         latent_dim=opt.latent_dim,
#         no_clip=opt.no_clip,
#         no_eff=opt.no_eff)
#     return encoder


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--opt_path', type=str, help='Opt path')
#     parser.add_argument('--text', type=str, default="How are you doing today?", help='Text description for motion generation')
#     parser.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
#     parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Path to save generation result')
#     parser.add_argument('--npy_path', type=str, default="", help='Path to save 3D keypoints sequence')
#     parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
#     args = parser.parse_args()
    
#     device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
#     opt = get_opt(args.opt_path, device)
#     opt.do_denoise = True

#     assert opt.dataset_name == "t2m"
#     assert args.motion_length <= 196
#     opt.data_root = '/home/ltdoanh/jupyter/jupyter/ldtan/HumanML3D/HumanML3D'
#     opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
#     opt.text_dir = pjoin(opt.data_root, 'texts')
#     opt.joints_num = 22
#     opt.dim_pose = 263
#     dim_word = 300
#     dim_pos_ohot = len(POS_enumerator)
#     num_classes = 200 // opt.unit_length

#     mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
#     std = np.load(pjoin(opt.meta_dir, 'std.npy'))

#     encoder = build_models(opt).to(device)
#     trainer = DDPMTrainer(opt, encoder)
#     trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

#     trainer.eval_mode()
#     trainer.to(opt.device)

#     result_dict = {}
#     with torch.no_grad():
#         if args.motion_length != -1:
#             caption = [args.text]
#             m_lens = torch.LongTensor([args.motion_length]).to(device)
#             pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
#             motion = pred_motions[0].cpu().numpy()
#             print(motion.shape)

#             motion = motion * std + mean
#             title = args.text + " #%d" % motion.shape[0]
#             plot_t2m(motion, args.result_path, args.npy_path, title)

import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin
import sys

# Thêm Motion_Diffusion directory vào sys.path để import utils
motion_diffusion_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if motion_diffusion_dir not in sys.path:
    sys.path.insert(0, motion_diffusion_dir)

import utils.paramUtil as paramUtil
from utils.plot_script import *
from utils.get_opt import get_opt
from trainers import DDPMTrainer
from models import MotionTransformer
from utils.utils import *
from utils.motion_process import recover_from_ric

def build_models(opt, motion_length=None):
    # Nếu opt không có max_motion_length (thường với BEAT), dùng motion_length tuyến tính
    max_frames = getattr(opt, 'max_motion_length', motion_length or 300)
    
    encoder = MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=max_frames,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder

def visualize_motion(data, result_path, npy_path, caption, opt):
    """
    Hàm visualization đa năng cho T2M, KIT và BEAT
    """
    dataset_name = opt.dataset_name
    joints_num = opt.joints_num
    
    # --- XỬ LÝ DỮ LIỆU THEO TỪNG DATASET ---
    if dataset_name == 'beat':
        # BEAT: Dữ liệu là Position (XYZ) thuần túy -> Chỉ cần Reshape
        # data shape: (Seq_Len, 264) -> (Seq_Len, 88, 3)
        print(f"[Info] Processing BEAT data (Pure Position), Shape: {data.shape}")
        
        # Reshape về (Frame, Joints, 3)
        # Lưu ý: Cần đảm bảo data đã được denormalize (nhân std + mean) ở bên ngoài
        joint = data.reshape(data.shape[0], joints_num, 3)
        
        # Lấy kinematic chain của BEAT
        kinematic_chain = paramUtil.beat_kinematic_chain
        
    else:
        # T2M / KIT: Dữ liệu là RIC Features -> Cần recover_from_ric
        print(f"[Info] Processing {dataset_name} data (RIC Features), Shape: {data.shape}")
        
        # recover_from_ric trả về (Seq_Len, Joints, 3)
        joint = recover_from_ric(torch.from_numpy(data).float(), joints_num).numpy()
        
        # Lấy kinematic chain tương ứng
        if dataset_name == 't2m':
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            kinematic_chain = paramUtil.kit_kinematic_chain
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- VẼ VÀ LƯU ---
    print(f"[Info] Final Joint Shape for Plotting: {joint.shape}")
    print(f"[Debug] Joint min/max: {joint.min():.4f} / {joint.max():.4f}")
    print(f"[Debug] Joint std (movement): {joint.std():.6f}")
    
    # Lọc nhiễu (Temporal Filter) để chuyển động mượt hơn
    joint = motion_temporal_filter(joint, sigma=1)
    
    # Gọi hàm vẽ 3D (xuất ra .mp4 hoặc .gif tùy thư viện plot_script của bạn)
    print(f"[Info] Calling plot_3d_motion with kinematic_chain type: {type(kinematic_chain)}")
    plot_3d_motion(result_path, kinematic_chain, joint, title=caption, fps=20)
    
    # Lưu file .npy nếu cần
    if npy_path != "":
        np.save(npy_path, joint)
        print(f"[Info] Saved raw joints to {npy_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_path', type=str, required=True, help='Path to option file (opt.txt)')
    parser.add_argument('--text', type=str, default="How are you doing today?", help='Text prompt')
    parser.add_argument('--motion_length', type=int, default=60, help='Length of motion (frames)')
    parser.add_argument('--result_path', type=str, default="test_sample.gif", help='Output file path')
    parser.add_argument('--npy_path', type=str, default="", help='Save npy path')
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU ID")
    parser.add_argument('--model_path', type=str, default="", help="Path to specific .tar checkpoint (optional)")
    args = parser.parse_args()
    
    # Setup Device
    device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
    
    # Load Options
    opt = get_opt(args.opt_path, device)
    opt.do_denoise = True
    
    # --- CẤU HÌNH ĐỘNG DỰA TRÊN DATASET NAME TRONG OPT ---
    print(f"\n🚀 Starting Generation for dataset: {opt.dataset_name.upper()}")
    
    if opt.dataset_name == 't2m':
        opt.data_root = '/home/ltdoanh/jupyter/jupyter/ldtan/HumanML3D/HumanML3D'
        opt.joints_num = 22
        opt.dim_pose = 263
        if not hasattr(opt, 'max_motion_length'):
            opt.max_motion_length = 196  # T2M default
    
    elif opt.dataset_name == 'kit':
        opt.data_root = './datasets/KIT-ML' # Cập nhật path đúng của bạn
        opt.joints_num = 21
        opt.dim_pose = 251
        if not hasattr(opt, 'max_motion_length'):
            opt.max_motion_length = 196  # KIT default
        
    elif opt.dataset_name == 'beat':
        opt.data_root = './datasets/BEAT_numpy' # Cập nhật path đúng
        opt.joints_num = 88   # Đã cập nhật theo phân tích trước
        opt.dim_pose = 264    # 88 * 3
        if not hasattr(opt, 'max_motion_length'):
            opt.max_motion_length = 360  # BEAT default (khớp với checkpoint)
    
    else:
        raise ValueError(f"Unknown dataset name in opt: {opt.dataset_name}")

    # Cập nhật đường dẫn meta (Mean/Std)
    opt.meta_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'meta')
    
    # Load Mean/Std
    try:
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
        std = np.load(pjoin(opt.meta_dir, 'std.npy'))
        print(f"[Info] Loaded Mean/Std. Mean shape: {mean.shape}")
    except Exception as e:
        print(f"[Error] Failed to load Mean/Std from {opt.meta_dir}")
        sys.exit(1)

    # Build Model
    encoder = build_models(opt, motion_length=opt.max_motion_length).to(device)
    trainer = DDPMTrainer(opt, encoder)
    
    # Load Checkpoint
    if args.model_path != "":
        # Load file cụ thể nếu user truyền vào
        trainer.load(args.model_path)
    else:
        # Mặc định load epoch được chỉ định trong opt
        trainer.load(pjoin(opt.model_dir, opt.which_epoch + '.tar'))

    trainer.eval_mode()
    trainer.to(opt.device)

    # --- GENERATION LOOP ---
    with torch.no_grad():
        if args.motion_length != -1:
            caption = [args.text]
            # Dùng opt.max_motion_length cho trainer, không dùng args.motion_length
            m_lens = torch.LongTensor([opt.max_motion_length]).to(device)
            
            # Generate motion (Data đang ở dạng Normalized Features)
            pred_motions = trainer.generate(caption, m_lens, opt.dim_pose)
            motion = pred_motions[0].cpu().numpy() # Shape: (Seq, Dim)
            
            # Nếu args.motion_length khác opt.max_motion_length, cắt motion
            if args.motion_length < motion.shape[0]:
                print(f"[Info] Trimming motion from {motion.shape[0]} to {args.motion_length} frames")
                motion = motion[:args.motion_length]
            
            print(f"[Info] Generated raw motion shape: {motion.shape}")

            # Denormalize (Quan trọng cho cả 3 dataset)
            motion = motion * std + mean
            
            title = args.text + " #%d" % motion.shape[0]
            
            # Gọi hàm visualization đa năng
            visualize_motion(motion, args.result_path, args.npy_path, title, opt)
            
    print("✅ Done!")