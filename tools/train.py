# train_beat.py
import numpy as np, torch
from os.path import join as pjoin
import os, sys
# đảm bảo này trỏ tới project root (cha của tools/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# project root ──┐
#                └─ text2motion/

import utils.paramUtil as paramUtil
from options.train_options import TrainCompOptions
from utils.plot_script import *      # noqa

from models   import MotionTransformer
from trainers import DDPMTrainer
from datasets import Beat2MotionDataset  # t2m / kit
from datasets.dataset import Text2MotionDataset

# -----------------------------------------------------------
# helper
# -----------------------------------------------------------
def build_model(opt, dim_pose):
    return MotionTransformer(
        input_feats   = dim_pose,
        num_frames    = opt.max_motion_length,
        num_layers    = opt.num_layers,
        latent_dim    = opt.latent_dim,
        no_clip       = opt.no_clip,
        no_eff        = opt.no_eff,
    )

def validate_dataset(dataset):
    bad_samples = []
    for i in range(len(dataset)):
        try:
            sample = dataset[i]
            if sample is None:
                raise ValueError("Sample is None")
        except Exception as e:
            print(f"[Dataset ERROR] idx={i}, name={dataset.name_list[i]}: {e}")
            bad_samples.append(dataset.name_list[i])
    print(f"✅ Tổng số lỗi: {len(bad_samples)} / {len(dataset)}")
    return bad_samples

def create_train_split(motion_dir, split_file):
    """Create train.txt by listing all .npy files in motion_dir.
    
    Args:
        motion_dir: Path to directory containing .npy motion files
        split_file: Path to output train.txt
    """
    if os.path.exists(split_file):
        print(f"[INFO] {split_file} already exists")
        return
        
    os.makedirs(os.path.dirname(split_file), exist_ok=True)
    # Walk recursively so we pick up .npy files inside subfolders (e.g. npy/1/*.npy)
    motion_files = []
    for root, _, files in os.walk(motion_dir):
        for f in files:
            if f.endswith('.npy'):
                full = os.path.join(root, f)
                # write relative path from motion_dir, without extension
                rel = os.path.relpath(full, motion_dir)
                name = os.path.splitext(rel)[0]
                # normalize to use os.sep (Dataset will join with motion_dir)
                motion_files.append(name)
    
    if not motion_files:
        raise ValueError(f"No .npy files found in {motion_dir}")
        
    print(f"Found {len(motion_files)} motion files")
    # Sort for deterministic ordering; use os.path.normpath before sort to keep consistent
    motion_files_sorted = sorted(motion_files, key=lambda x: x.replace('\\', '/'))
    with open(split_file, 'w', encoding='utf-8') as f:
        for name in motion_files_sorted:
            f.write(f"{name}\n")
    print(f"Created {split_file}")

# -----------------------------------------------------------
# main
# -----------------------------------------------------------
if __name__ == '__main__':
    parser = TrainCompOptions()
    opt    = parser.parse()

    opt.device = torch.device('cuda')
    torch.autograd.set_detect_anomaly(True)

    # -------------------------------------------------------
    # dataset‑specific paths & hyper‑params
    # -------------------------------------------------------
    if opt.dataset_name == 't2m':
        opt.data_root       = './datasets/HumanML3D/HumanML3D'
        opt.motion_dir      = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir        = pjoin(opt.data_root, 'texts')
        opt.joints_num      = 22
        fps                 = 20
        opt.max_motion_length = 196
        dim_pose            = 263
        kinematic_chain     = paramUtil.t2m_kinematic_chain
        DatasetClass        = Text2MotionDataset

    elif opt.dataset_name == 'kit':
        opt.data_root       = './datasets/KIT-ML'
        opt.motion_dir      = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir        = pjoin(opt.data_root, 'texts')
        opt.joints_num      = 21
        fps                 = 12.5
        opt.max_motion_length = 196
        dim_pose            = 251
        kinematic_chain     = paramUtil.kit_kinematic_chain
        DatasetClass        = Text2MotionDataset

    elif opt.dataset_name == 'beat':                           # ← NEW BRANCH
        opt.data_root       = pjoin(ROOT, 'datasets/BEAT_test')   # absolute path
        opt.motion_dir      = pjoin(opt.data_root, 'npy')
        opt.text_dir        = pjoin(opt.data_root, 'txt')
        opt.joints_num      = 55
        fps                 = 60
        # choose a workable clip length for diffusion (e.g. ~6 s = 360 frames)
        opt.max_motion_length = 360
        dim_pose            = 264                                # axis‑angle for 55 joints
        # kinematic_chain     = paramUtil.beat_kinematic_chain     # add in paramUtil
        DatasetClass        = Beat2MotionDataset

    else:
        raise KeyError(f'Unknown dataset {opt.dataset_name}')

    # -------------------------------------------------------
    # bookkeeping
    # -------------------------------------------------------
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir  = pjoin(opt.save_root, 'meta')
    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir,  exist_ok=True)

    # -------------------------------------------------------
    # mean / std  (computed automatically for BEAT first run)
    # -------------------------------------------------------
    mean_path = pjoin(opt.meta_dir, 'mean.npy')
    std_path  = pjoin(opt.meta_dir, 'std.npy')
    mean = np.load(mean_path) if os.path.exists(mean_path) else None
    std  = np.load(std_path)  if os.path.exists(std_path)  else None

    # -------------------------------------------------------
    # build model & trainer
    # -------------------------------------------------------
    encoder = build_model(opt, dim_pose).to(opt.device)
    trainer = DDPMTrainer(opt, encoder)

    # -------------------------------------------------------
    # dataset & loader
    # -------------------------------------------------------
    train_split = pjoin(opt.data_root, 'train.txt')             # ids list
    
    # Auto-create train.txt if not exists (for BEAT dataset)
    if opt.dataset_name == 'beat':
        create_train_split(opt.motion_dir, train_split)

    train_set   = DatasetClass(opt, mean, std, train_split, opt.times)
    bad_ids = validate_dataset(train_set)
    trainer.train(train_set)
    print("🚀 Training complete!")
