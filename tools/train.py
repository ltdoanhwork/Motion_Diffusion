# train_beat.py
import numpy as np, torch
from os.path import join as pjoin
import os, sys
import argparse

try:
    import yaml
except ImportError:
    yaml = None

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
# config helper
# -----------------------------------------------------------
def _to_cli_args_from_yaml(cfg):
    if not isinstance(cfg, dict):
        raise ValueError("YAML config must be a mapping/dictionary at top level.")

    cli_args = []
    args_map = cfg.get("args", {})
    flags_map = cfg.get("flags", {})

    if args_map is not None:
        if not isinstance(args_map, dict):
            raise ValueError("Field 'args' in YAML must be a mapping/dictionary.")
        for key, value in args_map.items():
            opt = f"--{key}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(opt)
                continue
            if isinstance(value, list):
                cli_args.append(opt)
                cli_args.extend([str(v) for v in value])
                continue
            if value is None:
                continue
            cli_args.extend([opt, str(value)])

    if flags_map is not None:
        if not isinstance(flags_map, dict):
            raise ValueError("Field 'flags' in YAML must be a mapping/dictionary.")
        for key, enabled in flags_map.items():
            if bool(enabled):
                cli_args.append(f"--{key}")

    return cli_args


def _merge_config_into_argv(argv):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_opt, remaining = pre_parser.parse_known_args(argv)

    if not pre_opt.config:
        return remaining

    if yaml is None:
        raise ImportError(
            "PyYAML is required for --config support. Install with: pip install pyyaml"
        )
    if not os.path.exists(pre_opt.config):
        raise FileNotFoundError(f"Config file not found: {pre_opt.config}")

    with open(pre_opt.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg_args = _to_cli_args_from_yaml(cfg)
    print(f"[INFO] Loaded config: {pre_opt.config}")
    return cfg_args + remaining


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
    print(f" Tổng số lỗi: {len(bad_samples)} / {len(dataset)}")
    return bad_samples

def create_train_test_splits(motion_dir, train_file, test_file, test_ratio=0.1, split_seed=3407):
    """Create train.txt and test.txt by scanning .npy files and splitting by ratio."""
    if os.path.exists(train_file) and os.path.exists(test_file):
        print(f"[INFO] {train_file} and {test_file} already exist")
        return

    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    os.makedirs(os.path.dirname(train_file), exist_ok=True)

    # Walk recursively so we pick up .npy files inside subfolders (e.g. npy/1/*.npy)
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

    motion_files = sorted(set(motion_files))
    print(f"Found {len(motion_files)} motion files")

    rng = np.random.RandomState(split_seed)
    shuffled_idx = rng.permutation(len(motion_files))
    n_test = max(1, int(round(len(motion_files) * test_ratio)))
    n_test = min(n_test, len(motion_files) - 1)

    test_idx = set(shuffled_idx[:n_test].tolist())
    train_ids = [motion_files[i] for i in range(len(motion_files)) if i not in test_idx]
    test_ids = [motion_files[i] for i in range(len(motion_files)) if i in test_idx]

    with open(train_file, 'w', encoding='utf-8') as f:
        for name in train_ids:
            f.write(f"{name}\n")

    with open(test_file, 'w', encoding='utf-8') as f:
        for name in test_ids:
            f.write(f"{name}\n")

    print(f"Created {train_file} ({len(train_ids)} samples)")
    print(f"Created {test_file} ({len(test_ids)} samples)")

def check_data_shape(motion_dir):
    import glob
    files = glob.glob(os.path.join(motion_dir, "**/*.npy"), recursive=True)
    if not files:
        print("Không tìm thấy file .npy nào!")
        return 0, 0
    
    data = np.load(files[0])
    print(f"1. Kiểm tra file: {files[0]}")
    print(f"2. Shape gốc: {data.shape}") 
    return data.shape[1] 


def infer_beat_joints_num(dim_pose, motion_rep):
    rep = str(motion_rep).lower()
    if rep in ("rot6d", "rotation_6d", "6d"):
        if dim_pose % 6 == 0:
            return dim_pose // 6
        if (dim_pose - 3) % 6 == 0:
            return (dim_pose - 3) // 6
        return None

    if rep in ("axis_angle", "position", "rep15d"):
        if dim_pose % 3 == 0 and (dim_pose // 3) in (75, 88):
            return dim_pose // 3
        if dim_pose % 6 == 0 and (dim_pose // 6) in (75, 88):
            return dim_pose // 6
        if (dim_pose - 3) % 6 == 0 and ((dim_pose - 3) // 6) in (75, 88):
            return (dim_pose - 3) // 6
    return None

# -----------------------------------------------------------
# main
# -----------------------------------------------------------
if __name__ == '__main__':
    merged_argv = _merge_config_into_argv(sys.argv[1:])
    sys.argv = [sys.argv[0]] + merged_argv

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

    elif opt.dataset_name == 'beat':                          
        opt.data_root       = pjoin(ROOT, 'datasets/BEAT_numpy')   
        opt.motion_dir      = pjoin(opt.data_root, 'npy')
        opt.text_dir        = pjoin(opt.data_root, 'txt')
        fps                 = 60
        opt.max_motion_length = 360
        real_dim_pose = check_data_shape(opt.motion_dir)
        if real_dim_pose > 0:
            dim_pose = real_dim_pose
            print(f"Đã cập nhật dim_pose theo dữ liệu thật: {dim_pose}")
        else:
            dim_pose = 264                                
        inferred_joints = infer_beat_joints_num(dim_pose, getattr(opt, "motion_rep", "axis_angle"))
        if inferred_joints in (75, 88):
            opt.joints_num = inferred_joints
            print(f"[INFO] BEAT joints_num inferred from dim_pose={dim_pose}, motion_rep={opt.motion_rep}: {opt.joints_num}")
        else:
            opt.joints_num = 88
            print(f"[WARN] Could not infer BEAT joints_num from dim_pose={dim_pose}, fallback to 88")

        if opt.joints_num == 75:
            kinematic_chain = paramUtil.beat75_kinematic_chain
        else:
            kinematic_chain = paramUtil.beat_kinematic_chain
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
    train_split = pjoin(opt.data_root, 'train.txt')
    test_split = pjoin(opt.data_root, 'test.txt')
    
    # Auto-create train.txt + test.txt if not exists (for BEAT dataset)
    if opt.dataset_name == 'beat':
        create_train_test_splits(
            opt.motion_dir,
            train_split,
            test_split,
            test_ratio=opt.test_ratio,
            split_seed=opt.split_seed
        )

    train_set   = DatasetClass(opt, mean, std, train_split, opt.times)
    bad_ids = validate_dataset(train_set)
    trainer.train(train_set)
    print("==> Training complete! Congratulations!")
