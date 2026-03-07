import argparse
import glob
import os
import sys
from os.path import join as pjoin

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import MotionTransformer
from trainers import DDPMTrainer
from utils import paramUtil
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.utils import motion_temporal_filter


def infer_dim_pose(motion_dir, default_dim):
    try:
        files = glob.glob(os.path.join(motion_dir, "**/*.npy"), recursive=True)
        if not files:
            return default_dim
        arr = np.load(files[0])
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim == 2:
            return int(arr.shape[1])
    except Exception:
        return default_dim
    return default_dim


def setup_dataset(opt):
    if opt.dataset_name == "t2m":
        opt.data_root = pjoin(ROOT, "datasets", "HumanML3D", "HumanML3D")
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 22
        opt.max_motion_length = 196
        opt.dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == "kit":
        opt.data_root = pjoin(ROOT, "datasets", "KIT-ML")
        opt.motion_dir = pjoin(opt.data_root, "new_joint_vecs")
        opt.joints_num = 21
        opt.max_motion_length = 196
        opt.dim_pose = 251
        kinematic_chain = paramUtil.kit_kinematic_chain
    elif opt.dataset_name == "beat":
        opt.data_root = pjoin(ROOT, "datasets", "BEAT_numpy")
        opt.motion_dir = pjoin(opt.data_root, "npy")
        opt.joints_num = 88
        opt.max_motion_length = 360
        opt.dim_pose = infer_dim_pose(opt.motion_dir, 264)
        kinematic_chain = paramUtil.beat_kinematic_chain
    else:
        raise KeyError(f"Unknown dataset {opt.dataset_name}")
    return kinematic_chain


def build_model(opt):
    return MotionTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff,
    )


def resolve_checkpoint(opt, ckpt_path):
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path
    latest = pjoin(opt.model_dir, "latest.tar")
    if os.path.isfile(latest):
        return latest
    all_ckpt = sorted(glob.glob(pjoin(opt.model_dir, "ckpt_e*.tar")))
    if not all_ckpt:
        raise FileNotFoundError(f"No checkpoint found in {opt.model_dir}")
    return all_ckpt[-1]


def to_joint_positions(motion, dataset_name, joints_num):
    if dataset_name == "beat":
        return motion.reshape(motion.shape[0], joints_num, 3)
    return recover_from_ric(torch.from_numpy(motion).float(), joints_num).cpu().numpy()


def chain_to_edges(chain_list):
    edges = []
    for chain in chain_list:
        for i in range(len(chain) - 1):
            edges.append((chain[i], chain[i + 1]))
    return edges


def beat_edges():
    return list(paramUtil.beat_bone_pairs)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def render_frame_3d(joints, edges, out_file, title=""):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], s=8, c="black")
    for s, e in edges:
        ax.plot(
            [joints[s, 0], joints[e, 0]],
            [joints[s, 1], joints[e, 1]],
            [joints[s, 2], joints[e, 2]],
            linewidth=1.2,
            color="royalblue",
        )
    min_xyz = joints.min(axis=0)
    max_xyz = joints.max(axis=0)
    center = (min_xyz + max_xyz) / 2.0
    radius = max(float((max_xyz - min_xyz).max() / 2.0), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=20, azim=-75)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def render_frame_2d(joints, edges, out_file, title=""):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    x = joints[:, 0]
    z = joints[:, 2]
    ax.scatter(x, z, s=10, c="black")
    for s, e in edges:
        ax.plot([x[s], x[e]], [z[s], z[e]], linewidth=1.2, color="tomato")
    xmin, xmax = float(x.min()), float(x.max())
    zmin, zmax = float(z.min()), float(z.max())
    cx, cz = (xmin + xmax) / 2.0, (zmin + zmax) / 2.0
    r = max((xmax - xmin) / 2.0, (zmax - zmin) / 2.0, 1e-3)
    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cz - r, cz + r)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_file, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser("Inference + export 2D/3D frames")
    parser.add_argument("--opt_path", type=str, required=True, help="Path to opt.txt")
    parser.add_argument("--text", type=str, required=True, help="Prompt text")
    parser.add_argument("--motion_length", type=int, default=120, help="Number of frames to export")
    parser.add_argument("--out_dir", type=str, default=pjoin(ROOT, "results", "infer_frames"))
    parser.add_argument("--checkpoint", type=str, default="", help="Specific checkpoint .tar")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id, -1 for CPU")
    parser.add_argument("--smooth_sigma", type=float, default=1.0, help="Temporal smoothing sigma")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id >= 0 and torch.cuda.is_available() else "cpu")
    opt = get_opt(args.opt_path, device)
    opt.device = device
    opt.is_train = False

    kinematic_chain = setup_dataset(opt)
    if opt.dataset_name == "beat":
        edges = beat_edges()
    else:
        edges = chain_to_edges(kinematic_chain)

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))

    model = build_model(opt).to(device)
    trainer = DDPMTrainer(opt, model)

    ckpt = resolve_checkpoint(opt, args.checkpoint)
    print(f"[INFO] Loading checkpoint: {ckpt}")
    trainer.load(ckpt)
    trainer.eval_mode()
    trainer.to(device)

    max_len = min(max(1, int(args.motion_length)), int(opt.max_motion_length))
    captions = [args.text]
    m_lens = torch.LongTensor([max_len]).to(device)

    with torch.no_grad():
        pred = trainer.generate(captions, m_lens, opt.dim_pose)[0].cpu().numpy()
    pred = pred[:max_len]
    pred = pred * std + mean
    joints = to_joint_positions(pred, opt.dataset_name, opt.joints_num)
    joints = motion_temporal_filter(joints, sigma=args.smooth_sigma)

    frames_2d_dir = pjoin(args.out_dir, "frames_2d")
    frames_3d_dir = pjoin(args.out_dir, "frames_3d")
    ensure_dir(frames_2d_dir)
    ensure_dir(frames_3d_dir)

    total = joints.shape[0]
    for i in range(total):
        frame = joints[i]
        name = f"frame_{i:04d}.png"
        render_frame_2d(frame, edges, pjoin(frames_2d_dir, name), title=f"2D frame {i}")
        render_frame_3d(frame, edges, pjoin(frames_3d_dir, name), title=f"3D frame {i}")
        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"[INFO] Exported {i + 1}/{total} frames")

    npy_out = pjoin(args.out_dir, "generated_joints.npy")
    np.save(npy_out, joints)
    print(f"[DONE] Saved joints: {npy_out}")
    print(f"[DONE] 2D frames: {frames_2d_dir}")
    print(f"[DONE] 3D frames: {frames_3d_dir}")


if __name__ == "__main__":
    main()
