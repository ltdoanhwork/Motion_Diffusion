"""
Evaluate latent autoencoder quality in one report:
1) Reconstruction / velocity metrics
2) Latent statistics
3) Code usage statistics
4) Reconstruction samples (saved as .npy)
"""
import argparse
import json
import os
import sys
from os.path import join as pjoin
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PYMO_DIR = os.path.join(ROOT, "datasets", "pymo")
if PYMO_DIR not in sys.path:
    sys.path.insert(0, PYMO_DIR)

from datasets import Beat2MotionDataset
from datasets.dataset import Text2MotionDataset
from models.vq.model import RVQVAE


class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = None
        self.max_val = None

    def update(self, arr):
        x = arr.reshape(-1).astype(np.float64)
        if x.size == 0:
            return
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        self.min_val = x_min if self.min_val is None else min(self.min_val, x_min)
        self.max_val = x_max if self.max_val is None else max(self.max_val, x_max)
        for v in x:
            self.count += 1
            delta = v - self.mean
            self.mean += delta / self.count
            delta2 = v - self.mean
            self.m2 += delta * delta2

    def summary(self):
        if self.count <= 1:
            std = 0.0
        else:
            std = float(np.sqrt(self.m2 / (self.count - 1)))
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std": std,
            "min": float(self.min_val if self.min_val is not None else 0.0),
            "max": float(self.max_val if self.max_val is not None else 0.0),
        }


class DummyOpt:
    def __init__(self, args):
        self.motion_dir = args.motion_dir
        self.text_dir = args.text_dir
        self.max_motion_length = args.max_motion_length
        self.dataset_name = args.dataset_name
        self.motion_rep = args.motion_rep
        self.is_train = False
        self.joints_num = args.joints_num
        self.max_text_len = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate latent VQ model quality")
    parser.add_argument("--dataset_name", type=str, default="beat", choices=["beat", "t2m", "kit"])
    parser.add_argument("--data_root", type=str, default="./datasets/BEAT_numpy")
    parser.add_argument("--motion_rep", type=str, default="position", choices=["position", "axis_angle"])
    parser.add_argument("--max_motion_length", type=int, default=360)
    parser.add_argument("--joints_num", type=int, default=88)
    parser.add_argument("--dim_pose", type=int, default=264)

    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--vq_name", type=str, required=True, help="Experiment name under checkpoints/<dataset>/<vq_name>")
    parser.add_argument("--vq_checkpoint", type=str, default="", help="Optional explicit checkpoint path")
    parser.add_argument("--split_file", type=str, default="", help="Optional explicit split file path")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means evaluate all")
    parser.add_argument("--num_sample_saves", type=int, default=8)

    # RVQVAE config
    parser.add_argument("--codebook_size", type=int, default=1024)
    parser.add_argument("--codebook_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--down_t", type=int, default=3)
    parser.add_argument("--stride_t", type=int, default=2)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dilation_growth_rate", type=int, default=3)
    parser.add_argument("--num_quantizers", type=int, default=10)
    parser.add_argument("--shared_codebook", action="store_true")
    parser.add_argument("--quantize_dropout_prob", type=float, default=0.0)
    parser.add_argument("--latent_mode", type=str, default="kl", choices=["vq", "kl"])
    parser.add_argument("--use_posterior_sample", action="store_true", default=False)

    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


def resolve_dataset(args):
    if args.dataset_name == "beat":
        args.motion_dir = pjoin(args.data_root, "npy")
        args.text_dir = pjoin(args.data_root, "txt")
        dataset_cls = Beat2MotionDataset
    elif args.dataset_name == "t2m":
        args.motion_dir = pjoin(args.data_root, "new_joint_vecs")
        args.text_dir = pjoin(args.data_root, "texts")
        dataset_cls = Text2MotionDataset
    else:
        args.motion_dir = pjoin(args.data_root, "new_joint_vecs")
        args.text_dir = pjoin(args.data_root, "texts")
        dataset_cls = Text2MotionDataset
    return dataset_cls


def load_stats(args):
    save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.vq_name)
    mean_path = pjoin(save_root, "meta", "mean.npy")
    std_path = pjoin(save_root, "meta", "std.npy")
    if os.path.exists(mean_path) and os.path.exists(std_path):
        return np.load(mean_path), np.load(std_path)

    pipeline_path = pjoin(ROOT, "global_pipeline.pkl")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Missing normalization stats at {mean_path}/{std_path} and {pipeline_path}"
        )
    pipeline = joblib.load(pipeline_path)
    scaler = pipeline.named_steps["stdscale"]
    return scaler.data_mean_, scaler.data_std_


def resolve_checkpoint(args):
    if args.vq_checkpoint:
        if not os.path.exists(args.vq_checkpoint):
            raise FileNotFoundError(args.vq_checkpoint)
        return args.vq_checkpoint

    model_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.vq_name, "model")
    candidates = [
        pjoin(model_dir, "best_model.tar"),
        pjoin(model_dir, "latest.tar"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def build_model(args, device):
    qargs = SimpleNamespace(
        num_quantizers=args.num_quantizers,
        shared_codebook=args.shared_codebook,
        quantize_dropout_prob=args.quantize_dropout_prob,
        mu=0.99,
    )
    model = RVQVAE(
        qargs,
        input_width=args.dim_pose,
        nb_code=args.codebook_size,
        code_dim=args.codebook_dim,
        output_emb_width=args.codebook_dim,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation="relu",
        norm=None,
        embed_dim=args.embed_dim,
        double_z=(args.latent_mode == "kl"),
    )
    ckpt = torch.load(resolve_checkpoint(args), map_location="cpu")
    if isinstance(ckpt, dict) and "vq_model" in ckpt:
        state = ckpt["vq_model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def get_split_file(args):
    if args.split_file:
        return args.split_file
    val_file = pjoin(args.data_root, "val.txt")
    if os.path.exists(val_file):
        return val_file
    train_file = pjoin(args.data_root, "train.txt")
    if os.path.exists(train_file):
        return train_file
    raise FileNotFoundError("Cannot find split file. Pass --split_file explicitly.")


def update_code_usage(code_idx, code_hist):
    if code_idx is None:
        return
    # Expected shape: (B, T, Q)
    if code_idx.ndim == 2:
        code_idx = code_idx.unsqueeze(-1)
    if code_idx.ndim != 3:
        return
    num_quantizers = code_idx.shape[-1]
    for q in range(num_quantizers):
        q_idx = code_idx[..., q].reshape(-1)
        q_idx = q_idx[q_idx >= 0]
        if q_idx.numel() == 0:
            continue
        counts = torch.bincount(q_idx, minlength=code_hist[q].shape[0]).cpu().numpy()
        code_hist[q] += counts


def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dataset_cls = resolve_dataset(args)
    mean, std = load_stats(args)
    split_file = get_split_file(args)

    dataset = dataset_cls(DummyOpt(args), mean, std, split_file, times=1)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    model = build_model(args, device)

    save_root = pjoin(args.checkpoints_dir, args.dataset_name, args.vq_name)
    out_dir = pjoin(save_root, "latent_eval")
    sample_dir = pjoin(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    rec_mse_list = []
    rec_mae_list = []
    vel_mse_list = []
    vel_mae_list = []
    latent_stats = RunningStats()

    code_hist = np.zeros((args.num_quantizers, args.codebook_size), dtype=np.int64)
    saved = 0
    processed_batches = 0

    with torch.no_grad():
        for batch in loader:
            captions, motions, m_lens = batch
            motions = motions.to(device).float()

            if args.latent_mode == "kl":
                posterior = model.encode(motions)
                z = posterior.sample() if args.use_posterior_sample else posterior.mode()
                # z shape: (B, C, T)
                x_q, _, q_info = model.quantize(z)
                code_idx = q_info[2] if isinstance(q_info, tuple) and len(q_info) >= 3 else None
            else:
                code_idx, all_codes = model.encode(motions)
                z = torch.sum(all_codes, dim=0)  # (B, C, T)

            recon = model.decode(z)

            # Metrics
            rec_diff = recon - motions
            rec_mse_list.append(float(torch.mean(rec_diff ** 2).item()))
            rec_mae_list.append(float(torch.mean(torch.abs(rec_diff)).item()))

            vel_gt = motions[:, 1:] - motions[:, :-1]
            vel_pred = recon[:, 1:] - recon[:, :-1]
            vel_diff = vel_pred - vel_gt
            vel_mse_list.append(float(torch.mean(vel_diff ** 2).item()))
            vel_mae_list.append(float(torch.mean(torch.abs(vel_diff)).item()))

            # Latent stats in (B, T, C)
            latent_bt_c = z.permute(0, 2, 1).contiguous().cpu().numpy()
            latent_stats.update(latent_bt_c)

            # Code usage
            if code_idx is not None:
                update_code_usage(code_idx.detach().cpu(), code_hist)

            # Save reconstruction samples
            if saved < args.num_sample_saves:
                b = motions.shape[0]
                n_save = min(args.num_sample_saves - saved, b)
                for i in range(n_save):
                    np.save(
                        pjoin(sample_dir, f"sample_{saved:03d}_gt.npy"),
                        motions[i].detach().cpu().numpy(),
                    )
                    np.save(
                        pjoin(sample_dir, f"sample_{saved:03d}_recon.npy"),
                        recon[i].detach().cpu().numpy(),
                    )
                    saved += 1

            processed_batches += 1
            if args.max_batches > 0 and processed_batches >= args.max_batches:
                break

    code_usage = []
    for q in range(code_hist.shape[0]):
        used = int(np.sum(code_hist[q] > 0))
        total = int(code_hist[q].sum())
        code_usage.append(
            {
                "quantizer": q,
                "used_codes": used,
                "usage_ratio": float(used / max(1, args.codebook_size)),
                "total_assignments": total,
            }
        )

    report = {
        "config": {
            "dataset_name": args.dataset_name,
            "vq_name": args.vq_name,
            "split_file": split_file,
            "latent_mode": args.latent_mode,
            "use_posterior_sample": bool(args.use_posterior_sample),
            "processed_batches": processed_batches,
        },
        "reconstruction_metrics": {
            "rec_mse_mean": float(np.mean(rec_mse_list)) if rec_mse_list else 0.0,
            "rec_mae_mean": float(np.mean(rec_mae_list)) if rec_mae_list else 0.0,
            "vel_mse_mean": float(np.mean(vel_mse_list)) if vel_mse_list else 0.0,
            "vel_mae_mean": float(np.mean(vel_mae_list)) if vel_mae_list else 0.0,
        },
        "latent_stats": latent_stats.summary(),
        "code_usage": code_usage,
        "sample_dir": sample_dir,
        "num_saved_samples": saved,
    }

    json_path = pjoin(out_dir, "latent_eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = pjoin(out_dir, "latent_eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Latent Evaluation Report\n\n")
        f.write("## Reconstruction / Velocity\n")
        f.write(f"- rec_mse_mean: {report['reconstruction_metrics']['rec_mse_mean']:.6f}\n")
        f.write(f"- rec_mae_mean: {report['reconstruction_metrics']['rec_mae_mean']:.6f}\n")
        f.write(f"- vel_mse_mean: {report['reconstruction_metrics']['vel_mse_mean']:.6f}\n")
        f.write(f"- vel_mae_mean: {report['reconstruction_metrics']['vel_mae_mean']:.6f}\n\n")
        f.write("## Latent Stats\n")
        latent = report["latent_stats"]
        f.write(f"- mean: {latent['mean']:.6f}\n")
        f.write(f"- std: {latent['std']:.6f}\n")
        f.write(f"- min: {latent['min']:.6f}\n")
        f.write(f"- max: {latent['max']:.6f}\n\n")
        f.write("## Code Usage\n")
        for item in report["code_usage"]:
            f.write(
                f"- q{item['quantizer']}: used {item['used_codes']}/{args.codebook_size} "
                f"({item['usage_ratio']:.4f}), assignments={item['total_assignments']}\n"
            )
        f.write("\n## Reconstruction Samples\n")
        f.write(f"- directory: `{sample_dir}`\n")
        f.write(f"- saved pairs: {saved}\n")

    print("=" * 70)
    print("LATENT EVALUATION COMPLETE")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Samples: {sample_dir}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate(parse_args())
