import argparse
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataset import Beat2MotionDataset
from datasets.evaluator_sbert import SBERTMotionTextModel
from utils.get_opt import get_opt


def collate_caption_motion(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        raise ValueError("All samples are invalid in current batch")
    batch.sort(key=lambda x: x[2], reverse=True)
    captions = [x[0] for x in batch]
    motions = torch.from_numpy(np.stack([x[1] for x in batch], axis=0)).float()
    m_lens = torch.LongTensor([int(x[2]) for x in batch])
    return captions, motions, m_lens


def retrieval_loss(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_m = F.cross_entropy(logits.t(), labels)
    loss = 0.5 * (loss_t + loss_m)
    with torch.no_grad():
        acc_t = (logits.argmax(dim=1) == labels).float().mean()
        acc_m = (logits.argmax(dim=0) == labels).float().mean()
        acc = 0.5 * (acc_t + acc_m)
        temp = (1.0 / logits.new_tensor(logits.shape[0]))  # dummy init for graph safety
    return loss, acc


def run_epoch(loader, model, optimizer, device, train=True, grad_clip=1.0):
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    with torch.set_grad_enabled(train):
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for captions, motions, m_lens in pbar:
            motions = motions.to(device)
            m_lens = m_lens.to(device)
            text_emb = model.encode_text(captions, device)
            motion_emb = model.encode_motion(motions, m_lens)
            logits = model.similarity_logits(text_emb, motion_emb)
            labels = torch.arange(logits.shape[0], device=logits.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
            with torch.no_grad():
                acc = 0.5 * (
                    (logits.argmax(dim=1) == labels).float().mean()
                    + (logits.argmax(dim=0) == labels).float().mean()
                )
                temp = float((1.0 / model.logit_scale.exp().clamp(max=100.0)).item())

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = motions.shape[0]
            total_loss += float(loss.item()) * bs
            total_acc += float(acc.item()) * bs
            total_count += bs
            pbar.set_postfix(loss=float(loss.item()), acc=float(acc.item()), temp=temp)

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_acc / total_count


def save_checkpoint(path, model, optimizer, epoch, best_val, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "movement_encoder": model.movement_encoder.state_dict(),
            "motion_encoder": model.motion_encoder.state_dict(),
            "text_proj": model.text_proj.state_dict(),
            "logit_scale": model.logit_scale.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_opt", type=str, required=True)
    parser.add_argument("--name", type=str, default="text_mot_match_sbert")
    parser.add_argument("--text_model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early_stop_metric", type=str, default="val_acc", choices=["val_acc", "val_loss"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--train_split", type=str, default="")
    parser.add_argument("--val_split", type=str, default="")
    parser.add_argument("--text_dir", type=str, default="")
    parser.add_argument("--finetune_text_encoder", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    opt = get_opt(args.source_opt, device)
    if opt.dataset_name != "beat":
        raise ValueError(f"Only beat is supported, got {opt.dataset_name}")
    opt.data_root = pjoin(ROOT, "datasets", "BEAT_numpy")
    opt.motion_dir = pjoin(opt.data_root, "npy")
    opt.text_dir = args.text_dir if args.text_dir else pjoin(opt.data_root, "txt")
    opt.max_motion_length = getattr(opt, "max_motion_length", 360)
    opt.motion_rep = getattr(opt, "motion_rep", "axis_angle")
    opt.unit_length = getattr(opt, "unit_length", 4)
    opt.joints_num = 88
    opt.dim_pose = 264
    opt.max_text_len = 20
    opt.times = 1
    opt.is_train = False

    train_split = args.train_split if args.train_split else pjoin(opt.data_root, "train.txt")
    val_split = args.val_split if args.val_split else pjoin(opt.data_root, "val.txt")
    if not os.path.exists(val_split):
        val_split = pjoin(opt.data_root, "test.txt")
    print(f"[INFO] train_split={train_split}")
    print(f"[INFO] val_split={val_split}")
    print(f"[INFO] text_dir={opt.text_dir}")

    mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
    std = np.load(pjoin(opt.meta_dir, "std.npy"))
    train_set = Beat2MotionDataset(opt, mean, std, train_split, times=1, w_vectorizer=None, eval_mode=False)
    val_set = Beat2MotionDataset(opt, mean, std, val_split, times=1, w_vectorizer=None, eval_mode=False)
    print(f"[INFO] train samples: {len(train_set)}")
    print(f"[INFO] val samples: {len(val_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        collate_fn=collate_caption_motion,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=collate_caption_motion,
    )

    model = SBERTMotionTextModel(
        dim_pose=opt.dim_pose,
        unit_length=opt.unit_length,
        text_model_name=args.text_model_name,
        train_text_encoder=args.finetune_text_encoder,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, int(round(args.epochs * args.warmup_ratio)))
    warmup_epochs = min(warmup_epochs, max(1, args.epochs - 1))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.lr * args.min_lr_ratio),
        ],
        milestones=[warmup_epochs],
    )

    save_root = pjoin(ROOT, "checkpoints", "beat", args.name)
    model_dir = pjoin(save_root, "model")
    os.makedirs(model_dir, exist_ok=True)
    best_path = pjoin(model_dir, "finest.tar")
    best_acc_path = pjoin(model_dir, "best_by_val_acc.tar")

    cfg = {
        "dim_pose": opt.dim_pose,
        "unit_length": opt.unit_length,
        "text_model_name": args.text_model_name,
    }
    best_val = float("inf")
    best_val_acc = -1.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, model, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(val_loader, model, optimizer, device, train=False)
        lr = optimizer.param_groups[0]["lr"]
        temp = float((1.0 / model.logit_scale.exp().clamp(max=100.0)).item())
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} temp={temp:.4f} lr={lr:.8f}"
        )

        if epoch % args.save_every == 0:
            save_checkpoint(pjoin(model_dir, f"ckpt_e{epoch:03d}.tar"), model, optimizer, epoch, best_val, cfg)

        prev_best_val = best_val
        prev_best_val_acc = best_val_acc

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, best_val, cfg)
            print(f"[INFO] New best checkpoint: {best_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_acc_path, model, optimizer, epoch, best_val, cfg)
            print(f"[INFO] New best-by-acc checkpoint: {best_acc_path}")

        improved = (val_acc > prev_best_val_acc) if args.early_stop_metric == "val_acc" else (val_loss < prev_best_val)
        if improved:
            no_improve = 0
        else:
            no_improve += 1

        scheduler.step()
        if no_improve >= args.patience:
            print(
                f"[INFO] Early stopping at epoch {epoch} "
                f"(metric={args.early_stop_metric}, patience={args.patience})"
            )
            break

    print(f"[INFO] Training finished. Best val loss: {best_val:.6f}")
    print(f"[INFO] Best val acc: {best_val_acc:.4f}")
    print(f"[INFO] Finest checkpoint: {best_path}")
    print(f"[INFO] Best-by-acc checkpoint: {best_acc_path}")


if __name__ == "__main__":
    main()
    
# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -u tools/train_evaluator_sbert.py --source_opt checkpoints/beat/beat_geometry/opt.txt --name text_mot_match_sbert --batch_size 32 --workers 0 --epochs 50 
# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -u tools/evaluation.py checkpoints/beat/beat_baseline/opt.txt
# PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python3 -u tools/evaluation.py checkpoints/beat/beat_geometry/opt.txt
