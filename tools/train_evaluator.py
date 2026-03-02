import argparse
import glob
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.dataset import Beat2MotionDataset
from datasets.evaluator import _get_word_vectorizer, collate_fn
from datasets.evaluator_models import MovementConvEncoder, MotionEncoderBiGRUCo, TextEncoderBiGRUCo
from utils.get_opt import get_opt
from utils.word_vectorizer import POS_enumerator


def infer_dim_pose(motion_dir, default_dim=264):
    files = glob.glob(os.path.join(motion_dir, "**/*.npy"), recursive=True)
    if not files:
        return default_dim
    arr = np.load(files[0])
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim == 2:
        return int(arr.shape[1])
    return default_dim


def build_evaluator_models(device, dim_pose):
    movement_enc = MovementConvEncoder(dim_pose - 4, 512, 512).to(device)
    text_enc = TextEncoderBiGRUCo(
        word_size=300,
        pos_size=len(POS_enumerator),
        hidden_size=512,
        output_size=512,
        device=device,
    ).to(device)
    motion_enc = MotionEncoderBiGRUCo(
        input_size=512,
        hidden_size=1024,
        output_size=512,
        device=device,
    ).to(device)
    return movement_enc, text_enc, motion_enc


def retrieval_loss(text_embed, motion_embed, temperature):
    text_embed = F.normalize(text_embed, dim=-1)
    motion_embed = F.normalize(motion_embed, dim=-1)
    logits = torch.matmul(text_embed, motion_embed.transpose(0, 1)) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_t = F.cross_entropy(logits, labels)
    loss_m = F.cross_entropy(logits.transpose(0, 1), labels)
    loss = 0.5 * (loss_t + loss_m)
    with torch.no_grad():
        acc_t = (logits.argmax(dim=1) == labels).float().mean()
        acc_m = (logits.argmax(dim=0) == labels).float().mean()
        acc = 0.5 * (acc_t + acc_m)
    return loss, acc


def forward_embeddings(batch, models, unit_length, device):
    movement_enc, text_enc, motion_enc = models
    if len(batch) < 6:
        raise ValueError(f"Unexpected batch format with {len(batch)} fields")
    word_embs, pos_ohot, _, cap_lens, motions, m_lens = batch[:6]
    word_embs = word_embs.float().to(device)
    pos_ohot = pos_ohot.float().to(device)
    motions = motions.float().to(device)
    cap_lens = cap_lens.to(device).long()
    m_lens = m_lens.to(device).long()
    # Some dataset entries may keep original clip lengths after temporal crop/pad.
    # Clamp to the actual sequence length to avoid invalid RNN packed lengths.
    m_lens = torch.clamp(m_lens, min=1, max=motions.shape[1])

    align_idx = torch.argsort(m_lens, descending=True)
    motions = motions[align_idx]
    motion_lens = torch.clamp(m_lens[align_idx] // unit_length, min=1)

    text_embed = text_enc(word_embs, pos_ohot, cap_lens)
    text_embed = text_embed[align_idx]

    movements = movement_enc(motions[..., :-4])
    motion_lens = torch.clamp(motion_lens, max=movements.shape[1])
    motion_embed = motion_enc(movements, motion_lens)
    return text_embed, motion_embed


def run_epoch(loader, models, optimizer, unit_length, device, temperature, train=True, grad_clip=1.0):
    movement_enc, text_enc, motion_enc = models
    modules = [movement_enc, text_enc, motion_enc]
    for module in modules:
        module.train(train)

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    with torch.set_grad_enabled(train):
        iterator = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in iterator:
            bs = batch[0].shape[0]
            text_embed, motion_embed = forward_embeddings(batch, models, unit_length, device)
            loss, acc = retrieval_loss(text_embed, motion_embed, temperature)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    params = []
                    for module in modules:
                        params.extend(list(module.parameters()))
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                optimizer.step()

            total_loss += loss.item() * bs
            total_acc += acc.item() * bs
            total_count += bs
            iterator.set_postfix(loss=loss.item(), acc=acc.item())

    if total_count == 0:
        return 0.0, 0.0
    return total_loss / total_count, total_acc / total_count


def save_checkpoint(path, models, optimizer, epoch, best_val):
    movement_enc, text_enc, motion_enc = models
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_val": best_val,
            "movement_encoder": movement_enc.state_dict(),
            "text_encoder": text_enc.state_dict(),
            "motion_encoder": motion_enc.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source_opt", type=str, required=True, help="Path to an existing BEAT opt.txt")
    parser.add_argument("--name", type=str, default="text_mot_match", help="Checkpoint folder name under checkpoints/beat/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    source_opt = get_opt(args.source_opt, device)
    if source_opt.dataset_name != "beat":
        raise ValueError(f"source_opt dataset_name must be beat, got {source_opt.dataset_name}")

    source_opt.data_root = pjoin(ROOT, "datasets", "BEAT_numpy")
    source_opt.motion_dir = pjoin(source_opt.data_root, "npy")
    source_opt.text_dir = pjoin(source_opt.data_root, "txt")
    source_opt.max_motion_length = getattr(source_opt, "max_motion_length", 360)
    source_opt.max_text_len = getattr(source_opt, "max_text_len", 20)
    source_opt.motion_rep = getattr(source_opt, "motion_rep", "axis_angle")
    source_opt.unit_length = getattr(source_opt, "unit_length", 4)
    source_opt.times = 1
    source_opt.joints_num = 88
    source_opt.dim_pose = infer_dim_pose(source_opt.motion_dir, 264)

    mean_path = pjoin(source_opt.meta_dir, "mean.npy")
    std_path = pjoin(source_opt.meta_dir, "std.npy")
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing mean/std in {source_opt.meta_dir}")
    mean = np.load(mean_path)
    std = np.load(std_path)

    train_split = pjoin(source_opt.data_root, "train.txt")
    val_split = pjoin(source_opt.data_root, "val.txt")
    if not os.path.exists(val_split):
        val_split = pjoin(source_opt.data_root, "test.txt")

    w_vectorizer = _get_word_vectorizer()
    source_opt.is_train = False
    train_set = Beat2MotionDataset(source_opt, mean, std, train_split, times=1, w_vectorizer=w_vectorizer, eval_mode=True)
    val_set = Beat2MotionDataset(source_opt, mean, std, val_split, times=1, w_vectorizer=w_vectorizer, eval_mode=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    print(f"[INFO] train samples: {len(train_set)}")
    print(f"[INFO] val samples: {len(val_set)}")

    models = build_evaluator_models(device, source_opt.dim_pose)
    optimizer = torch.optim.AdamW(
        list(models[0].parameters()) + list(models[1].parameters()) + list(models[2].parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_root = pjoin(ROOT, "checkpoints", "beat", args.name)
    model_dir = pjoin(save_root, "model")
    os.makedirs(model_dir, exist_ok=True)
    best_path = pjoin(model_dir, "finest.tar")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            train_loader, models, optimizer, source_opt.unit_length, device, args.temperature, train=True
        )
        val_loss, val_acc = run_epoch(
            val_loader, models, optimizer, source_opt.unit_length, device, args.temperature, train=False
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f}"
        )

        if epoch % args.save_every == 0:
            save_checkpoint(pjoin(model_dir, f"ckpt_e{epoch:03d}.tar"), models, optimizer, epoch, best_val)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, models, optimizer, epoch, best_val)
            print(f"[INFO] New best checkpoint: {best_path}")

    print(f"[INFO] Training finished. Best val loss: {best_val:.6f}")
    print(f"[INFO] Finest checkpoint: {best_path}")


if __name__ == "__main__":
    main()
