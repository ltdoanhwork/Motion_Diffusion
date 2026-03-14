import argparse
import glob
import os
import re
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


def normalize_caption_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_pos_tag(token):
    if not token:
        return "OTHER"
    if token.isdigit():
        return "NUM"
    if token in {"a", "an", "the"}:
        return "DET"
    if token in {"in", "on", "at", "to", "for", "with", "from", "of", "by", "into", "over", "under", "across"}:
        return "ADP"
    if token in {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"}:
        return "PRON"
    if token in {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had"}:
        return "AUX"
    if token.endswith("ly"):
        return "ADV"
    if token.endswith(("ing", "ed", "en", "ify", "ise", "ize")):
        return "VERB"
    if token.endswith(("ous", "ful", "ive", "al", "ic", "able", "ible", "ish", "less")):
        return "ADJ"
    return "NOUN"


def caption_to_word_pos_tokens(caption, max_tokens):
    norm = normalize_caption_text(caption)
    words = [w for w in norm.split(" ") if w]
    if not words:
        words = ["unk"]
    words = words[:max_tokens]
    tokens = []
    for w in words:
        pos = simple_pos_tag(w)
        if pos not in POS_enumerator:
            pos = "OTHER"
        tokens.append(f"{w}/{pos}")
    return norm, tokens


def preprocess_dataset_text(dataset, split_name):
    total_caps = 0
    for item in dataset.data_dict.values():
        new_text = []
        for t in item.get("text", []):
            cap_raw = t.get("caption", "")
            cap_norm, tokens_wp = caption_to_word_pos_tokens(cap_raw, dataset.opt.max_text_len)
            t_new = dict(t)
            t_new["caption"] = cap_norm
            t_new["tokens"] = tokens_wp
            new_text.append(t_new)
            total_caps += 1
        item["text"] = new_text
    print(f"[INFO] {split_name}: normalized+POS tagged {total_caps} captions")


def retrieval_loss(
    text_embed,
    motion_embed,
    temperature,
    logit_scale_param=None,
    hard_neg_weight=0.0,
    hard_neg_margin=0.2,
):
    text_embed = F.normalize(text_embed, dim=-1)
    motion_embed = F.normalize(motion_embed, dim=-1)
    if logit_scale_param is not None:
        scale = logit_scale_param.exp().clamp(max=100.0)
    else:
        scale = 1.0 / max(temperature, 1e-6)
    logits = torch.matmul(text_embed, motion_embed.transpose(0, 1)) * scale
    labels = torch.arange(logits.shape[0], device=logits.device)
    ce_t = F.cross_entropy(logits, labels)
    ce_m = F.cross_entropy(logits.transpose(0, 1), labels)
    ce_loss = 0.5 * (ce_t + ce_m)

    hard_neg_loss = torch.tensor(0.0, device=logits.device)
    if hard_neg_weight > 0:
        pos = logits.diag()
        neg_rows = logits.masked_fill(torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool), float("-inf"))
        hard_row = neg_rows.max(dim=1).values
        hard_col = neg_rows.max(dim=0).values
        hard_neg_loss = 0.5 * (
            F.relu(hard_row - pos + hard_neg_margin).mean()
            + F.relu(hard_col - pos + hard_neg_margin).mean()
        )
    loss = ce_loss + hard_neg_weight * hard_neg_loss

    with torch.no_grad():
        acc_t = (logits.argmax(dim=1) == labels).float().mean()
        acc_m = (logits.argmax(dim=0) == labels).float().mean()
        acc = 0.5 * (acc_t + acc_m)
        effective_temp = (1.0 / scale).item()
    return loss, acc, ce_loss.detach(), hard_neg_loss.detach(), effective_temp


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


def run_epoch(
    loader,
    models,
    optimizer,
    unit_length,
    device,
    temperature,
    logit_scale_param=None,
    hard_neg_weight=0.0,
    hard_neg_margin=0.2,
    train=True,
    grad_clip=1.0,
    clip_params=None,
):
    movement_enc, text_enc, motion_enc = models
    modules = [movement_enc, text_enc, motion_enc]
    for module in modules:
        module.train(train)

    total_loss = 0.0
    total_acc = 0.0
    total_ce = 0.0
    total_hn = 0.0
    total_count = 0
    last_temp = temperature
    with torch.set_grad_enabled(train):
        iterator = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in iterator:
            bs = batch[0].shape[0]
            text_embed, motion_embed = forward_embeddings(batch, models, unit_length, device)
            loss, acc, ce_loss, hard_neg_loss, eff_temp = retrieval_loss(
                text_embed,
                motion_embed,
                temperature,
                logit_scale_param=logit_scale_param,
                hard_neg_weight=hard_neg_weight,
                hard_neg_margin=hard_neg_margin,
            )
            last_temp = eff_temp

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                optimizer.step()

            total_loss += loss.item() * bs
            total_acc += acc.item() * bs
            total_ce += ce_loss.item() * bs
            total_hn += hard_neg_loss.item() * bs
            total_count += bs
            iterator.set_postfix(loss=loss.item(), acc=acc.item(), temp=eff_temp)

    if total_count == 0:
        return 0.0, 0.0, 0.0, 0.0, last_temp
    return (
        total_loss / total_count,
        total_acc / total_count,
        total_ce / total_count,
        total_hn / total_count,
        last_temp,
    )


def save_checkpoint(path, models, optimizer, epoch, best_val, logit_scale_param=None):
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
            "logit_scale": (logit_scale_param.detach().cpu() if logit_scale_param is not None else None),
        },
        path,
    )


def collect_caption_token_stats(dataset):
    captions = []
    tokens = []
    total_tokens = 0
    pos_like_tokens = 0
    for item in dataset.data_dict.values():
        for t in item.get("text", []):
            cap = str(t.get("caption", "")).strip().lower()
            if cap:
                captions.append(cap)
            cur_tokens = t.get("tokens", [])
            if not isinstance(cur_tokens, list):
                cur_tokens = str(cur_tokens).split()
            for tok in cur_tokens:
                tok = str(tok).strip()
                if not tok:
                    continue
                tokens.append(tok.lower())
                total_tokens += 1
                if "/" in tok:
                    pos_like_tokens += 1
    return {
        "caption_set": set(captions),
        "token_set": set(tokens),
        "num_captions": len(captions),
        "num_tokens": len(tokens),
        "caption_unique": len(set(captions)),
        "token_unique": len(set(tokens)),
        "token_pos_ratio": (pos_like_tokens / total_tokens) if total_tokens > 0 else 0.0,
    }


def summarize_caption_stats(dataset, split_name):
    stats = collect_caption_token_stats(dataset)
    if stats["num_captions"] == 0:
        print(f"[WARN] {split_name}: no captions found")
        return stats
    print(
        f"[INFO] {split_name} caption stats: "
        f"total_pairs={stats['num_captions']}, "
        f"unique={stats['caption_unique']}, "
        f"unique_ratio={stats['caption_unique'] / stats['num_captions']:.4f}"
    )
    print(
        f"[INFO] {split_name} token stats: "
        f"total_tokens={stats['num_tokens']}, "
        f"unique_tokens={stats['token_unique']}, "
        f"pos_tag_format_ratio={stats['token_pos_ratio']:.4f}"
    )
    return stats


def summarize_split_overlap(train_stats, val_stats):
    val_caption_set = val_stats["caption_set"]
    val_token_set = val_stats["token_set"]
    caption_overlap = (
        len(train_stats["caption_set"] & val_caption_set) / len(val_caption_set)
        if val_caption_set
        else 0.0
    )
    token_overlap = (
        len(train_stats["token_set"] & val_token_set) / len(val_token_set)
        if val_token_set
        else 0.0
    )
    print(
        f"[INFO] train/val overlap: "
        f"caption_overlap_vs_val={caption_overlap:.4f}, "
        f"token_overlap_vs_val={token_overlap:.4f}"
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
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio over total epochs")
    parser.add_argument("--warmup_start_factor", type=float, default=0.1, help="Initial LR factor for warmup")
    parser.add_argument("--min_lr_ratio", type=float, default=0.01, help="Minimum LR ratio in cosine phase")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on val_loss")
    parser.add_argument("--fixed_temperature", action="store_true", help="Disable learnable temperature")
    parser.add_argument("--hard_neg_weight", type=float, default=0.2, help="Weight for hard negative hinge term")
    parser.add_argument("--hard_neg_margin", type=float, default=0.2, help="Margin for hard negative hinge term")
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
    source_opt.text_dir = args.text_dir if args.text_dir else pjoin(source_opt.data_root, "txt")
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

    train_split = args.train_split if args.train_split else pjoin(source_opt.data_root, "train.txt")
    val_split = args.val_split if args.val_split else pjoin(source_opt.data_root, "val.txt")
    if not os.path.exists(val_split):
        val_split = pjoin(source_opt.data_root, "test.txt")

    w_vectorizer = _get_word_vectorizer()
    source_opt.is_train = False
    train_set = Beat2MotionDataset(source_opt, mean, std, train_split, times=1, w_vectorizer=w_vectorizer, eval_mode=True)
    val_set = Beat2MotionDataset(source_opt, mean, std, val_split, times=1, w_vectorizer=w_vectorizer, eval_mode=True)
    preprocess_dataset_text(train_set, "train")
    preprocess_dataset_text(val_set, "val")
    train_caption_stats = summarize_caption_stats(train_set, "train")
    val_caption_stats = summarize_caption_stats(val_set, "val")
    summarize_split_overlap(train_caption_stats, val_caption_stats)

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
    optim_params = list(models[0].parameters()) + list(models[1].parameters()) + list(models[2].parameters())
    logit_scale_param = None
    if not args.fixed_temperature:
        logit_scale_param = torch.nn.Parameter(torch.log(torch.tensor(1.0 / args.temperature, device=device)))
        optim_params.append(logit_scale_param)
        print(f"[INFO] Learnable temperature enabled. Initial temp={args.temperature:.4f}")
    else:
        print(f"[INFO] Fixed temperature={args.temperature:.4f}")

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup_epochs = max(1, int(round(args.epochs * args.warmup_ratio)))
    warmup_epochs = min(warmup_epochs, max(1, args.epochs - 1))
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=args.lr * args.min_lr_ratio,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    save_root = pjoin(ROOT, "checkpoints", "beat", args.name)
    model_dir = pjoin(save_root, "model")
    os.makedirs(model_dir, exist_ok=True)
    best_path = pjoin(model_dir, "finest.tar")

    best_val = float("inf")
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_ce, train_hn, train_temp = run_epoch(
            train_loader,
            models,
            optimizer,
            source_opt.unit_length,
            device,
            args.temperature,
            logit_scale_param=logit_scale_param,
            hard_neg_weight=args.hard_neg_weight,
            hard_neg_margin=args.hard_neg_margin,
            train=True,
            clip_params=optim_params,
        )
        val_loss, val_acc, val_ce, val_hn, val_temp = run_epoch(
            val_loader,
            models,
            optimizer,
            source_opt.unit_length,
            device,
            args.temperature,
            logit_scale_param=logit_scale_param,
            hard_neg_weight=args.hard_neg_weight,
            hard_neg_margin=args.hard_neg_margin,
            train=False,
            clip_params=optim_params,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} train_ce={train_ce:.6f} train_hn={train_hn:.6f} train_acc={train_acc:.4f} temp={train_temp:.4f} "
            f"val_loss={val_loss:.6f} val_ce={val_ce:.6f} val_hn={val_hn:.6f} val_acc={val_acc:.4f} temp={val_temp:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f}"
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                pjoin(model_dir, f"ckpt_e{epoch:03d}.tar"),
                models,
                optimizer,
                epoch,
                best_val,
                logit_scale_param=logit_scale_param,
            )

        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            save_checkpoint(best_path, models, optimizer, epoch, best_val, logit_scale_param=logit_scale_param)
            print(f"[INFO] New best checkpoint: {best_path}")
        else:
            no_improve += 1

        scheduler.step()
        if no_improve >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch} (no val_loss improvement for {args.patience} epochs)")
            break

    print(f"[INFO] Training finished. Best val loss: {best_val:.6f}")
    print(f"[INFO] Finest checkpoint: {best_path}")


if __name__ == "__main__":
    main()
