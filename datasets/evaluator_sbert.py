import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluator_models import MovementConvEncoder, MotionEncoderBiGRUCo


class SentenceTextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", trainable=False):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "transformers is required for SBERT evaluator. Install it with: pip install transformers"
            ) from e
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if not trainable:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @staticmethod
    def _mean_pool(last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def forward(self, texts, device):
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        pooled = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        return pooled


class SBERTMotionTextModel(nn.Module):
    def __init__(
        self,
        dim_pose=264,
        dim_movement_latent=512,
        dim_motion_hidden=1024,
        dim_coemb_hidden=512,
        unit_length=4,
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        train_text_encoder=False,
    ):
        super().__init__()
        self.unit_length = unit_length
        self.movement_encoder = MovementConvEncoder(dim_pose - 4, 512, dim_movement_latent)
        self.motion_encoder = MotionEncoderBiGRUCo(
            input_size=dim_movement_latent,
            hidden_size=dim_motion_hidden,
            output_size=dim_coemb_hidden,
            device=torch.device("cpu"),
        )
        self.text_encoder = SentenceTextEncoder(model_name=text_model_name, trainable=train_text_encoder)
        text_dim = self.text_encoder.model.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, dim_coemb_hidden),
            nn.LayerNorm(dim_coemb_hidden),
            nn.GELU(),
            nn.Linear(dim_coemb_hidden, dim_coemb_hidden),
        )
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

    def encode_motion(self, motions, m_lens):
        m_lens = torch.clamp(m_lens.long(), min=1, max=motions.shape[1])
        align_idx = torch.argsort(m_lens, descending=True)
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]
        movements = self.movement_encoder(motions[..., :-4])
        motion_lens = torch.clamp(m_lens // self.unit_length, min=1, max=movements.shape[1])
        motion_emb = self.motion_encoder(movements, motion_lens)
        inv_idx = torch.argsort(align_idx)
        return motion_emb[inv_idx]

    def encode_text(self, captions, device):
        txt = self.text_encoder(captions, device)
        return self.text_proj(txt)

    def similarity_logits(self, text_emb, motion_emb):
        text_emb = F.normalize(text_emb, dim=-1)
        motion_emb = F.normalize(motion_emb, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.matmul(text_emb, motion_emb.t()) * scale


class EvaluatorModelWrapperSBERT:
    def __init__(self, opt):
        if not hasattr(opt, "evaluator_path"):
            raise ValueError("Missing opt.evaluator_path for SBERT evaluator")
        if not os.path.exists(opt.evaluator_path):
            raise FileNotFoundError(f"Evaluator checkpoint not found: {opt.evaluator_path}")

        ckpt = torch.load(opt.evaluator_path, map_location=opt.device)
        cfg = ckpt.get("config", {})
        dim_pose = int(cfg.get("dim_pose", getattr(opt, "dim_pose", 264)))
        unit_length = int(cfg.get("unit_length", getattr(opt, "unit_length", 4)))
        text_model_name = cfg.get("text_model_name", "sentence-transformers/all-MiniLM-L6-v2")

        self.model = SBERTMotionTextModel(
            dim_pose=dim_pose,
            unit_length=unit_length,
            text_model_name=text_model_name,
            train_text_encoder=False,
        )
        self.model.movement_encoder.load_state_dict(ckpt["movement_encoder"])
        self.model.motion_encoder.load_state_dict(ckpt["motion_encoder"])
        self.model.text_proj.load_state_dict(ckpt["text_proj"])
        if "logit_scale" in ckpt and ckpt["logit_scale"] is not None:
            with torch.no_grad():
                self.model.logit_scale.copy_(ckpt["logit_scale"].to(self.model.logit_scale.device))

        self.device = opt.device
        self.model.to(self.device)
        self.model.eval()
        print(f"Loading SBERT Evaluation Wrapper (Epoch {ckpt.get('epoch', -1)}) Completed!!")

    def get_co_embeddings_from_captions(self, captions, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device).long()
            text_emb = self.model.encode_text(captions, self.device)
            motion_emb = self.model.encode_motion(motions, m_lens)
        return text_emb, motion_emb

    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens.detach().to(self.device).long()
            motion_emb = self.model.encode_motion(motions, m_lens)
        return motion_emb
