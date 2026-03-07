import torch
import torch.nn.functional as F
import random
import time
import os
import json
import math
import numpy as np
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
from tqdm import tqdm
import copy
try:
    from torch.amp import autocast, GradScaler  # PyTorch >= 2.x preferred API
    _USE_TORCH_AMP = True
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast, GradScaler
    _USE_TORCH_AMP = False

# from mmcv.runner import get_dist_info
from models.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader
from datasets.emage_utils.rotation_conversions import axis_angle_to_rotation_6d, rotation_6d_to_matrix
import utils.paramUtil as paramUtil


class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.VELOCITY,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.RESCALED_MSE
        )
        

        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler
        self.use_min_snr_weighting = getattr(args, "use_min_snr_weighting", False)
        self.min_snr_gamma = float(getattr(args, "min_snr_gamma", 5.0))
        self.use_uncertainty_weighting = getattr(args, "use_uncertainty_weighting", False)
        self.uncertainty_lr_scale = float(getattr(args, "uncertainty_lr_scale", 10.0))
        self.use_sinkhorn_geom_loss = getattr(args, "use_sinkhorn_geom_loss", False)
        self.sinkhorn_epsilon = float(getattr(args, "sinkhorn_epsilon", 0.05))
        self.sinkhorn_iters = int(getattr(args, "sinkhorn_iters", 20))
        self.sinkhorn_max_frames = int(getattr(args, "sinkhorn_max_frames", 2048))
        self.use_sobolev_training = getattr(args, "use_sobolev_training", False)
        self.sobolev_weight = float(getattr(args, "sobolev_weight", 0.0))
        self.sobolev_lambda_h1 = float(getattr(args, "sobolev_lambda_h1", 1.0))
        self.sobolev_lambda_h2 = float(getattr(args, "sobolev_lambda_h2", 1.0))
        self.sobolev_order = int(getattr(args, "sobolev_order", 2))
        self.sobolev_stochastic = getattr(args, "sobolev_stochastic", False)
        self.sobolev_num_projections = int(getattr(args, "sobolev_num_projections", 1))
        self.init_log_sigma_diff = float(getattr(args, "init_log_sigma_diff", 0.0))
        self.init_log_sigma_vel = float(getattr(args, "init_log_sigma_vel", 0.0))
        self.init_log_sigma_acc = float(getattr(args, "init_log_sigma_acc", 0.0))
        self.init_log_sigma_geom = float(getattr(args, "init_log_sigma_geom", 0.0))
        self.alphas_cumprod_t = torch.tensor(
            self.diffusion.alphas_cumprod, dtype=torch.float32, device=self.device
        )
        self._printed_geom_unit_info = False
        self._last_geom_components = {}
        self.enable_body_part_control = getattr(args, "enable_body_part_control", False)
        self.enable_time_varied_control = getattr(args, "enable_time_varied_control", False)
        self.body_part_lambda1 = float(getattr(args, "body_part_lambda1", 0.0))
        self.time_varied_lambda2 = float(getattr(args, "time_varied_lambda2", 0.0))
        self.use_guidance_scheduling = getattr(args, "use_guidance_scheduling", False)
        self.guidance_schedule_type = str(getattr(args, "guidance_schedule_type", "constant"))
        self.use_inference_guidance = getattr(args, "use_inference_guidance", False)
        self.inference_guidance_mode = str(getattr(args, "inference_guidance_mode", "temporal_smooth"))
        self.inference_guidance_weight = float(getattr(args, "inference_guidance_weight", 0.0))
        self.inference_sampler = str(getattr(args, "inference_sampler", "ddpm")).lower()
        self.ddim_eta = float(getattr(args, "ddim_eta", 0.0))
        self.dpm_solverpp_steps = int(getattr(args, "dpm_solverpp_steps", 20))
        self.body_part_control_config = self._load_control_config(
            getattr(args, "body_part_control_config", ""),
            expected_key="parts",
        )
        self.time_varied_control_config = self._load_control_config(
            getattr(args, "time_varied_control_config", ""),
            expected_key="intervals",
        )

        if self.use_uncertainty_weighting:
            self.log_sigma_diff = torch.nn.Parameter(torch.tensor(self.init_log_sigma_diff, device=self.device))
            self.log_sigma_vel = torch.nn.Parameter(torch.tensor(self.init_log_sigma_vel, device=self.device))
            self.log_sigma_acc = torch.nn.Parameter(torch.tensor(self.init_log_sigma_acc, device=self.device))
            self.log_sigma_geom = torch.nn.Parameter(torch.tensor(self.init_log_sigma_geom, device=self.device))

        self.motion_mean = None
        self.motion_std = None
        mean_path = pjoin(getattr(self.opt, "meta_dir", ""), "mean.npy")
        std_path = pjoin(getattr(self.opt, "meta_dir", ""), "std.npy")
        if os.path.exists(mean_path) and os.path.exists(std_path):
            mean_np = np.load(mean_path).astype(np.float32)
            std_np = np.load(std_path).astype(np.float32)
            self.motion_mean = torch.from_numpy(mean_np).to(self.device).view(1, 1, -1)
            self.motion_std = torch.from_numpy(std_np).to(self.device).view(1, 1, -1)
        else:
            print("[WARN] mean/std not found in meta_dir; geometric loss will use normalized motion.")

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

        # THÊM: EMA model
        if args.ema_decay > 0:
            self.encoder_ema = copy.deepcopy(encoder)
            self.ema_decay = args.ema_decay
        else:
            self.encoder_ema = None

        self._init_skeleton_data()

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _as_length_list(length):
        if torch.is_tensor(length):
            return length.detach().cpu().tolist()
        return list(length)

    def _load_control_config(self, config_path, expected_key):
        if not config_path:
            return None
        if not os.path.exists(config_path):
            print(f"[WARN] Control config not found: {config_path}")
            return None
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to load control config {config_path}: {exc}")
            return None
        if not isinstance(cfg, dict) or expected_key not in cfg:
            print(f"[WARN] Invalid control config format in {config_path}. Missing key: {expected_key}")
            return None
        return cfg

    @staticmethod
    def _pairwise_smoothing_term(pred_terms):
        n_terms = len(pred_terms)
        if n_terms < 2:
            return torch.zeros_like(pred_terms[0]) if n_terms == 1 else None
        corr = torch.zeros_like(pred_terms[0])
        count = 0
        for i in range(n_terms):
            for j in range(i + 1, n_terms):
                diff = pred_terms[i] - pred_terms[j]
                denom = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)
                corr = corr + (diff / denom)
                count += 1
        return -corr / max(count, 1)

    @staticmethod
    def _build_feature_mask(indices, dim_pose, device, dtype):
        mask = torch.zeros(1, 1, dim_pose, device=device, dtype=dtype)
        if indices is None:
            return mask
        valid_indices = [int(i) for i in indices if 0 <= int(i) < dim_pose]
        if valid_indices:
            mask[:, :, valid_indices] = 1.0
        return mask

    @staticmethod
    def _interval_to_range(interval_cfg, total_len):
        start = interval_cfg.get("start", interval_cfg.get("l", 0))
        end = interval_cfg.get("end", interval_cfg.get("r", total_len))
        if isinstance(start, float) and 0.0 <= start <= 1.0:
            start = int(round(start * total_len))
        if isinstance(end, float) and 0.0 <= end <= 1.0:
            end = int(round(end * total_len))
        start = max(0, min(int(start), total_len))
        end = max(start + 1, min(int(end), total_len))
        return start, end

    def _expand_text_condition(self, text_value, batch_size):
        if isinstance(text_value, str):
            return [text_value] * batch_size
        if isinstance(text_value, list):
            if len(text_value) == batch_size:
                return text_value
            if len(text_value) == 1:
                return text_value * batch_size
        raise ValueError("Control text must be a string or list with length equal to batch size.")

    def _get_guidance_scale_t(self, t, guidance_scale):
        if not self.use_guidance_scheduling:
            return torch.full_like(t, float(guidance_scale), dtype=torch.float32)

        t_float = t.float()
        max_t = max(int(self.diffusion_steps) - 1, 1)
        if self.guidance_schedule_type == "sqrt_alpha":
            idx = t.long().clamp(min=0, max=max_t)
            alpha_bar_t = self.alphas_cumprod_t[idx].clamp(min=1e-8)
            mult = torch.sqrt(alpha_bar_t)
        elif self.guidance_schedule_type == "linear":
            mult = t_float / float(max_t)
        elif self.guidance_schedule_type == "sine":
            mult = torch.sin(math.pi * t_float / float(max_t))
        else:
            mult = torch.ones_like(t_float)
        return float(guidance_scale) * mult

    def _apply_inference_guidance(self, x, pred):
        if (not self.use_inference_guidance) or self.inference_guidance_weight == 0.0:
            return pred
        if self.inference_guidance_mode != "temporal_smooth":
            return pred
        if x.shape[1] < 3:
            return pred
        # Energy guidance with temporal smoothness prior: E = ||acc(x)||^2.
        # We use -grad(E) as correction direction.
        with torch.enable_grad():
            x_var = x.detach().requires_grad_(True)
            vel = x_var[:, 1:] - x_var[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            smooth_energy = (acc ** 2).mean()
            grad = torch.autograd.grad(smooth_energy, x_var, retain_graph=False, create_graph=False)[0]
        return pred - self.inference_guidance_weight * grad.detach()

    def _init_skeleton_data(self):
        """
        Hàm này chuẩn bị sẵn toàn bộ dữ liệu xương khớp lên GPU (self.device).
        Sau này khi train, model chỉ việc lấy ra dùng, không cần tạo lại.
        """
        raw_bone_pairs = paramUtil.beat_bone_pairs

        # Tối ưu: Tách thành 2 Tensor Parent và Child đưa lên GPU ngay lập tức
        parents = [p for p, c in raw_bone_pairs]
        children = [c for p, c in raw_bone_pairs]
        
        # Lưu vào self để dùng lại sau này
        self.bone_parent_indices = torch.LongTensor(parents).to(self.device)
        self.bone_child_indices = torch.LongTensor(children).to(self.device)

        raw_skeleton_tree = paramUtil.beat_skeleton_tree

        # Tối ưu: Pre-process offset thành Tensor trên GPU
        # Tạo một list mới chứa (parent, child, offset_tensor_gpu)
        self.optimized_skeleton_tree = []
        
        for parent, child, offset in raw_skeleton_tree:
            # Tạo tensor offset, đưa lên GPU
            offset_tensor = torch.tensor(offset, dtype=torch.float32, device=self.device)
            # Reshape sẵn thành (1, 1, 3, 1) để tiện broadcasting trong phép nhân ma trận sau này
            # Shape cũ: (3) -> Shape mới: (1, 1, 3, 1)
            offset_tensor = offset_tensor.view(1, 1, 3, 1)
            
            self.optimized_skeleton_tree.append((parent, child, offset_tensor))

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()

        self.caption = caption
        if self.opt.is_train and random.random() < 0.1:
            caption = [""] * len(caption)

        self.motions = motions
        x_start = motions
        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)
        self.timesteps = t
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']
        self.pred_xstart = output.get('pred_xstart', self.fake_noise)
        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def generate_batch(
        self,
        caption,
        m_lens,
        dim_pose,
        guidance_scale=4.5,
        body_part_control=None,
        time_varied_control=None,
        inpaint_motion=None,
        inpaint_mask=None,
        model=None,
    ):
        model = self._unwrap_model(self.encoder if model is None else model)
        m_lens_list = self._as_length_list(m_lens)
        B = len(caption)

        # 1. Encode text thật
        xf_proj, xf_out = model.encode_text(caption, self.device)

        # 2. Encode text rỗng (Unconditional)
        uncond_caption = [""] * B
        xf_proj_uncond, xf_out_uncond = model.encode_text(uncond_caption, self.device)

        body_cfg = None
        time_cfg = None
        if self.enable_body_part_control:
            body_cfg = body_part_control if body_part_control is not None else self.body_part_control_config
        if self.enable_time_varied_control:
            time_cfg = time_varied_control if time_varied_control is not None else self.time_varied_control_config

        # --- MODEL WRAPPER ---
        def model_fn(x, t, **kwargs):
            # Lấy cond và uncond từ kwargs
            cond_dict = kwargs.get('cond')
            uncond_dict = kwargs.get('uncond')
            
            if cond_dict is None or uncond_dict is None:
                 # Fallback phòng trường hợp library truyền tham số kiểu khác
                 # (Thường không vào đây nếu gọi đúng chuẩn)
                 return model(x, t, **kwargs)

            # x_in: (2*B, T, D)
            x_in = torch.cat([x, x], dim=0)
            t_in = torch.cat([t, t], dim=0)
            
            # Gộp các key trong dictionary
            combined_kwargs = {}
            for k in cond_dict.keys():
                if torch.is_tensor(cond_dict[k]):
                    combined_kwargs[k] = torch.cat([cond_dict[k], uncond_dict[k]], dim=0)
                else:
                    # Nếu là list (như mask length), cộng list lại
                    combined_kwargs[k] = cond_dict[k] + uncond_dict[k]

            # Unpack dictionary để truyền đúng vào các tham số (length, xf_proj, xf_out...)
            out_combined = model(x_in, t_in, **combined_kwargs)
            
            # Tách kết quả ra
            out_cond, out_uncond = torch.split(out_combined, B, dim=0)
            
            scale_t = self._get_guidance_scale_t(t, guidance_scale).to(device=x.device, dtype=out_uncond.dtype)
            scale_t = scale_t.view(B, *([1] * (out_uncond.dim() - 1)))
            # CFG with optional timestep scheduling.
            base_pred = out_uncond + scale_t * (out_cond - out_uncond)

            controlled_preds = []

            if isinstance(body_cfg, dict) and body_cfg.get("parts"):
                body_terms = []
                body_raw_preds = []
                coverage = torch.zeros_like(base_pred[:, :1, :])
                for part_cfg in body_cfg["parts"]:
                    part_text = self._expand_text_condition(part_cfg.get("text", ""), B)
                    part_weight = float(part_cfg.get("weight", 1.0))
                    part_xf_proj, part_xf_out = model.encode_text(part_text, self.device)
                    out_part = model(
                        x,
                        t,
                        length=cond_dict.get("length", m_lens_list),
                        xf_proj=part_xf_proj,
                        xf_out=part_xf_out,
                    )
                    guided_part = out_uncond + scale_t * (out_part - out_uncond)
                    feat_mask = self._build_feature_mask(
                        part_cfg.get("indices"),
                        dim_pose=x.shape[-1],
                        device=x.device,
                        dtype=x.dtype,
                    )
                    body_terms.append(guided_part * feat_mask * part_weight)
                    body_raw_preds.append(guided_part)
                    coverage = coverage + feat_mask

                if body_terms:
                    body_pred = torch.stack(body_terms, dim=0).sum(dim=0)
                    correction = self._pairwise_smoothing_term(body_raw_preds)
                    if correction is not None and self.body_part_lambda1 != 0.0:
                        body_pred = body_pred + self.body_part_lambda1 * correction
                    uncovered = (coverage <= 0).to(base_pred.dtype)
                    body_pred = body_pred + uncovered * base_pred
                    controlled_preds.append(body_pred)

            if isinstance(time_cfg, dict) and time_cfg.get("intervals"):
                time_terms = []
                time_raw_preds = []
                T = x.shape[1]
                coverage = torch.zeros(1, T, 1, device=x.device, dtype=x.dtype)
                for interval_cfg in time_cfg["intervals"]:
                    interval_text = self._expand_text_condition(interval_cfg.get("text", ""), B)
                    interval_weight = float(interval_cfg.get("weight", 1.0))
                    start, end = self._interval_to_range(interval_cfg, T)
                    time_xf_proj, time_xf_out = model.encode_text(interval_text, self.device)
                    out_time = model(
                        x,
                        t,
                        length=cond_dict.get("length", m_lens_list),
                        xf_proj=time_xf_proj,
                        xf_out=time_xf_out,
                    )
                    guided_time = out_uncond + scale_t * (out_time - out_uncond)
                    interval_mask = torch.zeros(1, T, 1, device=x.device, dtype=x.dtype)
                    interval_mask[:, start:end, :] = 1.0
                    padded_term = guided_time * interval_mask * interval_weight
                    time_terms.append(padded_term)
                    time_raw_preds.append(padded_term)
                    coverage = coverage + interval_mask

                if time_terms:
                    time_pred = torch.stack(time_terms, dim=0).sum(dim=0)
                    correction = self._pairwise_smoothing_term(time_raw_preds)
                    if correction is not None and self.time_varied_lambda2 != 0.0:
                        time_pred = time_pred + self.time_varied_lambda2 * correction
                    uncovered = (coverage <= 0).to(base_pred.dtype)
                    time_pred = time_pred + uncovered * base_pred
                    controlled_preds.append(time_pred)

            if controlled_preds:
                final_pred = torch.stack(controlled_preds, dim=0).mean(dim=0)
            else:
                final_pred = base_pred
            final_pred = self._apply_inference_guidance(x, final_pred)
            return final_pred

        T = min(max(m_lens_list), model.num_frames)

        inpaint_motion_tensor = None
        inpaint_mask_tensor = None
        if inpaint_motion is not None:
            if torch.is_tensor(inpaint_motion):
                inpaint_motion_tensor = inpaint_motion.to(self.device).float()
            else:
                inpaint_motion_tensor = torch.tensor(inpaint_motion, device=self.device, dtype=torch.float32)
            if inpaint_motion_tensor.dim() == 2:
                inpaint_motion_tensor = inpaint_motion_tensor.unsqueeze(0)
        if inpaint_mask is not None:
            if torch.is_tensor(inpaint_mask):
                inpaint_mask_tensor = inpaint_mask.to(self.device).float()
            else:
                inpaint_mask_tensor = torch.tensor(inpaint_mask, device=self.device, dtype=torch.float32)
            if inpaint_mask_tensor.dim() == 2:
                inpaint_mask_tensor = inpaint_mask_tensor.unsqueeze(0)

        common_kwargs = {
            'model_kwargs': {
                'cond': {'xf_proj': xf_proj, 'xf_out': xf_out, 'length': m_lens_list},
                'uncond': {'xf_proj': xf_proj_uncond, 'xf_out': xf_out_uncond, 'length': m_lens_list}
            },
            'device': self.device,
            'progress': True,
            'clip_denoised': False,
        }
        if self.inference_sampler == "ddpm":
            output = self.diffusion.p_sample_loop(
                model_fn,
                (B, T, dim_pose),
                inpaint_motion=inpaint_motion_tensor,
                inpaint_mask=inpaint_mask_tensor,
                **common_kwargs,
            )
        elif self.inference_sampler == "ddim":
            if inpaint_motion_tensor is not None or inpaint_mask_tensor is not None:
                print("[WARN] DDIM path currently ignores inpaint_motion/inpaint_mask; use inference_sampler=ddpm for inpainting.")
            output = self.diffusion.ddim_sample_loop(
                model_fn,
                (B, T, dim_pose),
                eta=self.ddim_eta,
                **common_kwargs,
            )
        elif self.inference_sampler == "dpm_solverpp":
            output = self.diffusion.dpm_solverpp_sample_loop(
                model_fn,
                (B, T, dim_pose),
                num_inference_steps=self.dpm_solverpp_steps,
                solver_order=2,
                inpaint_motion=inpaint_motion_tensor,
                inpaint_mask=inpaint_mask_tensor,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unknown inference_sampler: {self.inference_sampler}")
        
        return output
    
    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
        model_to_use = self.encoder_ema if self.encoder_ema is not None else self.encoder
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        if self.encoder_ema is not None:
            self.encoder_ema.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(
                batch_caption,
                batch_m_lens,
                dim_pose,
                guidance_scale=float(getattr(self.opt, "cfg_scale", 4.5)),
                model=model_to_use,
            )
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output

    def backward_G(self):
        # Keep loss computation in FP32 to avoid Half/Float dtype mismatch in backward.
        fake_noise = self.fake_noise.float()
        real_noise = self.real_noise.float()
        src_mask = self.src_mask.float()
        pred_motion = self.pred_xstart.float()
        gt_motion = self.motions.float()

        diff_per_frame = self.mse_criterion(fake_noise, real_noise).mean(dim=-1)  # (B, T)
        valid_per_sample = src_mask.sum(dim=1) + 1e-8
        diff_per_sample = (diff_per_frame * src_mask).sum(dim=1) / valid_per_sample  # (B,)
        if self.use_min_snr_weighting:
            alpha_bar = self.alphas_cumprod_t[self.timesteps.long()].clamp(min=1e-8, max=1.0 - 1e-8)
            snr_t = alpha_bar / (1.0 - alpha_bar)
            w_t = torch.clamp(snr_t, max=self.min_snr_gamma).to(diff_per_sample.dtype)
            loss_diff = (w_t * diff_per_sample).mean()
        else:
            loss_diff = diff_per_sample.mean()
        
        loss_vel = None
        loss_acc = None
        loss_geom = None
        
        # Velocity loss (H1 Sobolev term when enabled)
        if self.opt.use_velocity_loss:
            vel_gt = gt_motion[:, 1:] - gt_motion[:, :-1]
            vel_pred = pred_motion[:, 1:] - pred_motion[:, :-1]
            vel_mask = (src_mask[:, 1:] * src_mask[:, :-1]).unsqueeze(-1)  # (B, T-1, 1)
            use_stochastic_sobolev = self.use_sobolev_training and self.sobolev_stochastic
            if use_stochastic_sobolev:
                vel_valid = vel_mask.expand_as(vel_pred) > 0.5
                if vel_valid.any():
                    vel_pred_valid = vel_pred[vel_valid].view(-1, vel_pred.shape[-1])
                    vel_gt_valid = vel_gt[vel_valid].view(-1, vel_gt.shape[-1])
                    loss_vel = self._random_projection_mse(vel_pred_valid, vel_gt_valid)
                else:
                    loss_vel = torch.tensor(0.0, device=pred_motion.device, dtype=pred_motion.dtype)
            else:
                vel_sq = self.mse_criterion(vel_pred, vel_gt) * vel_mask
                denom_vel = vel_mask.sum() * vel_pred.shape[-1] + 1e-8
                loss_vel = vel_sq.sum() / denom_vel
        
        # Acceleration loss (H2 Sobolev term when enabled)
        if self.opt.use_acceleration_loss and loss_vel is not None:
            acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
            acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            acc_mask = (vel_mask[:, 1:] * vel_mask[:, :-1])  # (B, T-2, 1)
            use_stochastic_sobolev = self.use_sobolev_training and self.sobolev_stochastic
            if use_stochastic_sobolev:
                acc_valid = acc_mask.expand_as(acc_pred) > 0.5
                if acc_valid.any():
                    acc_pred_valid = acc_pred[acc_valid].view(-1, acc_pred.shape[-1])
                    acc_gt_valid = acc_gt[acc_valid].view(-1, acc_gt.shape[-1])
                    loss_acc = self._random_projection_mse(acc_pred_valid, acc_gt_valid)
                else:
                    loss_acc = torch.tensor(0.0, device=pred_motion.device, dtype=pred_motion.dtype)
            else:
                acc_sq = self.mse_criterion(acc_pred, acc_gt) * acc_mask
                denom_acc = acc_mask.sum() * acc_pred.shape[-1] + 1e-8
                loss_acc = acc_sq.sum() / denom_acc
        
        # THÊM: Geometric Loss (Bone length consistency)
        if self.opt.use_geometric_loss:
            pred_motion_geom = self._denormalize_motion(pred_motion)
            gt_motion_geom = self._denormalize_motion(gt_motion)
            # Geometric/FK path is numerically sensitive under FP16; force FP32.
            if _USE_TORCH_AMP:
                with autocast(device_type='cuda', enabled=False):
                    loss_geom = self.compute_geometric_loss(
                        pred_motion_geom.float(),
                        gt_motion_geom.float(),
                        src_mask=src_mask,
                    )
            else:
                with autocast(enabled=False):
                    loss_geom = self.compute_geometric_loss(
                        pred_motion_geom.float(),
                        gt_motion_geom.float(),
                        src_mask=src_mask,
                    )

        if self.use_uncertainty_weighting:
            contrib_diff_data = 0.5 * torch.exp(-2.0 * self.log_sigma_diff) * loss_diff
            contrib_diff_total = contrib_diff_data + self.log_sigma_diff
            total_loss = contrib_diff_total
            contrib_vel_data = torch.tensor(0.0, device=total_loss.device)
            contrib_acc_data = torch.tensor(0.0, device=total_loss.device)
            contrib_geom_data = torch.tensor(0.0, device=total_loss.device)
            contrib_vel_total = torch.tensor(0.0, device=total_loss.device)
            contrib_acc_total = torch.tensor(0.0, device=total_loss.device)
            contrib_geom_total = torch.tensor(0.0, device=total_loss.device)
            if self.opt.use_velocity_loss and loss_vel is not None:
                contrib_vel_data = 0.5 * torch.exp(-2.0 * self.log_sigma_vel) * loss_vel
                contrib_vel_total = contrib_vel_data + self.log_sigma_vel
                total_loss = total_loss + contrib_vel_total
            if self.opt.use_acceleration_loss and loss_acc is not None:
                contrib_acc_data = 0.5 * torch.exp(-2.0 * self.log_sigma_acc) * loss_acc
                contrib_acc_total = contrib_acc_data + self.log_sigma_acc
                total_loss = total_loss + contrib_acc_total
            if self.opt.use_geometric_loss and loss_geom is not None:
                contrib_geom_data = 0.5 * torch.exp(-2.0 * self.log_sigma_geom) * loss_geom
                contrib_geom_total = contrib_geom_data + self.log_sigma_geom
                total_loss = total_loss + contrib_geom_total
        else:
            total_loss = loss_diff
            if self.opt.use_velocity_loss and loss_vel is not None:
                total_loss = total_loss + self.opt.velocity_weight * loss_vel
            if self.opt.use_acceleration_loss and loss_acc is not None:
                total_loss = total_loss + self.opt.acceleration_weight * loss_acc
            if self.opt.use_geometric_loss and loss_geom is not None:
                total_loss = total_loss + self.opt.geometric_weight * loss_geom

        if not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite loss detected (nan/inf). Try lower aux loss weights or disable AMP.")

        self.loss_mot_rec = total_loss
        loss_logs = OrderedDict({
            'loss_mot_rec': self.loss_mot_rec.item()
        })
        loss_logs['loss_diff'] = loss_diff.item()
        if self.use_min_snr_weighting:
            loss_logs['loss_diff_minsnr'] = loss_diff.item()
        
        if self.opt.use_velocity_loss:
            loss_logs['loss_vel'] = loss_vel.item()
        if self.opt.use_acceleration_loss:
            loss_logs['loss_acc'] = loss_acc.item()
        if self.opt.use_geometric_loss:
            loss_logs['loss_geom'] = loss_geom.item()
            if self._last_geom_components:
                loss_logs['loss_dir'] = self._last_geom_components.get('loss_dir', torch.tensor(0.0, device=self.device)).item()
                loss_logs['loss_len'] = self._last_geom_components.get('loss_len', torch.tensor(0.0, device=self.device)).item()
                loss_logs['loss_fk'] = self._last_geom_components.get('loss_fk', torch.tensor(0.0, device=self.device)).item()
        if self.use_uncertainty_weighting:
            loss_logs['sigma_diff'] = torch.exp(self.log_sigma_diff).item()
            loss_logs['sigma_vel'] = torch.exp(self.log_sigma_vel).item()
            loss_logs['sigma_acc'] = torch.exp(self.log_sigma_acc).item()
            loss_logs['sigma_geom'] = torch.exp(self.log_sigma_geom).item()
            loss_logs['contrib_diff_data'] = contrib_diff_data.item()
            loss_logs['contrib_diff_total'] = contrib_diff_total.item()
            loss_logs['contrib_vel_data'] = contrib_vel_data.item()
            loss_logs['contrib_vel_total'] = contrib_vel_total.item()
            loss_logs['contrib_acc_data'] = contrib_acc_data.item()
            loss_logs['contrib_acc_total'] = contrib_acc_total.item()
            loss_logs['contrib_geom_data'] = contrib_geom_data.item()
            loss_logs['contrib_geom_total'] = contrib_geom_total.item()
        
        return loss_logs

    def _random_projection_mse(self, a, b):
        """
        Stochastic Sobolev approximation:
        compare random directional projections instead of full Jacobian-like tensors.
        a, b: (..., D)
        """
        D = a.shape[-1]
        n_proj = max(self.sobolev_num_projections, 1)
        total = torch.tensor(0.0, device=a.device, dtype=a.dtype)
        for _ in range(n_proj):
            v = torch.randn(D, device=a.device, dtype=a.dtype)
            v = v / (torch.norm(v) + 1e-8)
            proj_a = torch.matmul(a, v)
            proj_b = torch.matmul(b, v)
            total = total + ((proj_a - proj_b) ** 2).mean()
        return total / n_proj

    def compute_sobolev_loss(self, motion_pred, motion_gt, src_mask=None):
        """
        Sobolev regularization (H1/H2) over temporal derivatives:
        L_sob = lambda1 * ||d_t xhat - d_t x||^2 + lambda2 * ||d_tt xhat - d_tt x||^2
        Supports stochastic directional approximation.
        """
        if src_mask is None:
            src_mask = torch.ones(
                motion_pred.shape[0],
                motion_pred.shape[1],
                device=motion_pred.device,
                dtype=motion_pred.dtype,
            )
        else:
            src_mask = src_mask.to(device=motion_pred.device, dtype=motion_pred.dtype)

        vel_gt = motion_gt[:, 1:] - motion_gt[:, :-1]
        vel_pred = motion_pred[:, 1:] - motion_pred[:, :-1]
        vel_mask = (src_mask[:, 1:] * src_mask[:, :-1]).unsqueeze(-1)
        vel_valid = vel_mask.expand_as(vel_pred) > 0.5

        if vel_valid.any():
            vel_pred_valid = vel_pred[vel_valid].view(-1, vel_pred.shape[-1])
            vel_gt_valid = vel_gt[vel_valid].view(-1, vel_gt.shape[-1])
            if self.sobolev_stochastic:
                h1 = self._random_projection_mse(vel_pred_valid, vel_gt_valid)
            else:
                h1 = F.mse_loss(vel_pred_valid, vel_gt_valid)
        else:
            h1 = torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype)

        total = self.sobolev_lambda_h1 * h1
        if self.sobolev_order >= 2:
            acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
            acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            acc_mask = vel_mask[:, 1:] * vel_mask[:, :-1]
            acc_valid = acc_mask.expand_as(acc_pred) > 0.5
            if acc_valid.any():
                acc_pred_valid = acc_pred[acc_valid].view(-1, acc_pred.shape[-1])
                acc_gt_valid = acc_gt[acc_valid].view(-1, acc_gt.shape[-1])
                if self.sobolev_stochastic:
                    h2 = self._random_projection_mse(acc_pred_valid, acc_gt_valid)
                else:
                    h2 = F.mse_loss(acc_pred_valid, acc_gt_valid)
            else:
                h2 = torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype)
            total = total + self.sobolev_lambda_h2 * h2
        return total

    def _denormalize_motion(self, motion):
        if self.motion_mean is None or self.motion_std is None:
            return motion
        if motion.shape[-1] != self.motion_mean.shape[-1]:
            return motion
        return motion * self.motion_std + self.motion_mean

    def _sinkhorn_ot_cost(self, x, y):
        """
        Entropic OT cost between paired point clouds x and y.
        x, y: (N, J, 3), uniform marginals.
        Return: (N,) OT_epsilon(x, y)
        """
        eps = max(self.sinkhorn_epsilon, 1e-6)
        iters = max(self.sinkhorn_iters, 1)
        N, J, _ = x.shape
        C = torch.cdist(x, y, p=2) ** 2  # (N, J, J)

        log_a = -torch.log(torch.tensor(float(J), device=x.device, dtype=x.dtype))
        log_b = log_a
        f = torch.zeros((N, J), device=x.device, dtype=x.dtype)
        g = torch.zeros((N, J), device=x.device, dtype=x.dtype)

        for _ in range(iters):
            f = eps * (log_a - torch.logsumexp((g.unsqueeze(1) - C) / eps, dim=2))
            g = eps * (log_b - torch.logsumexp((f.unsqueeze(2) - C) / eps, dim=1))

        log_pi = (f.unsqueeze(2) + g.unsqueeze(1) - C) / eps
        pi = torch.exp(log_pi)
        return (pi * C).sum(dim=(1, 2))

    def _sinkhorn_divergence(self, x, y):
        """
        Sinkhorn divergence:
        S_eps(x, y) = OT_eps(x, y) - 0.5 OT_eps(x, x) - 0.5 OT_eps(y, y)
        """
        ot_xy = self._sinkhorn_ot_cost(x, y)
        ot_xx = self._sinkhorn_ot_cost(x, x)
        ot_yy = self._sinkhorn_ot_cost(y, y)
        return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy

    def compute_geometric_loss(self, motion_pred, motion_gt, src_mask=None):
        feat_dim = motion_pred.shape[-1]
        rep = str(getattr(self.opt, "motion_rep", "axis_angle")).lower()
        if rep in ("rot6d", "rotation_6d", "6d"):
            if feat_dim % 6 != 0:
                raise ValueError(f"Expected rot6d feature dim divisible by 6, got {feat_dim}")
            n_joints = feat_dim // 6
            motion_pred_6d = motion_pred.reshape(motion_pred.shape[0], motion_pred.shape[1], n_joints, 6)
            motion_gt_6d = motion_gt.reshape(motion_gt.shape[0], motion_gt.shape[1], n_joints, 6)
        else:
            if feat_dim % 3 != 0:
                raise ValueError(f"Expected axis-angle feature dim divisible by 3, got {feat_dim}")
            n_joints = feat_dim // 3
            motion_pred_reshaped = motion_pred.reshape(motion_pred.shape[0], motion_pred.shape[1], n_joints, 3)
            motion_gt_reshaped = motion_gt.reshape(motion_gt.shape[0], motion_gt.shape[1], n_joints, 3)
            # Chuyển từ Axis-Angle (3D) sang 6D Rotation để thỏa mãn yêu cầu của forward_kinematics mới
            motion_pred_6d = axis_angle_to_rotation_6d(motion_pred_reshaped)
            motion_gt_6d = axis_angle_to_rotation_6d(motion_gt_reshaped)
        
        positions_pred = self.forward_kinematics(motion_pred_6d, motion_raw=motion_pred) 
        positions_gt = self.forward_kinematics(motion_gt_6d, motion_raw=motion_gt)

        if src_mask is None:
            src_mask = torch.ones(
                motion_pred.shape[0],
                motion_pred.shape[1],
                device=motion_pred.device,
                dtype=motion_pred.dtype,
            )
        else:
            src_mask = src_mask.to(device=motion_pred.device, dtype=motion_pred.dtype)
        valid_frames = src_mask.sum() + 1e-8

        # Build bone vectors once to infer scale and for classical geom loss.
        pred_parent_pos = torch.index_select(positions_pred, 2, self.bone_parent_indices)
        pred_child_pos = torch.index_select(positions_pred, 2, self.bone_child_indices)
        gt_parent_pos = torch.index_select(positions_gt, 2, self.bone_parent_indices)
        gt_child_pos = torch.index_select(positions_gt, 2, self.bone_child_indices)
        bone_pred = pred_child_pos - pred_parent_pos
        bone_gt = gt_child_pos - gt_parent_pos
        bone_len_pred = torch.norm(bone_pred, dim=-1, keepdim=True) + 1e-6
        bone_len_gt = torch.norm(bone_gt, dim=-1, keepdim=True) + 1e-6

        # BEAT offsets are in centimeters: force cm -> m to stabilize FK/geometric scale.
        scale_factor = 0.01
        positions_pred = positions_pred * scale_factor
        positions_gt = positions_gt * scale_factor
        bone_pred = bone_pred * scale_factor
        bone_gt = bone_gt * scale_factor
        bone_len_pred = bone_len_pred * scale_factor
        bone_len_gt = bone_len_gt * scale_factor
        if not self._printed_geom_unit_info:
            print("[INFO] Geometric/FK: converted BEAT joint scale from cm to m (x0.01).")
            self._printed_geom_unit_info = True
        
        # Optional: Sinkhorn divergence geometric loss (distribution-wise on point clouds).
        if self.use_sinkhorn_geom_loss:
            valid_idx = (src_mask > 0.5).reshape(-1)
            pred_clouds = positions_pred.reshape(-1, n_joints, 3)[valid_idx]
            gt_clouds = positions_gt.reshape(-1, n_joints, 3)[valid_idx]
            if pred_clouds.shape[0] == 0:
                self._last_geom_components = {
                    'loss_dir': torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype),
                    'loss_len': torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype),
                    'loss_fk': torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype),
                }
                return torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype)
            if self.sinkhorn_max_frames > 0 and pred_clouds.shape[0] > self.sinkhorn_max_frames:
                perm = torch.randperm(pred_clouds.shape[0], device=pred_clouds.device)[:self.sinkhorn_max_frames]
                pred_clouds = pred_clouds[perm]
                gt_clouds = gt_clouds[perm]
            sinkhorn_vals = self._sinkhorn_divergence(pred_clouds, gt_clouds)
            self._last_geom_components = {
                'loss_dir': torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype),
                'loss_len': torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype),
                'loss_fk': sinkhorn_vals.mean(),
            }
            return sinkhorn_vals.mean()

        # --- Classical geometric/FK loss path ---
        fk_loss = torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype)
        if getattr(self.opt, 'use_fk_loss', False):
            fk_sq = (positions_pred - positions_gt) ** 2
            fk_per_frame = fk_sq.mean(dim=(-1, -2))
            fk_loss = (fk_per_frame * src_mask).sum() / valid_frames
            if not self.use_uncertainty_weighting:
                fk_weight = getattr(self.opt, 'fk_weight', 1.0)
                fk_loss = fk_loss * fk_weight

        # Loss 1: Direction (Cosine similarity)
        bone_pred_norm = bone_pred / bone_len_pred
        bone_gt_norm = bone_gt / bone_len_gt
        dir_per_bone = 1 - (bone_pred_norm * bone_gt_norm).sum(dim=-1)
        dir_per_frame = dir_per_bone.mean(dim=-1)
        dir_loss = (dir_per_frame * src_mask).sum() / valid_frames
        
        # Loss 2: Length consistency
        len_per_bone = torch.abs(bone_len_pred - bone_len_gt).squeeze(-1)
        len_per_frame = len_per_bone.mean(dim=-1)
        len_loss = (len_per_frame * src_mask).sum() / valid_frames
        self._last_geom_components = {
            'loss_dir': dir_loss,
            'loss_len': len_loss,
            'loss_fk': fk_loss,
        }
        return dir_loss + len_loss + fk_loss

    def recover_root_translation(self, motion_data):
        """
        Recover root translation for FK.
        For BEAT axis-angle representation, root translation channels are not
        explicitly available in the motion vector, so we keep root fixed at origin.
        """
        B, T, _ = motion_data.shape
        return torch.zeros((B, T, 1, 3), device=motion_data.device, dtype=motion_data.dtype)

    def forward_kinematics(self, rot6d, motion_raw=None):
        """
        Input: rot6d có shape (B, T, n_joints, 6)
        Output: positions có shape (B, T, n_joints, 3)
        """
        B, T, n_joints, dims = rot6d.shape
        
        # Kiểm tra nhanh: Nếu input là 6D thì dims phải bằng 6
        assert dims == 6, f"Input cho hàm này phải là 6D, nhưng nhận được {dims}D"
        
        # 2. Chuyển đổi từ 6D sang Matrix (3x3)
        # Input: (B, T, J, 6) -> Output: (B, T, J, 3, 3)
        local_rot_mats = rotation_6d_to_matrix(rot6d)
                
        # NOTE: Avoid in-place writes on tensors that participate in autograd.
        # Build FK results as python lists then stack once at the end.
        global_rot_mats = [None] * n_joints
        positions = [None] * n_joints

        # Root (Hips): fixed at origin for BEAT axis-angle path.
        if motion_raw is not None:
            root_pos = self.recover_root_translation(motion_raw)  # (B, T, 1, 3)
            positions[0] = root_pos.squeeze(2)
        else:
            positions[0] = torch.zeros(B, T, 3, device=rot6d.device, dtype=rot6d.dtype)

        global_rot_mats[0] = local_rot_mats[:, :, 0]

        # Duyệt qua cây xương đã tối ưu (Parent -> Child)
        for parent, child, offset in self.optimized_skeleton_tree:
            parent_rot = global_rot_mats[parent]
            if parent_rot is None:
                continue

            # Global rotation của child
            child_rot = torch.matmul(parent_rot, local_rot_mats[:, :, child])
            # Position của child: parent_pos + parent_rot @ offset
            rot_offset = torch.matmul(parent_rot, offset).squeeze(-1)
            child_pos = positions[parent] + rot_offset

            global_rot_mats[child] = child_rot
            positions[child] = child_pos

        # Fallback for any disconnected/unused joints
        for j in range(n_joints):
            if positions[j] is None:
                positions[j] = positions[0]

        return torch.stack(positions, dim=2)
    
    def update_ema(self):
        """Update EMA parameters"""
        if self.encoder_ema is None:
            return
        
        with torch.no_grad():
            for p_ema, p in zip(self.encoder_ema.parameters(), self.encoder.parameters()):
                p_ema.data = self.ema_decay * p_ema.data + (1 - self.ema_decay) * p.data

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        self.update_ema() 
        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)
        self.alphas_cumprod_t = self.alphas_cumprod_t.to(device)
        if self.motion_mean is not None and self.motion_std is not None:
            self.motion_mean = self.motion_mean.to(device)
            self.motion_std = self.motion_std.to(device)
        if self.use_uncertainty_weighting:
            self.log_sigma_diff.data = self.log_sigma_diff.data.to(device)
            self.log_sigma_vel.data = self.log_sigma_vel.data.to(device)
            self.log_sigma_acc.data = self.log_sigma_acc.data.to(device)
            self.log_sigma_geom.data = self.log_sigma_geom.data.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        if hasattr(self, 'scheduler'):
            state['scheduler'] = self.scheduler.state_dict()
        if self.use_uncertainty_weighting:
            state['uncertainty'] = {
                'log_sigma_diff': self.log_sigma_diff.detach().cpu(),
                'log_sigma_vel': self.log_sigma_vel.detach().cpu(),
                'log_sigma_acc': self.log_sigma_acc.detach().cpu(),
                'log_sigma_geom': self.log_sigma_geom.detach().cpu(),
            }
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.use_uncertainty_weighting and 'uncertainty' in checkpoint:
            unc = checkpoint['uncertainty']
            self.log_sigma_diff.data.copy_(unc['log_sigma_diff'].to(self.device))
            self.log_sigma_vel.data.copy_(unc['log_sigma_vel'].to(self.device))
            self.log_sigma_acc.data.copy_(unc['log_sigma_acc'].to(self.device))
            self.log_sigma_geom.data.copy_(unc['log_sigma_geom'].to(self.device))
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        self.to(self.device)
        optim_params = [{'params': list(self.encoder.parameters()), 'lr': self.opt.lr}]
        if self.use_uncertainty_weighting:
            sigma_lr = self.opt.lr * self.uncertainty_lr_scale
            optim_params.append({
                'params': [
                    self.log_sigma_diff,
                    self.log_sigma_vel,
                    self.log_sigma_acc,
                    self.log_sigma_geom,
                ],
                'lr': sigma_lr,
                'weight_decay': 0.0,
            })
            print(f"[INFO] Uncertainty LR group enabled: sigma_lr={sigma_lr:.2e}")
        self.opt_encoder = optim.AdamW(optim_params, lr=self.opt.lr)
        
        # Build dataloader TRƯỚC khi tạo scheduler
        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True)

        grad_accum_steps = max(1, int(getattr(self.opt, "gradient_accumulation_steps", 1)))

        # Tính đúng T_max theo số optimizer steps (không phải micro-batches)
        steps_per_epoch = max(1, (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps)
        total_steps = steps_per_epoch * self.opt.num_epochs
        # warmup_steps = min(500, steps_per_epoch) 
        warmup_steps = int(0.05 * total_steps)
        # cosine_tmax = max(1, total_steps - warmup_steps)
        cosine_tmax = total_steps - warmup_steps
        
        warmup_scheduler = LinearLR(
            self.opt_encoder, 
            start_factor=0.1, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.opt_encoder, 
            T_max=cosine_tmax,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.opt_encoder,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)
        
        start_time = time.time()
        amp_enabled = torch.cuda.is_available() and getattr(self.opt, "use_amp", False)
        if _USE_TORCH_AMP:
            scaler = GradScaler("cuda", enabled=amp_enabled)
        else:
            scaler = GradScaler(enabled=amp_enabled)
        logs = OrderedDict()
        
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            accum_counter = 0
            self.zero_grad([self.opt_encoder])
            
            for i, batch_data in enumerate(epoch_bar):
                if batch_data is None:
                    continue
                
                # Forward diffusion path under AMP
                if _USE_TORCH_AMP:
                    amp_ctx = autocast(device_type='cuda', enabled=amp_enabled)
                else:
                    amp_ctx = autocast(enabled=amp_enabled)

                with amp_ctx:
                    self.forward(batch_data)
                # Compute composite losses outside AMP for better numerical stability.
                loss_logs = self.backward_G()

                # Backward theo micro-batch, normalize loss theo accumulation steps
                scaled_loss = self.loss_mot_rec / grad_accum_steps
                scaler.scale(scaled_loss).backward()
                accum_counter += 1

                # Chỉ cập nhật optimizer khi đủ accumulation window
                if accum_counter >= grad_accum_steps:
                    scaler.unscale_(self.opt_encoder)
                    self.clip_norm([self.encoder])
                    scaler.step(self.opt_encoder)
                    scaler.update()

                    self.update_ema()
                    self.scheduler.step()
                    self.zero_grad([self.opt_encoder])
                    accum_counter = 0
                
                # Logging
                for k, v in loss_logs.items():
                    logs[k] = logs.get(k, 0) + v
                
                it += 1
                
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in mean_loss.items()})
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                # Save latest checkpoint periodically by iteration
                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # Flush gradient còn dư nếu epoch kết thúc giữa chừng accumulation
            if accum_counter > 0:
                scaler.unscale_(self.opt_encoder)
                self.clip_norm([self.encoder])
                scaler.step(self.opt_encoder)
                scaler.update()

                self.update_ema()
                self.scheduler.step()
                self.zero_grad([self.opt_encoder])
            
            # Save epoch checkpoint using 1-based epoch index
            if (epoch + 1) % self.opt.save_every_e == 0:
                self.save(
                    pjoin(self.opt.model_dir, 'ckpt_e%03d.tar' % (epoch + 1)), 
                    epoch, 
                    total_it=it
                )

            # Always refresh latest checkpoint at the end of each epoch
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            
            print(f" Epoch {epoch} completed successfully. Time elapsed: {(time.time() - start_time)/60:.2f} mins.")
