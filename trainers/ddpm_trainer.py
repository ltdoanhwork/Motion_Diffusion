import torch
import torch.nn.functional as F
import random
import time
import os
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
        self.init_log_sigma_diff = float(getattr(args, "init_log_sigma_diff", 0.0))
        self.init_log_sigma_vel = float(getattr(args, "init_log_sigma_vel", 0.0))
        self.init_log_sigma_acc = float(getattr(args, "init_log_sigma_acc", 0.0))
        self.init_log_sigma_geom = float(getattr(args, "init_log_sigma_geom", 0.0))
        self.alphas_cumprod_t = torch.tensor(
            self.diffusion.alphas_cumprod, dtype=torch.float32, device=self.device
        )
        self._printed_geom_unit_info = False

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

    def generate_batch(self, caption, m_lens, dim_pose, guidance_scale=4.5):
        # 1. Encode text thật
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)

        # 2. Encode text rỗng (Unconditional)
        B = len(caption)
        uncond_caption = [""] * B
        xf_proj_uncond, xf_out_uncond = self.encoder.encode_text(uncond_caption, self.device)
        
        # --- MODEL WRAPPER ---
        def model_fn(x, t, **kwargs):
            # Lấy cond và uncond từ kwargs
            cond_dict = kwargs.get('cond')
            uncond_dict = kwargs.get('uncond')
            
            if cond_dict is None or uncond_dict is None:
                 # Fallback phòng trường hợp library truyền tham số kiểu khác
                 # (Thường không vào đây nếu gọi đúng chuẩn)
                 return self.encoder(x, t, **kwargs)

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
            out_combined = self.encoder(x_in, t_in, **combined_kwargs)
            
            # Tách kết quả ra
            out_cond, out_uncond = torch.split(out_combined, B, dim=0)
            
            # Công thức CFG
            return out_uncond + guidance_scale * (out_cond - out_uncond)

        T = min(m_lens.max(), self.encoder.num_frames)

        # Gọi p_sample_loop
        output = self.diffusion.p_sample_loop(
            model_fn,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                'cond': {'xf_proj': xf_proj, 'xf_out': xf_out, 'length': m_lens},
                'uncond': {'xf_proj': xf_proj_uncond, 'xf_out': xf_out_uncond, 'length': m_lens}
            },
            device=self.device 
        )
        
        return output
    
    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
        model_to_use = self.encoder_ema if self.encoder_ema is not None else self.encoder
        N = len(caption)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_caption = caption[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_caption = caption[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_caption, batch_m_lens, dim_pose)
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
        
        # THÊM: Velocity Loss (giảm jitter)
        if self.opt.use_velocity_loss:
            vel_gt = gt_motion[:, 1:] - gt_motion[:, :-1]
            vel_pred = pred_motion[:, 1:] - pred_motion[:, :-1]
            vel_mask = (src_mask[:, 1:] * src_mask[:, :-1]).unsqueeze(-1)  # (B, T-1, 1)
            vel_sq = self.mse_criterion(vel_pred, vel_gt) * vel_mask
            denom_vel = vel_mask.sum() * vel_pred.shape[-1] + 1e-8
            loss_vel = vel_sq.sum() / denom_vel
        
        # THÊM: Acceleration Loss (motion smoother)
        if self.opt.use_acceleration_loss and loss_vel is not None:
            acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
            acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            acc_mask = (vel_mask[:, 1:] * vel_mask[:, :-1])  # (B, T-2, 1)
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

    def _denormalize_motion(self, motion):
        if self.motion_mean is None or self.motion_std is None:
            return motion
        if motion.shape[-1] != self.motion_mean.shape[-1]:
            return motion
        return motion * self.motion_std + self.motion_mean

    def compute_geometric_loss(self, motion_pred, motion_gt, src_mask=None):
        # 2. Reshape về (B, T, J, 3) 
        # Model của bạn vẫn đang output Axis-Angle (3 channel) nên reshape về 3 là đúng
        feat_dim = motion_pred.shape[-1]
        if feat_dim % 3 != 0:
            raise ValueError(f"Expected axis-angle feature dim divisible by 3, got {feat_dim}")
        n_joints = feat_dim // 3
        
        motion_pred_reshaped = motion_pred.reshape(motion_pred.shape[0], motion_pred.shape[1], n_joints, 3)
        motion_gt_reshaped = motion_gt.reshape(motion_gt.shape[0], motion_gt.shape[1], n_joints, 3)
        
        # Chuyển từ Axis-Angle (3D) sang 6D Rotation để thỏa mãn yêu cầu của forward_kinematics mới
        motion_pred_6d = axis_angle_to_rotation_6d(motion_pred_reshaped)
        motion_gt_6d = axis_angle_to_rotation_6d(motion_gt_reshaped)
        
        # 3. Tính Positions (Giờ đầu vào đã là 6D -> Hợp lệ)
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

        # --- FK LOSS (Global Position Loss) ---
        fk_loss = torch.tensor(0.0, device=motion_pred.device, dtype=motion_pred.dtype)
        if getattr(self.opt, 'use_fk_loss', False): 
            # Tính MSE theo frame rồi mask các frame padding.
            fk_sq = (positions_pred - positions_gt) ** 2
            fk_per_frame = fk_sq.mean(dim=(-1, -2))
            fk_loss = (fk_per_frame * src_mask).sum() / valid_frames
            # In uncertainty-weighting mode, avoid manual hand-tuned scaling.
            if not self.use_uncertainty_weighting:
                fk_weight = getattr(self.opt, 'fk_weight', 1.0)
                fk_loss = fk_loss * fk_weight

        pred_parent_pos = torch.index_select(positions_pred, 2, self.bone_parent_indices)
        pred_child_pos = torch.index_select(positions_pred, 2, self.bone_child_indices)
        
        gt_parent_pos = torch.index_select(positions_gt, 2, self.bone_parent_indices)
        gt_child_pos = torch.index_select(positions_gt, 2, self.bone_child_indices)
        
        # Tính vector xương
        bone_pred = pred_child_pos - pred_parent_pos 
        bone_gt = gt_child_pos - gt_parent_pos      
        
        # Tính độ dài xương
        bone_len_pred = torch.norm(bone_pred, dim=-1, keepdim=True) + 1e-6
        bone_len_gt = torch.norm(bone_gt, dim=-1, keepdim=True) + 1e-6

        # Quick fix: auto-check unit scale and convert mm -> m if needed.
        # Typical human bone length in mm is often >100; in meters it's <2.
        median_bone_len = torch.median(bone_len_gt.detach())
        if median_bone_len > 100.0:
            positions_pred = positions_pred * 0.001
            positions_gt = positions_gt * 0.001
            bone_pred = bone_pred * 0.001
            bone_gt = bone_gt * 0.001
            bone_len_pred = bone_len_pred * 0.001
            bone_len_gt = bone_len_gt * 0.001
            if not self._printed_geom_unit_info:
                print("[INFO] Geometric/FK: detected mm-scale joints, converted to meters (x0.001).")
                self._printed_geom_unit_info = True
        elif not self._printed_geom_unit_info:
            print("[INFO] Geometric/FK: joint scale does not look like mm; keep original unit.")
            self._printed_geom_unit_info = True
        
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
        
        return dir_loss + len_loss + fk_loss

    def recover_root_translation(self, motion_data):
        """
        Khôi phục Global Position từ Velocity features.
        Mapping dựa trên Text2MotionDataset:
        - Index 0: Root Angular Velocity (Y-axis)
        - Index 1: Root Linear Velocity X (Local)
        - Index 2: Root Linear Velocity Z (Local)
        - Index 3: Root Height (Absolute Y)
        """
        # motion_data shape: (B, T, D)
        
        # 1. Tách dữ liệu
        r_rot_vel = motion_data[:, :, 0]      # (B, T)
        r_linear_vel = motion_data[:, :, 1:3] # (B, T, 2) -> (Vel_X, Vel_Z)
        r_y = motion_data[:, :, 3:4]          # (B, T, 1) -> Height
        
        # 2. Tính hướng mặt (Heading Angle) - Tích phân vận tốc góc
        # cumsum theo thời gian (dim=1)
        root_rot_angle = torch.cumsum(r_rot_vel, dim=1) # (B, T)
        
        # 3. Xoay vector vận tốc Local (X, Z) sang Global
        # Công thức xoay 2D:
        # x_global = x_local * cos(a) - z_local * sin(a)
        # z_global = x_local * sin(a) + z_local * cos(a)
        
        c = torch.cos(root_rot_angle)
        s = torch.sin(root_rot_angle)
        
        global_linear_vel = torch.zeros_like(r_linear_vel)
        global_linear_vel[:, :, 0] = r_linear_vel[:, :, 0] * c - r_linear_vel[:, :, 1] * s
        global_linear_vel[:, :, 1] = r_linear_vel[:, :, 0] * s + r_linear_vel[:, :, 1] * c
        
        # 4. Tính vị trí X, Z bằng cách cộng dồn (Tích phân)
        root_pos_xz = torch.cumsum(global_linear_vel, dim=1) # (B, T, 2)
        
        # 5. Ghép lại thành (X, Y, Z)
        # Lưu ý thứ tự trục của hệ tọa độ (thường là X, Y, Z)
        root_positions = torch.cat([
            root_pos_xz[:, :, 0:1],  # Global X
            r_y,                     # Global Y (Height có sẵn, không cần tích phân)
            root_pos_xz[:, :, 1:2]   # Global Z
        ], dim=-1) # (B, T, 3)
        
        return root_positions.unsqueeze(2) # (B, T, 1, 3) để khớp shape khớp xương

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

        # Root (Hips)
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
        warmup_steps = min(500, steps_per_epoch)  # Warmup 1 epoch hoặc 500 optimizer steps
        cosine_tmax = max(1, total_steps - warmup_steps)
        
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

            # Flush gradient còn dư nếu epoch kết thúc giữa chừng accumulation
            if accum_counter > 0:
                scaler.unscale_(self.opt_encoder)
                self.clip_norm([self.encoder])
                scaler.step(self.opt_encoder)
                scaler.update()

                self.update_ema()
                self.scheduler.step()
                self.zero_grad([self.opt_encoder])
            
            # Save checkpoint
            if epoch % self.opt.save_every_e == 0:
                self.save(
                    pjoin(self.opt.model_dir, 'ckpt_e%03d.tar' % epoch), 
                    epoch, 
                    total_it=it
                )
            
            print(f" Epoch {epoch} completed successfully. Time elapsed: {(time.time() - start_time)/60:.2f} mins.")
