import torch
import torch.nn.functional as F
import random
import time
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
from torch.cuda.amp import autocast, GradScaler

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
        # 1. DANH SÁCH BONE PAIRS (Dùng cho Geometric Loss)
        # Copy nguyên list bone_pairs dài ngoằng vào đây
        raw_bone_pairs = [
            (0, 1),  # Hips -> Spine
            (1, 2),  # Spine -> Spine1
            (2, 3),  # Spine1 -> Spine2
            (3, 4),  # Spine2 -> Spine3
            (4, 5),  # Spine3 -> Neck
            (5, 6),  # Neck -> Neck1
            (6, 7),  # Neck1 -> Head
            (7, 8),  # Head -> HeadEnd
            (8, 9),  # HeadEnd -> HeadEnd_Nub
            (4, 10),  # Spine3 -> RightShoulder
            (10, 11),  # RightShoulder -> RightArm
            (11, 12),  # RightArm -> RightForeArm
            (12, 13),  # RightForeArm -> RightHand
            (13, 14),  # RightHand -> RightHandMiddle1
            (14, 15),  # RightHandMiddle1 -> RightHandMiddle2
            (15, 16),  # RightHandMiddle2 -> RightHandMiddle3
            (16, 17),  # RightHandMiddle3 -> RightHandMiddle4
            (17, 18),  # RightHandMiddle4 -> RightHandMiddle4_Nub
            (13, 19),  # RightHand -> RightHandRing
            (19, 20),  # RightHandRing -> RightHandRing1
            (20, 21),  # RightHandRing1 -> RightHandRing2
            (21, 22),  # RightHandRing2 -> RightHandRing3
            (22, 23),  # RightHandRing3 -> RightHandRing4
            (23, 24),  # RightHandRing4 -> RightHandRing4_Nub
            (19, 25),  # RightHandRing -> RightHandPinky
            (25, 26),  # RightHandPinky -> RightHandPinky1
            (26, 27),  # RightHandPinky1 -> RightHandPinky2
            (27, 28),  # RightHandPinky2 -> RightHandPinky3
            (28, 29),  # RightHandPinky3 -> RightHandPinky4
            (29, 30),  # RightHandPinky4 -> RightHandPinky4_Nub
            (13, 31),  # RightHand -> RightHandIndex
            (31, 32),  # RightHandIndex -> RightHandIndex1
            (32, 33),  # RightHandIndex1 -> RightHandIndex2
            (33, 34),  # RightHandIndex2 -> RightHandIndex3
            (34, 35),  # RightHandIndex3 -> RightHandIndex4
            (35, 36),  # RightHandIndex4 -> RightHandIndex4_Nub
            (31, 37),  # RightHandIndex -> RightHandThumb1
            (37, 38),  # RightHandThumb1 -> RightHandThumb2
            (38, 39),  # RightHandThumb2 -> RightHandThumb3
            (39, 40),  # RightHandThumb3 -> RightHandThumb4
            (40, 41),  # RightHandThumb4 -> RightHandThumb4_Nub
            (4, 42),  # Spine3 -> LeftShoulder
            (42, 43),  # LeftShoulder -> LeftArm
            (43, 44),  # LeftArm -> LeftForeArm
            (44, 45),  # LeftForeArm -> LeftHand
            (45, 46),  # LeftHand -> LeftHandMiddle1
            (46, 47),  # LeftHandMiddle1 -> LeftHandMiddle2
            (47, 48),  # LeftHandMiddle2 -> LeftHandMiddle3
            (48, 49),  # LeftHandMiddle3 -> LeftHandMiddle4
            (49, 50),  # LeftHandMiddle4 -> LeftHandMiddle4_Nub
            (45, 51),  # LeftHand -> LeftHandRing
            (51, 52),  # LeftHandRing -> LeftHandRing1
            (52, 53),  # LeftHandRing1 -> LeftHandRing2
            (53, 54),  # LeftHandRing2 -> LeftHandRing3
            (54, 55),  # LeftHandRing3 -> LeftHandRing4
            (55, 56),  # LeftHandRing4 -> LeftHandRing4_Nub
            (51, 57),  # LeftHandRing -> LeftHandPinky
            (57, 58),  # LeftHandPinky -> LeftHandPinky1
            (58, 59),  # LeftHandPinky1 -> LeftHandPinky2
            (59, 60),  # LeftHandPinky2 -> LeftHandPinky3
            (60, 61),  # LeftHandPinky3 -> LeftHandPinky4
            (61, 62),  # LeftHandPinky4 -> LeftHandPinky4_Nub
            (45, 63),  # LeftHand -> LeftHandIndex
            (63, 64),  # LeftHandIndex -> LeftHandIndex1
            (64, 65),  # LeftHandIndex1 -> LeftHandIndex2
            (65, 66),  # LeftHandIndex2 -> LeftHandIndex3
            (66, 67),  # LeftHandIndex3 -> LeftHandIndex4
            (67, 68),  # LeftHandIndex4 -> LeftHandIndex4_Nub
            (63, 69),  # LeftHandIndex -> LeftHandThumb1
            (69, 70),  # LeftHandThumb1 -> LeftHandThumb2
            (70, 71),  # LeftHandThumb2 -> LeftHandThumb3
            (71, 72),  # LeftHandThumb3 -> LeftHandThumb4
            (72, 73),  # LeftHandThumb4 -> LeftHandThumb4_Nub
            (0, 74),  # Hips -> RightUpLeg
            (74, 75),  # RightUpLeg -> RightLeg
            (75, 76),  # RightLeg -> RightFoot
            (76, 77),  # RightFoot -> RightForeFoot
            (77, 78),  # RightForeFoot -> RightToeBase
            (78, 79),  # RightToeBase -> RightToeBaseEnd
            (79, 80),  # RightToeBaseEnd -> RightToeBaseEnd_Nub
            (0, 81),  # Hips -> LeftUpLeg
            (81, 82),  # LeftUpLeg -> LeftLeg
            (82, 83),  # LeftLeg -> LeftFoot
            (83, 84),  # LeftFoot -> LeftForeFoot
            (84, 85),  # LeftForeFoot -> LeftToeBase
            (85, 86),  # LeftToeBase -> LeftToeBaseEnd
            (86, 87),  # LeftToeBaseEnd -> LeftToeBaseEnd_Nub
        ]

        # Tối ưu: Tách thành 2 Tensor Parent và Child đưa lên GPU ngay lập tức
        parents = [p for p, c in raw_bone_pairs]
        children = [c for p, c in raw_bone_pairs]
        
        # Lưu vào self để dùng lại sau này
        self.bone_parent_indices = torch.LongTensor(parents).to(self.device)
        self.bone_child_indices = torch.LongTensor(children).to(self.device)

        # 2. SKELETON TREE (Dùng cho Forward Kinematics)
        # Copy nguyên list skeleton_tree dài ngoằng vào đây
        raw_skeleton_tree = [
            (0, 1, [0.0, 6.269896, -2.264934]),  # Hips -> Spine
            (1, 2, [0.0, 12.478628, -2.20032]),  # Spine -> Spine1
            (2, 3, [0.0, 12.622911, -1.104362]),  # Spine1 -> Spine2
            (3, 4, [0.0, 12.671129, 0.0]),  # Spine2 -> Spine3
            (4, 5, [0.0, 16.291454, 1.629145]),  # Spine3 -> Neck
            (5, 6, [0.0, 3.456791, 0.30243]),  # Neck -> Neck1
            (6, 7, [0.0, 3.417274, 0.602559]),  # Neck1 -> Head
            (7, 8, [0.0, 9.72773, -0.0]),  # Head -> HeadEnd
            (8, 9, [0.0, 9.727722, -0.0]),  # HeadEnd -> HeadEnd_Nub
            (4, 10, [0.0, 11.636753, 5.87917]),  # Spine3 -> RightShoulder
            (10, 11, [-19.553394, 8e-06, 0.0]),  # RightShoulder -> RightArm
            (11, 12, [-30.623638, 1.1e-05, 0.0]),  # RightArm -> RightForeArm
            (12, 13, [-25.458359, 8e-06, 0.0]),  # RightForeArm -> RightHand
            (13, 14, [-9.328308, 4e-06, 0.0]),  # RightHand -> RightHandMiddle1
            (14, 15, [-4.931488, 0.0, 0.0]),  # RightHandMiddle1 -> RightHandMiddle2
            (15, 16, [-3.177132, 0.0, 0.0]),  # RightHandMiddle2 -> RightHandMiddle3
            (16, 17, [-1.92765, 0.0, 0.0]),  # RightHandMiddle3 -> RightHandMiddle4
            (17, 18, [-1.927643, 0.0, 0.0]),  # RightHandMiddle4 -> RightHandMiddle4_Nub
            (13, 19, [-0.25, -0.25, -0.855603]),  # RightHand -> RightHandRing
            (19, 20, [-8.228668, 4e-06, -0.742586]),  # RightHandRing -> RightHandRing1
            (20, 21, [-4.579102, 0.0, -0.413236]),  # RightHandRing1 -> RightHandRing2
            (21, 22, [-3.089668, 0.0, -0.278823]),  # RightHandRing2 -> RightHandRing3
            (22, 23, [-1.908813, 0.0, -0.172259]),  # RightHandRing3 -> RightHandRing4
            (23, 24, [-1.908813, 0.0, -0.172258]),  # RightHandRing4 -> RightHandRing4_Nub
            (19, 25, [-0.172089, -0.25, -0.87461]),  # RightHandRing -> RightHandPinky
            (25, 26, [-6.812508, 4e-06, -1.523409]),  # RightHandPinky -> RightHandPinky1
            (26, 27, [-3.617294, 0.0, -0.808899]),  # RightHandPinky1 -> RightHandPinky2
            (27, 28, [-2.311783, 0.0, -0.516959]),  # RightHandPinky2 -> RightHandPinky3
            (28, 29, [-1.725502, 0.0, -0.385855]),  # RightHandPinky3 -> RightHandPinky4
            (29, 30, [-1.725502, 0.0, -0.385856]),  # RightHandPinky4 -> RightHandPinky4_Nub
            (13, 31, [-0.25, -0.25, 0.855603]),  # RightHand -> RightHandIndex
            (31, 32, [-9.013367, 4e-06, 0.8134]),  # RightHandIndex -> RightHandIndex1
            (32, 33, [-4.737785, 0.0, 0.427556]),  # RightHandIndex1 -> RightHandIndex2
            (33, 34, [-2.835075, 0.0, 0.255847]),  # RightHandIndex2 -> RightHandIndex3
            (34, 35, [-1.745514, 0.0, 0.157522]),  # RightHandIndex3 -> RightHandIndex4
            (35, 36, [-1.745514, 0.0, 0.157522]),  # RightHandIndex4 -> RightHandIndex4_Nub
            (31, 37, [-0.172089, -0.75, 0.87461]),  # RightHandIndex -> RightHandThumb1
            (37, 38, [-5.4757, 0.845421, 2.271264]),  # RightHandThumb1 -> RightHandThumb2
            (38, 39, [-3.582893, 0.553185, 1.486152]),  # RightHandThumb2 -> RightHandThumb3
            (39, 40, [-2.19529, 0.338943, 0.910584]),  # RightHandThumb3 -> RightHandThumb4
            (40, 41, [-2.19529, 0.338943, 0.910584]),  # RightHandThumb4 -> RightHandThumb4_Nub
            (4, 42, [0.0, 11.636753, 5.87917]),  # Spine3 -> LeftShoulder
            (42, 43, [19.553394, 8e-06, 0.0]),  # LeftShoulder -> LeftArm
            (43, 44, [30.623638, 1.1e-05, 0.0]),  # LeftArm -> LeftForeArm
            (44, 45, [25.458359, 8e-06, 0.0]),  # LeftForeArm -> LeftHand
            (45, 46, [9.327454, 4e-06, 0.0]),  # LeftHand -> LeftHandMiddle1
            (46, 47, [4.935944, 0.0, 0.0]),  # LeftHandMiddle1 -> LeftHandMiddle2
            (47, 48, [3.187286, 0.0, 0.0]),  # LeftHandMiddle2 -> LeftHandMiddle3
            (48, 49, [1.919037, 0.0, 0.0]),  # LeftHandMiddle3 -> LeftHandMiddle4
            (49, 50, [1.919052, 0.0, 0.0]),  # LeftHandMiddle4 -> LeftHandMiddle4_Nub
            (45, 51, [0.25, -0.25, -0.911864]),  # LeftHand -> LeftHandRing
            (51, 52, [8.228249, 4e-06, -0.742548]),  # LeftHandRing -> LeftHandRing1
            (52, 53, [4.570602, 0.0, -0.412469]),  # LeftHandRing1 -> LeftHandRing2
            (53, 54, [3.097679, 0.0, -0.279546]),  # LeftHandRing2 -> LeftHandRing3
            (54, 55, [1.900299, 0.0, -0.17149]),  # LeftHandRing3 -> LeftHandRing4
            (55, 56, [1.900299, 0.0, -0.17149]),  # LeftHandRing4 -> LeftHandRing4_Nub
            (51, 57, [0.16703, -0.25, -0.930643]),  # LeftHandRing -> LeftHandPinky
            (57, 58, [6.794594, 4e-06, -1.519404]),  # LeftHandPinky -> LeftHandPinky1
            (58, 59, [3.623344, 0.0, -0.810251]),  # LeftHandPinky1 -> LeftHandPinky2
            (59, 60, [2.307434, 0.0, -0.515988]),  # LeftHandPinky2 -> LeftHandPinky3
            (60, 61, [1.717804, 0.0, -0.384134]),  # LeftHandPinky3 -> LeftHandPinky4
            (61, 62, [1.717804, 0.0, -0.384135]),  # LeftHandPinky4 -> LeftHandPinky4_Nub
            (45, 63, [0.25, -0.25, 0.911864]),  # LeftHand -> LeftHandIndex
            (63, 64, [8.99826, 4e-06, 0.812038]),  # LeftHandIndex -> LeftHandIndex1
            (64, 65, [4.745354, 0.0, 0.428239]),  # LeftHandIndex1 -> LeftHandIndex2
            (65, 66, [2.836342, 0.0, 0.255961]),  # LeftHandIndex2 -> LeftHandIndex3
            (66, 67, [1.737732, 0.0, 0.15682]),  # LeftHandIndex3 -> LeftHandIndex4
            (67, 68, [1.737732, 0.0, 0.156819]),  # LeftHandIndex4 -> LeftHandIndex4_Nub
            (63, 69, [0.16703, -0.75, 0.930643]),  # LeftHandIndex -> LeftHandThumb1
            (69, 70, [5.434509, 0.839062, 2.254179]),  # LeftHandThumb1 -> LeftHandThumb2
            (70, 71, [3.593353, 0.554794, 1.490485]),  # LeftHandThumb2 -> LeftHandThumb3
            (71, 72, [2.185493, 0.337429, 0.906521]),  # LeftHandThumb3 -> LeftHandThumb4
            (72, 73, [2.185501, 0.337429, 0.906523]),  # LeftHandThumb4 -> LeftHandThumb4_Nub
            (0, 74, [-8.246678, 0.0, 0.0]),  # Hips -> RightUpLeg
            (74, 75, [0.0, -42.827576, 0.0]),  # RightUpLeg -> RightLeg
            (75, 76, [0.0, -43.165855, 0.0]),  # RightLeg -> RightFoot
            (76, 77, [0.0, -2.559708, 0.0]),  # RightFoot -> RightForeFoot
            (77, 78, [0.0, 0.0, 10.024612]),  # RightForeFoot -> RightToeBase
            (78, 79, [0.0, 0.0, 14.750254]),  # RightToeBase -> RightToeBaseEnd
            (79, 80, [0.0, 0.0, 14.75025]),  # RightToeBaseEnd -> RightToeBaseEnd_Nub
            (0, 81, [8.246678, 0.0, 0.0]),  # Hips -> LeftUpLeg
            (81, 82, [0.0, -42.827576, 0.0]),  # LeftUpLeg -> LeftLeg
            (82, 83, [0.0, -43.165855, 0.0]),  # LeftLeg -> LeftFoot
            (83, 84, [0.0, -2.559708, 0.0]),  # LeftFoot -> LeftForeFoot
            (84, 85, [0.0, 0.0, 10.024612]),  # LeftForeFoot -> LeftToeBase
            (85, 86, [0.0, 0.0, 14.330561]),  # LeftToeBase -> LeftToeBaseEnd
            (86, 87, [0.0, 0.0, 14.330564]),  # LeftToeBaseEnd -> LeftToeBaseEnd_Nub
        ]

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
        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"text": caption, "length": cur_len}
        )

        self.real_noise = output['target']
        self.fake_noise = output['pred']
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
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        
        loss_vel = None
        loss_acc = None
        loss_geom = None
        
        # THÊM: Velocity Loss (giảm jitter)
        if self.opt.use_velocity_loss:
            vel_gt = self.motions[:, 1:] - self.motions[:, :-1]
            vel_pred = self.fake_noise[:, 1:] - self.fake_noise[:, :-1]
            loss_vel = self.mse_criterion(vel_pred, vel_gt).mean()
            loss_mot_rec = loss_mot_rec + self.opt.velocity_weight * loss_vel
        
        # THÊM: Acceleration Loss (motion smoother)
        if self.opt.use_acceleration_loss and loss_vel is not None:
            acc_gt = vel_gt[:, 1:] - vel_gt[:, :-1]
            acc_pred = vel_pred[:, 1:] - vel_pred[:, :-1]
            loss_acc = self.mse_criterion(acc_pred, acc_gt).mean()
            loss_mot_rec = loss_mot_rec + self.opt.acceleration_weight * loss_acc
        
        # THÊM: Geometric Loss (Bone length consistency)
        if self.opt.use_geometric_loss:
            # Giả sử motion là axis-angle format (T, 264) = (T, 55*3)
            # Tính forward kinematics để lấy joint positions
            loss_geom = self.compute_geometric_loss(
                self.fake_noise, 
                self.motions
            )
            loss_mot_rec = loss_mot_rec + self.opt.geometric_weight * loss_geom
        
        self.loss_mot_rec = loss_mot_rec
        loss_logs = OrderedDict({
            'loss_mot_rec': self.loss_mot_rec.item()
        })
        
        if self.opt.use_velocity_loss:
            loss_logs['loss_vel'] = loss_vel.item()
        if self.opt.use_acceleration_loss:
            loss_logs['loss_acc'] = loss_acc.item()
        if self.opt.use_geometric_loss:
            loss_logs['loss_geom'] = loss_geom.item()
        
        return loss_logs

    def compute_geometric_loss(self, motion_pred, motion_gt):
        # 1. Import hàm chuyển đổi
        import sys
        sys.path.insert(0, './datasets')
        from emage_utils.rotation_conversions import axis_angle_to_rotation_6d

        # 2. Reshape về (B, T, J, 3) 
        # Model của bạn vẫn đang output Axis-Angle (3 channel) nên reshape về 3 là đúng
        n_joints = 88 # Hãy đảm bảo số này khớp với dataset BEAT của bạn
        
        motion_pred_reshaped = motion_pred.reshape(motion_pred.shape[0], motion_pred.shape[1], n_joints, 3)
        motion_gt_reshaped = motion_gt.reshape(motion_gt.shape[0], motion_gt.shape[1], n_joints, 3)
        
        # Chuyển từ Axis-Angle (3D) sang 6D Rotation để thỏa mãn yêu cầu của forward_kinematics mới
        motion_pred_6d = axis_angle_to_rotation_6d(motion_pred_reshaped)
        motion_gt_6d = axis_angle_to_rotation_6d(motion_gt_reshaped)
        
        # 3. Tính Positions (Giờ đầu vào đã là 6D -> Hợp lệ)
        positions_pred = self.forward_kinematics(motion_pred_6d, motion_raw=motion_pred) 
        positions_gt = self.forward_kinematics(motion_gt_6d, motion_raw=motion_gt)

        # --- FK LOSS (Global Position Loss) ---
        fk_loss = None 
        if getattr(self.opt, 'use_fk_loss', False): 
            # Tính MSE positions
            fk_loss = torch.mean((positions_pred - positions_gt) ** 2)
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
        
        # Loss 1: Direction (Cosine similarity)
        bone_pred_norm = bone_pred / bone_len_pred
        bone_gt_norm = bone_gt / bone_len_gt
        dir_loss = (1 - (bone_pred_norm * bone_gt_norm).sum(dim=-1)).mean()
        
        # Loss 2: Length consistency
        len_loss = torch.abs(bone_len_pred - bone_len_gt).mean()
        
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
        import sys
        sys.path.insert(0, './datasets')
        
        # 1. Thay đổi import
        from emage_utils.rotation_conversions import rotation_6d_to_matrix
        
        B, T, n_joints, dims = rot6d.shape
        
        # Kiểm tra nhanh: Nếu input là 6D thì dims phải bằng 6
        assert dims == 6, f"Input cho hàm này phải là 6D, nhưng nhận được {dims}D"
        
        # 2. Chuyển đổi từ 6D sang Matrix (3x3)
        # Input: (B, T, J, 6) -> Output: (B, T, J, 3, 3)
        local_rot_mats = rotation_6d_to_matrix(rot6d)
                
        # Tạo chỗ chứa Global Rotations và Positions
        global_rot_mats = torch.zeros_like(local_rot_mats) 
        positions = torch.zeros(B, T, n_joints, 3, device=rot6d.device)
        
        # Root (Hips)
        if motion_raw is not None:
            # Nếu có raw features, khôi phục vị trí thực tế
            root_pos = self.recover_root_translation(motion_raw) # (B, T, 1, 3)
            positions[:, :, 0] = root_pos.squeeze(2)
        else:
            # Mặc định tại (0,0,0) nếu không có raw data
            positions[:, :, 0] = 0

        global_rot_mats[:, :, 0] = local_rot_mats[:, :, 0]
        
        # Duyệt qua cây xương đã tối ưu (Parent -> Child)
        for parent, child, offset in self.optimized_skeleton_tree:
            # Tính Global Rotation của Child: Parent_Global @ Child_Local
            global_rot_mats[:, :, child] = torch.matmul(
                global_rot_mats[:, :, parent], 
                local_rot_mats[:, :, child]
            )
            
            # Tính Position của Child: Pos_Parent + (Parent_Global @ Offset)
            # Offset shape (1, 1, 3, 1) xoay theo hướng của Parent
            rot_offset = torch.matmul(global_rot_mats[:, :, parent], offset).squeeze(-1)
            
            positions[:, :, child] = positions[:, :, parent] + rot_offset
            
        return positions
    
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
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        if 'scheduler' in checkpoint and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        self.to(self.device)
        self.opt_encoder = optim.AdamW(self.encoder.parameters(), lr=self.opt.lr)
        
        # Build dataloader TRƯỚC khi tạo scheduler
        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True)
        
        # Tính đúng T_max
        total_steps = len(train_loader) * self.opt.num_epochs
        warmup_steps = min(500, len(train_loader))  # Warmup 1 epoch hoặc 500 steps
        
        warmup_scheduler = LinearLR(
            self.opt_encoder, 
            start_factor=0.1, 
            total_iters=warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            self.opt_encoder, 
            T_max=total_steps - warmup_steps,
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
        scaler = GradScaler()
        logs = OrderedDict()
        
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            
            for i, batch_data in enumerate(epoch_bar):
                if batch_data is None:
                    continue
                
                #  Forward pass với autocast
                with autocast():
                    self.forward(batch_data)
                    loss_logs = self.backward_G()
                
                #  Backward pass với scaler
                self.zero_grad([self.opt_encoder])
                scaler.scale(self.loss_mot_rec).backward()
                scaler.unscale_(self.opt_encoder)
                self.clip_norm([self.encoder])
                scaler.step(self.opt_encoder)
                scaler.update()  #  Reset scaler sau mỗi step
                
                # Update EMA
                self.update_ema()
                
                # Update scheduler sau mỗi step (step-wise, không epoch-wise)
                self.scheduler.step()
                
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
            
            # Save checkpoint
            if epoch % self.opt.save_every_e == 0:
                self.save(
                    pjoin(self.opt.model_dir, 'ckpt_e%03d.tar' % epoch), 
                    epoch, 
                    total_it=it
                )
            
            print(f" Epoch {epoch} completed")