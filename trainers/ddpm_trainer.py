import torch
import torch.nn.functional as F
import random
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist
from tqdm import tqdm
import copy


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

# Import motion losses (optional, will gracefully degrade if not present)
try:
    from models.motion_losses import MotionLossModule
    HAS_MOTION_LOSSES = True
except ImportError:
    HAS_MOTION_LOSSES = False
    print("[WARN] motion_losses module not found, using MSE only")


class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {'shadow': self.shadow, 'decay': self.decay}
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict.get('decay', self.decay)


class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        sampler = 'uniform'
        
        # Use configurable beta schedule (default to cosine for better results)
        beta_scheduler = getattr(args, 'beta_schedule', 'cosine')
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        
        # Prediction type
        prediction_type = getattr(args, 'prediction_type', 'epsilon')
        if prediction_type == 'epsilon':
            model_mean_type = ModelMeanType.EPSILON
        elif prediction_type == 'x_start':
            model_mean_type = ModelMeanType.START_X
        else:
            model_mean_type = ModelMeanType.EPSILON
        
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE
        )
        
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler
        
        # Motion losses
        use_velocity = getattr(args, 'use_velocity_loss', False)
        use_acceleration = getattr(args, 'use_acceleration_loss', False)
        use_geometric = getattr(args, 'use_geometric_loss', False)
        
        if HAS_MOTION_LOSSES and (use_velocity or use_acceleration or use_geometric):
            self.motion_loss = MotionLossModule(
                joints_num=55,
                fps=60.0,
                use_velocity=use_velocity,
                use_acceleration=use_acceleration,
                use_bone_length=use_geometric,
                use_foot_contact=False,
                velocity_weight=getattr(args, 'velocity_weight', 0.5),
                acceleration_weight=getattr(args, 'acceleration_weight', 0.1),
                bone_length_weight=getattr(args, 'geometric_weight', 0.3),
            )
        else:
            self.motion_loss = None
        
        # EMA
        ema_decay = getattr(args, 'ema_decay', 0.0)
        if ema_decay > 0:
            self.ema = EMA(encoder, decay=ema_decay)
        else:
            self.ema = None
        
        # CFG dropout
        self.cfg_dropout = getattr(args, 'cfg_dropout', 0.0)
        
        # Gradient clipping
        self.max_grad_norm = getattr(args, 'max_grad_norm', 1.0)

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list, max_norm=1.0):
        for network in network_list:
            clip_grad_norm_(network.parameters(), max_norm)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        caption, motions, m_lens = batch_data
        motions = motions.detach().to(self.device).float()
        
        # CFG: randomly drop text condition during training
        if self.training and self.cfg_dropout > 0 and not eval_mode:
            caption = list(caption)
            for i in range(len(caption)):
                if random.random() < self.cfg_dropout:
                    caption[i] = ""
            caption = tuple(caption)

        self.caption = caption
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
        self.timesteps = t
        
        # Store for motion loss calculation
        if self.motion_loss is not None:
            with torch.no_grad():
                x_t = self.diffusion.q_sample(x_start, t)
                pred_x0 = self.diffusion._predict_xstart_from_eps(
                    x_t=x_t, t=t, eps=self.fake_noise
                )
            self.pred_motion = pred_x0
            self.gt_motion = motions
        
        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def generate_batch(self, caption, m_lens, dim_pose):
        xf_proj, xf_out = self.encoder.encode_text(caption, self.device)
        
        B = len(caption)
        T = min(m_lens.max(), self.encoder.num_frames)
        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            clip_denoised=False,
            progress=True,
            model_kwargs={
                'xf_proj': xf_proj,
                'xf_out': xf_out,
                'length': m_lens
            })
        return output

    def generate(self, caption, m_lens, dim_pose, batch_size=1024):
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
        # Base diffusion loss
        loss_mot_rec = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
        loss_mot_rec = (loss_mot_rec * self.src_mask).sum() / self.src_mask.sum()
        
        loss_logs = OrderedDict()
        loss_logs['loss_diffusion'] = loss_mot_rec.item()
        
        total_loss = loss_mot_rec
        
        # Motion-specific losses
        if self.motion_loss is not None:
            motion_losses = self.motion_loss(
                self.pred_motion,
                self.gt_motion,
                mask=self.src_mask
            )
            
            if 'velocity' in motion_losses:
                loss_logs['loss_vel'] = motion_losses['velocity'].item()
            if 'acceleration' in motion_losses:
                loss_logs['loss_acc'] = motion_losses['acceleration'].item()
            if 'bone_length' in motion_losses:
                loss_logs['loss_geo'] = motion_losses['bone_length'].item()
            
            total_loss = total_loss + motion_losses['total']
        
        self.loss_mot_rec = total_loss
        loss_logs['loss_total'] = total_loss.item()
        
        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder], self.max_grad_norm)
        self.step([self.opt_encoder])
        
        # EMA update
        if self.ema is not None:
            self.ema.update()

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)
        if self.motion_loss is not None:
            self.motion_loss = self.motion_loss.to(device)

    def train_mode(self):
        self.encoder.train()
        self.training = True

    def eval_mode(self):
        self.encoder.eval()
        self.training = False

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
        
        # Save EMA weights
        if self.ema is not None:
            state['ema'] = self.ema.state_dict()
        
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        
        # Load EMA if available
        if self.ema is not None and 'ema' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema'])
        
        return checkpoint['ep'], checkpoint.get('total_it', 0)

    def train(self, train_dataset):
        self.to(self.device)
        self.opt_encoder = optim.AdamW(
            self.encoder.parameters(), 
            lr=self.opt.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)
        start_time = time.time()

        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True)

        # Print training config
        print(f"\n{'='*60}")
        print(f"🚀 Training with improved settings:")
        print(f"   Beta schedule: {getattr(self.opt, 'beta_schedule', 'cosine')}")
        print(f"   Velocity loss: {getattr(self.opt, 'use_velocity_loss', False)}")
        print(f"   Geometric loss: {getattr(self.opt, 'use_geometric_loss', False)}")
        print(f"   EMA decay: {getattr(self.opt, 'ema_decay', 0.0)}")
        print(f"   CFG dropout: {self.cfg_dropout}")
        print(f"{'='*60}\n")

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()

            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for i, batch_data in enumerate(epoch_bar):
                if batch_data is None:
                    continue
                
                self.forward(batch_data)
                log_dict = self.update()

                for k, v in log_dict.items():
                    logs[k] = logs.get(k, 0) + v

                it += 1

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    epoch_bar.set_postfix({k: f"{v:.4f}" for k, v in mean_loss.items()})
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)


