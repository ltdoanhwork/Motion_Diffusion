import os
import sys
ROOT_DIR = "/home/serverai/ltdoanh/Motion_Diffusion"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# === FIX LỖI MODULE NOT FOUND: Thêm pymo vào sys.path ===
PYMO_DIR = os.path.join(ROOT_DIR, 'datasets', 'pymo')

if os.path.exists(PYMO_DIR):
    if PYMO_DIR not in sys.path:
        print(f"🔌 Adding pymo to path: {PYMO_DIR}")
        sys.path.insert(0, PYMO_DIR)
else:
    print(f"⚠️ Warning: Pymo dir not found at {PYMO_DIR}")

try:
    import pymo
    from pymo.preprocessing import *
    print("✅ Pymo module loaded successfully!")
except ImportError as e:
    print(f"❌ Failed to import pymo: {e}")

import torch
import numpy as np
import joblib
from torch.utils.data import DataLoader, Dataset

from models import MotionTransformer
from trainers import DDPMTrainer
from models.vq.model import RVQVAE 
from tools.evaluation import (
    evaluate_matching_score, 
    evaluate_fid, 
    evaluate_diversity, 
    evaluate_multimodality
)
from datasets.evaluator import EvaluatorModelWrapper 
from datasets.dataset import Beat2MotionDataset

# ==========================================
# 2. Cấu hình - GIỐNG NHƯ INFERENCE
# ==========================================
class EvalConfig:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_train = False
        self.schedule_sampler = 'uniform'
        
        # Model Config (GIỐNG INFERENCE)
        self.input_feats = 512     
        self.num_frames = 45       
        self.num_layers = 8
        self.latent_dim = 512
        self.ff_size = 1024
        self.num_heads = 8
        self.dropout = 0.1
        self.activation = "gelu"
        self.dataset_name = 'beat' 
        self.do_denoise = True
        self.noise_schedule = 'cosine'
        self.diffusion_steps = 1000
        self.no_clip = False
        self.no_eff = False
        
        # Dataset Config
        self.max_motion_length = 196
        self.unit_length = 4
        self.motion_rep = 'position'
        self.joints_num = 22
        self.feat_bias = 5

        # Path Config
        self.data_root = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT_numpy"
        self.motion_dir = os.path.join(self.data_root, 'npy')
        self.text_dir = os.path.join(self.data_root, 'txt')
        self.meta_dir = self.data_root
        self.evaluator_path = "/home/serverai/ltdoanh/Motion_Diffusion/checkpoints/t2m/text_mot_match/model/finest.tar"
        self.result_dir = "/home/serverai/ltdoanh/Motion_Diffusion/results"
        self.save_motion_dir = os.path.join(self.result_dir, "generated_motions")
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.save_motion_dir, exist_ok=True)

        # Evaluator params
        self.dim_movement_enc_hidden = 512
        self.dim_movement_latent = 512
        self.dim_motion_hidden = 1024
        self.dim_text_hidden = 1024
        self.dim_coemb_hidden = 512
        self.dim_word = 300
        self.dim_pos_ohot = 15
        self.max_text_len = 20
        self.dim_pose = 263

class VQArgs:
    def __init__(self):
        self.num_quantizers = 1 
        self.shared_codebook = False
        self.quantize_dropout_prob = 0.0
        self.mu = 0.99 

opt = EvalConfig()
vq_args = VQArgs()

# ==========================================
# 3. Load Models - GIỐNG NHƯ INFERENCE
# ==========================================
print("🔄 Loading Trained Models...")
ckpt_path = "/home/serverai/ltdoanh/Motion_Diffusion/checkpoints/beat/vq_diffusion/model/best.pt"
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Tự động phát hiện num_quantizers
state_dict = checkpoint['model_state_dict']
max_layer_idx = 0
for k in state_dict.keys():
    if "vqvae.quantizer.layers." in k:
        try:
            parts = k.split('.')
            layer_idx = int(parts[parts.index('layers') + 1])
            if layer_idx > max_layer_idx:
                max_layer_idx = layer_idx
        except:
            pass
vq_args.num_quantizers = max_layer_idx + 1
print(f"🔍 Detected num_quantizers: {vq_args.num_quantizers}")

# Tách weights
trans_dict = {}
vqvae_dict = {}
for k, v in state_dict.items():
    if k.startswith('transformer.'):
        trans_dict[k[12:]] = v  
    elif k.startswith('vqvae.'):
        vqvae_dict[k[6:]] = v

# A. Motion Transformer
encoder = MotionTransformer(
    input_feats=opt.input_feats, num_frames=opt.num_frames, num_layers=opt.num_layers,
    latent_dim=opt.latent_dim, num_heads=opt.num_heads, ff_size=opt.ff_size,
    no_clip=opt.no_clip, no_eff=opt.no_eff
)
encoder.load_state_dict(trans_dict, strict=True)
print("✅ MotionTransformer Loaded!")

# B. RVQVAE
vqvae_model = RVQVAE(
    args=vq_args, input_width=264, nb_code=512, code_dim=512, output_emb_width=512, 
    down_t=3, stride_t=2, width=512, depth=3, dilation_growth_rate=3, activation='relu', norm=None
)
vqvae_model.load_state_dict(vqvae_dict, strict=True)
print("✅ RVQVAE Loaded!")

encoder.to(opt.device).eval()
vqvae_model.to(opt.device).eval()

# C. Trainer
trainer = DDPMTrainer(opt, encoder)

# Load Mean/Std từ Pipeline
pipeline_path = "/home/serverai/ltdoanh/Motion_Diffusion/global_pipeline.pkl"
print(f"🔧 Overriding Mean/Std from pipeline: {pipeline_path}")

pipeline = joblib.load(pipeline_path)
try:
    if 'stdscale' in pipeline.named_steps:
        scaler = pipeline.named_steps['stdscale']
    else:
        scaler = pipeline.steps[-1][1]
        
    trainer.mean = scaler.data_mean_
    trainer.std = scaler.data_std_
    
    print(f"   ✅ Loaded Mean shape: {trainer.mean.shape}")
    print(f"   ✅ Loaded Std shape: {trainer.std.shape}")
    
except Exception as e:
    print(f"❌ Lỗi load pipeline: {e}")
    trainer.mean = checkpoint['mean']
    trainer.std = checkpoint['std']

print("✅ Trainer Ready!")

print("⚖️ Loading Evaluator Model...")
eval_wrapper = EvaluatorModelWrapper(opt)
print("   ✅ Evaluator Loaded!")

# ==========================================
# 4. Load Dataset
# ==========================================
print("\n📋 Creating Validation Dataset...")

val_file = os.path.join(opt.data_root, 'val.txt')
if not os.path.exists(val_file):
    val_file = os.path.join(opt.data_root, 'test.txt')
    
if not os.path.exists(val_file):
    raise FileNotFoundError(f"Cannot find validation file in {opt.data_root}")

print(f"   Using split file: {val_file}")

with open(val_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]
    print(f"   Found {len(lines)} samples in split file")

# ==========================================
# SimpleEvalDataset
# ==========================================
class SimpleEvalDataset(Dataset):
    def __init__(self, clip_ids, motion_dir, text_dir, mean, std, max_len=196):
        self.data = []
        self.mean = mean
        self.std = std
        self.max_len = max_len
        
        print(f"\n   Loading dataset from {len(clip_ids)} IDs...")
        
        for i, cid in enumerate(clip_ids):
            variants = [
                cid,
                cid.split('/')[-1],
                cid.replace('/', '_'),
            ]
            
            found = False
            for variant in variants:
                motion_path = os.path.join(motion_dir, f'{variant}.npy')
                txt_path = os.path.join(text_dir, f'{variant}.txt')
                
                if os.path.exists(motion_path) and os.path.exists(txt_path):
                    try:
                        motion = np.load(motion_path)
                        if motion.ndim == 3 and motion.shape[0] == 1:
                            motion = motion.squeeze(0)
                        
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            caption = f.readline().strip()
                            if '#' in caption:
                                caption = caption.split('#')[0].strip()
                        
                        if len(caption) == 0:
                            continue
                        
                        self.data.append({
                            'id': cid,
                            'caption': caption,
                            'motion': motion,
                            'length': min(len(motion), max_len)
                        })
                        
                        found = True
                        if i < 3:
                            print(f"      ✅ Loaded #{i+1}: {variant}")
                            print(f"         Caption: {caption[:50]}...")
                            print(f"         Motion shape: {motion.shape}")
                        break
                        
                    except Exception as e:
                        continue
            
            if not found and i < 3:
                print(f"      ❌ Not found: {cid}")
        
        print(f"\n   ✅ Successfully loaded {len(self.data)} / {len(clip_ids)} samples")
        
        if len(self.data) == 0:
            raise ValueError("No valid samples found!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion'].copy()
        m_len = item['length']
        
        if len(motion) > self.max_len:
            motion = motion[:self.max_len]
        elif len(motion) < self.max_len:
            pad = np.zeros((self.max_len - len(motion), motion.shape[1]))
            motion = np.concatenate([motion, pad], axis=0)
        
        motion = (motion - self.mean) / self.std
        
        return item['caption'], motion.astype(np.float32), m_len

val_dataset = SimpleEvalDataset(lines, opt.motion_dir, opt.text_dir, trainer.mean, trainer.std)

def collate_fn(batch):
    captions, motions, m_lens = zip(*batch)
    motions = torch.from_numpy(np.stack(motions))
    m_lens = torch.tensor(m_lens, dtype=torch.long)
    
    batch_size = len(captions)
    word_embeddings = torch.zeros(batch_size, 1, 300)
    pos_one_hots = torch.zeros(batch_size, 1, 15)
    sent_lens = torch.tensor([len(c.split()) for c in captions])
    tokens = ['_'.join(c.split()[:5]) for c in captions]
    
    return word_embeddings, pos_one_hots, captions, sent_lens, motions, m_lens, tokens

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, 
                         drop_last=False, collate_fn=collate_fn)

print(f"   Created DataLoader with {len(val_loader)} batches")

# ==========================================
# 5. Generation + Lưu motions
# ==========================================
class GeneratedMotionDataset(Dataset):
    def __init__(self, motions_data, opt):
        self.data = motions_data
        self.opt = opt
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        motion = item['motion'].float()
        m_len = item['length']
        
        if motion.shape[0] < self.opt.max_motion_length:
            motion = torch.cat([
                motion,
                torch.zeros(self.opt.max_motion_length - motion.shape[0], motion.shape[1])
            ], dim=0)
        
        return (
            torch.zeros(1, 300).float(),
            torch.zeros(1, 15).float(),
            item['caption'],
            len(item['caption'].split()),
            motion,
            torch.tensor(m_len, dtype=torch.long),
            '_'.join(item['caption'].split()[:5])
        )

print("\n🚀 Generating Regular Motions (INFERENCE MODE)...")
generated_motions = []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        word_embeddings, pos_one_hots, captions, sent_len, real_motion, m_lens, tokens = batch
        
        print(f"Batch {i+1}/{len(val_loader)}: {len(captions)} samples")
        
        for j, caption in enumerate(captions):
            try:
                m_len_latent = torch.LongTensor([(m_lens[j].item() // 8)]).to(opt.device)
                
                pred_latent_list = trainer.generate([caption], m_len_latent, dim_pose=512)
                pred_latent = pred_latent_list[0]
                
                if pred_latent.dim() == 2:
                    pred_latent = pred_latent.unsqueeze(0)
                
                latent_input = pred_latent.permute(0, 2, 1)
                decoded_motion = vqvae_model.decoder(latent_input)
                motion = vqvae_model.postprocess(decoded_motion)

                if motion.shape[1] == 264 and motion.shape[2] != 264:
                    print(f"   🔄 Permuting motion from {motion.shape} to (B, T, C)")
                    motion = motion.permute(0, 2, 1)
                
                mean_tensor = torch.from_numpy(trainer.mean).to(motion.device)
                std_tensor = torch.from_numpy(trainer.std).to(motion.device)
                motion = motion * std_tensor.unsqueeze(0).unsqueeze(0) + mean_tensor.unsqueeze(0).unsqueeze(0)
                
                # === LƯU MOTION ===
                motion_np = motion[0].cpu().numpy()
                save_path = os.path.join(opt.save_motion_dir, f"motion_{i}_{j}.npy")
                np.save(save_path, motion_np)
                
                generated_motions.append({
                    'motion': motion[0].cpu(),
                    'length': m_lens[j].item(),
                    'caption': caption,
                    'save_path': save_path
                })
                
                print(f"  ✅ Generated {j+1}/{len(captions)} - Saved to {save_path}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

print(f"\n✅ Generated {len(generated_motions)} motions!")
print(f"💾 Motions saved to: {opt.save_motion_dir}")

if len(generated_motions) == 0:
    raise ValueError("No motions generated!")

gen_dataset = GeneratedMotionDataset(generated_motions, opt)
gen_loader = DataLoader(gen_dataset, batch_size=32, shuffle=False, num_workers=1, drop_last=True)

# ==========================================
# 6. MultiModality Dataset
# ==========================================
class MMGeneratedDataset(Dataset):
    def __init__(self, mm_motions_list, opt):
        self.mm_motions = mm_motions_list
        self.opt = opt
        
    def __len__(self):
        return len(self.mm_motions)
    
    def __getitem__(self, item):
        mm_batch = self.mm_motions[item]
        motions = []
        m_lens = []
        
        for motion, length in mm_batch:
            motion = motion.float()
            if motion.shape[0] < self.opt.max_motion_length:
                motion = torch.cat([
                    motion,
                    torch.zeros(self.opt.max_motion_length - motion.shape[0], motion.shape[1])
                ], dim=0)
            motions.append(motion.unsqueeze(0))
            m_lens.append(length)
        
        motions = torch.cat(motions, dim=0)
        m_lens = torch.tensor(m_lens, dtype=torch.long)
        
        sort_idx = torch.argsort(m_lens, descending=True)
        return motions[sort_idx], m_lens[sort_idx]

mm_num_samples = min(10, len(val_dataset))
mm_num_repeats = 10

print(f"\n🔄 Generating MultiModality Motions ({mm_num_samples} × {mm_num_repeats})...")

mm_generated_motions = []
mm_indices = np.random.choice(len(val_dataset), mm_num_samples, replace=False)

with torch.no_grad():
    for idx, mm_idx in enumerate(mm_indices):
        caption, _, m_len = val_dataset[mm_idx]
        
        mm_batch = []
        m_len_latent = torch.LongTensor([m_len // 8]).to(opt.device)
        
        for rep in range(mm_num_repeats):
            pred_latent = trainer.generate([caption], m_len_latent, dim_pose=512)[0]
            if pred_latent.dim() == 2:
                pred_latent = pred_latent.unsqueeze(0)
            
            latent_input = pred_latent.permute(0, 2, 1)
            decoded = vqvae_model.decoder(latent_input)
            motion = vqvae_model.postprocess(decoded)
            
            if motion.shape[1] == 264 and motion.shape[2] != 264:
                motion = motion.permute(0, 2, 1)
            
            mean_tensor = torch.from_numpy(trainer.mean).to(motion.device)
            std_tensor = torch.from_numpy(trainer.std).to(motion.device)
            motion = motion * std_tensor + mean_tensor
            
            # === LƯU MM MOTION ===
            motion_np = motion[0].cpu().numpy()
            mm_save_path = os.path.join(opt.save_motion_dir, f"mm_motion_{idx}_rep{rep}.npy")
            np.save(mm_save_path, motion_np)
            
            mm_batch.append((motion[0].cpu(), m_len))
        
        mm_generated_motions.append(mm_batch)
        print(f"\rMM Progress: {idx+1}/{mm_num_samples}", end='')

print("\n✅ MM Generation Done!")

mm_dataset = MMGeneratedDataset(mm_generated_motions, opt)
mm_loader = DataLoader(mm_dataset, batch_size=1, num_workers=1)

# ==========================================
# 7. Evaluation - FIX LỖI
# ==========================================
log_file = os.path.join(opt.result_dir, 'evaluation_log.txt')

print("\n📊 Computing Metrics...")

with open(log_file, 'w') as f:
    print("========== Evaluation ==========", file=f)
    
    motion_loaders = {'text2motion': gen_loader, 'ground truth': val_loader}
    
    # FIX: Kiểm tra xem có data không trước khi evaluate
    print("========== Evaluating Matching Score ==========")
    try:
        match_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)
        print(f"✅ Matching Score completed")
    except Exception as e:
        print(f"❌ Matching Score failed: {e}")
        match_score_dict, R_precision_dict, acti_dict = {}, {}, {}
    
    if acti_dict:
        print("========== Evaluating FID ==========")
        try:
            fid_dict = evaluate_fid(val_loader, acti_dict, f)
            print(f"✅ FID completed")
        except Exception as e:
            print(f"❌ FID failed: {e}")
            fid_dict = {}
        
        print("========== Evaluating Diversity ==========")
        try:
            div_dict = evaluate_diversity(acti_dict, f)
            print(f"✅ Diversity completed")
        except Exception as e:
            print(f"❌ Diversity failed: {e}")
            div_dict = {}
    else:
        print("⚠️ Skipping FID and Diversity (no embeddings)")
        fid_dict, div_dict = {}, {}
    
    print("========== Evaluating MultiModality ==========")
    mm_motion_loaders = {'text2motion': mm_loader}
    try:
        mm_dict = evaluate_multimodality(mm_motion_loaders, f)
        print(f"✅ MultiModality completed")
    except Exception as e:
        print(f"❌ MultiModality failed: {e}")
        mm_dict = {}
    
    print("\n========== Results ==========")
    print(f"Matching Score: {match_score_dict}")
    print(f"R-precision: {R_precision_dict}")
    print(f"FID: {fid_dict}")
    print(f"Diversity: {div_dict}")
    print(f"MultiModality: {mm_dict}")
    
    print("\n========== Results ==========", file=f)
    print(f"Matching Score: {match_score_dict}", file=f)
    print(f"R-precision: {R_precision_dict}", file=f)
    print(f"FID: {fid_dict}", file=f)
    print(f"Diversity: {div_dict}", file=f)
    print(f"MultiModality: {mm_dict}", file=f)

print(f"\n🏆 Evaluation completed!")
print(f"   Log file: {log_file}")
print(f"   Generated motions: {opt.save_motion_dir}")
print("\n✅ Done!")