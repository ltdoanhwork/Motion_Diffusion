# datasets/beat2motion_dataset.py
import os, random, codecs as cs, numpy as np, torch
from os.path import join as pjoin
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm


class Text2MotionDataset(data.Dataset):
    """Dataset for Text2Motion generation task.

    """
    def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False):
        self.opt = opt
        self.max_length = 20
        self.times = times
        self.w_vectorizer = w_vectorizer
        self.eval_mode = eval_mode
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                            if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                continue
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            while new_name in data_dict:
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                            data_dict[new_name] = {'motion': n_motion,
                                                    'length': len(n_motion),
                                                    'text':[text_dict]}
                            new_name_list.append(new_name)
                            length_list.append(len(n_motion))

                if flag:
                    data_dict[name] = {'motion': motion,
                                       'length': len(motion),
                                       'text':text_data}
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass


        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                       4 + (joints_num - 1) * 9: 4 + (
                                                                                                   joints_num - 1) * 9 + joints_num * 3] / 1.0
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                              4 + (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

            assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def real_len(self):
        return len(self.data_dict)

    def __len__(self):
        return self.real_len() * self.times

    def __getitem__(self, item):
        idx = item % self.real_len()
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data['caption']

        max_motion_length = self.opt.max_motion_length
        if m_length >= self.opt.max_motion_length:
            idx = random.randint(0, len(motion) - max_motion_length)
            motion = motion[idx: idx + max_motion_length]
        else:
            padding_len = max_motion_length - m_length
            D = motion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            motion = np.concatenate((motion, padding_zeros), axis=0)

        assert len(motion) == max_motion_length
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if self.eval_mode:
            tokens = text_data['tokens']
            if len(tokens) < self.opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:self.opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length
        return caption, motion, m_length
class Beat2MotionDataset(Dataset):
    """
    Dataset for BEAT‑like motion‑text clips stored as
      ├─ npy_segments/xxx.npy          (1, T, 264)
      └─ txt_segments/xxx.txt          captions / tags
    Ready for diffusion‑based motion generation.
    """
    def __init__(
        self,
        opt,                       # Hyper‑parameters & paths holder (any object with the attrs below)
        mean: np.ndarray | None,   # motion mean vector  (D,)
        std:  np.ndarray | None,   # motion std  vector  (D,)
        split_file: str,           # txt: list of clip ids to load
        times: int = 1,            # dataset “augmentation” factor
        w_vectorizer=None,         # optional word ⇒ (embedding, POS‑one‑hot)
        eval_mode: bool = False,   # True → return embeddings instead of raw caption
    ):
        super().__init__()

        self.opt         = opt
        self.times       = times
        self.w_vectorizer = w_vectorizer
        self.eval_mode   = eval_mode
        self.fps         = 60                          # ← your dataset FPS
        self.min_len     = 40 if opt.dataset_name == 'beat' else 24
        self.max_len     = opt.max_motion_length
        self.use_rep     = opt.motion_rep              # 'axis_angle' | 'rep15d' | 'position'

        # -------------------------------------------------
        # 1. Read split file
        # -------------------------------------------------
        with cs.open(split_file, 'r', encoding='utf‑8') as f:
            clip_ids = [ln.strip() for ln in f if ln.strip()]

        self.data_dict, name_lens = {}, []
        for cid in tqdm(clip_ids, desc='Loading motion‑text pairs'):
            try:
                motion_path = pjoin(opt.motion_dir, f'{cid}.npy')
                txt_path    = pjoin(opt.text_dir,   f'{cid}.txt')
                if not (os.path.exists(motion_path) and os.path.exists(txt_path)):
                    continue

                motion_raw = np.load(motion_path)
                if motion_raw.ndim == 3 and motion_raw.shape[0] == 1:
                    motion_raw = motion_raw.squeeze(0)
                elif motion_raw.ndim == 2:
                    pass  # đã đúng shape
                else:
                    print(f"[WARN] skip {cid}: unexpected shape {motion_raw.shape}")
                    continue
                # if T < self.min_len or T >= 200:
                #     continue
                T, D = motion_raw.shape
                # ----- optional representation conversion -----
                betas = None         # BEAT stores body shape; keep None if you do not use it
                if self.use_rep == 'rep15d':
                    rep = get_motion_rep_numpy(motion_raw, betas=betas, device='cpu')
                    motion_proc = rep['rep15d']               # (T, 55*15)
                elif self.use_rep == 'position':
                    rep = get_motion_rep_numpy(motion_raw, betas=betas, device='cpu')
                    motion_proc = rep['position'].reshape(T, -1)
                else:  # 'axis_angle'
                    motion_proc = motion_raw                  # (T, 264)

                # ----- load caption(s) -----
                with cs.open(txt_path, 'r', encoding='utf‑8') as f:
                    caption_lines = [ln.strip() for ln in f if ln.strip()]

                for ln in caption_lines:
                    # Case A: single caption without tags
                    if '#' not in ln:
                        cdict = {'caption': ln,
                                 'tokens': ln.split(' '),
                                 'f': 0.0, 'to': 0.0}
                        self._add_example(cid, motion_proc, cdict, betas)
                    # Case B: BEAT style with time stamps
                    else:
                        try:
                            cap, toks, *tags = ln.split('#')
                            toks   = toks.strip().split(' ')
                            f_tag  = float(tags[0]) if tags else 0.0
                            to_tag = float(tags[1]) if len(tags) > 1 else 0.0
                            sub_motion = self._slice_by_time(
                                motion_proc, f_tag, to_tag
                            ) if (f_tag or to_tag) else motion_proc
                            if len(sub_motion) < self.min_len or len(sub_motion) >= 200:
                                continue
                            cdict = {'caption': cap.strip(),
                                     'tokens': toks,
                                     'f': f_tag, 'to': to_tag}
                            new_id = f'{random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}_{cid}'
                            self._add_example(new_id, sub_motion, cdict, betas)
                        except Exception:
                            continue
            except Exception as e:
                print(f'[WARN] skip {cid}: {e}')

        # Sort by length for easier curriculum‑style batching
        self.name_list = sorted(self.data_dict, key=lambda n: self.data_dict[n]['length'])
        self.length_arr = np.array([self.data_dict[n]['length'] for n in self.name_list])

        # -------------------------------------------------
        # 2. Stats (mean / std)
        # -------------------------------------------------
        if opt.is_train and (mean is None or std is None):
            print('Computing dataset mean / std …')
            all_motion = np.concatenate([self.data_dict[n]['motion'] for n in self.name_list], 0)
            mean = all_motion.mean(0)
            std = all_motion.std(0) + 1e-8
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'),  std)
        self.mean, self.std = mean, std

    # -----------------------------------------------------
    # Helper functions
    # -----------------------------------------------------
    def _slice_by_time(self, motion, t0, t1):
        """Return sub‑sequence between seconds t0 and t1 (FPS aware)."""
        f0 = int(t0 * self.fps);  f1 = int(t1 * self.fps)
        return motion[f0:f1] if (f1 > f0) else motion

    def _add_example(self, key, motion, cdict, betas):
        self.data_dict[key] = {
            'motion':  motion,
            'length':  len(motion),
            'text':    [cdict],
            'betas':   betas
        }

    # -----------------------------------------------------
    # PyTorch Dataset API
    # -----------------------------------------------------
    def __len__(self):
        return len(self.name_list) * self.times

    def real_len(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        try:
            idx %= self.real_len()
            name = self.name_list[idx]
            item = self.data_dict[name]

            motion = item['motion'].copy()                  # (T, D)
            m_len  = item['length']
            # betas  = item['betas']
            text_d = random.choice(item['text'])

            if motion is None:
                raise ValueError("motion is None")
            if not isinstance(motion, np.ndarray):
                raise TypeError(f"motion is not numpy: got {type(motion)}")
            if len(motion.shape) != 2:
                raise ValueError(f"motion shape invalid: {motion.shape}")
            if m_len != motion.shape[0]:
                raise ValueError(f"length mismatch: m_len={m_len}, motion.shape={motion.shape}")

            # crop or pad
            if m_len >= self.max_len:
                start = random.randint(0, m_len - self.max_len)
                motion = motion[start:start + self.max_len]
            else:
                pad_len = self.max_len - m_len
                pad = np.zeros((pad_len, motion.shape[1]), dtype=motion.dtype)
                motion = np.concatenate([motion, pad], 0)

            if np.any(np.isnan(motion)):
                raise ValueError("motion contains NaN")

            motion = (motion - self.mean) / self.std
            caption = text_d.get('caption', '<no caption>')
            if not isinstance(caption, str):
                raise TypeError(f"caption is not string: {caption}")
            print(f"[INFO] {name} | len: {m_len} | caption: {caption}")
            print(f"  motion shape: {motion.shape} ")
            return caption, motion.astype(np.float32), m_len

        except Exception as e:
            print("="*60)
            print(f"[ERROR @ __getitem__]")
            print(f"idx: {idx}")
            print(f"name: {self.name_list[idx]}")
            print(f"error: {e}")
            print("="*60)
            return None


        # if self.eval_mode:
        #     tokens = text_d['tokens'][:self.opt.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens += ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)

        #     pos_ohs, w_embs = [], []
        #     for tok in tokens:
        #         w_emb, pos_oh = self.w_vectorizer[tok]
        #         pos_ohs.append(pos_oh[None])
        #         w_embs.append(w_emb[None])
        #     pos_ohs  = np.concatenate(pos_ohs, 0)
        #     w_embs   = np.concatenate(w_embs, 0)

        #     return w_embs, pos_ohs, text_d['caption'], sent_len, motion.astype(np.float32), m_len, betas

        # return text_d['caption'], motion.astype(np.float32), m_len, betas

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------
    def inv_transform(self, data: np.ndarray):
        """Back‑transform from z‑norm to original scale."""
        return data * self.std + self.mean
