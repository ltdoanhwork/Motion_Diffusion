from datetime import datetime
import glob
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import get_dataset_motion_loader, get_motion_loader
from models import MotionTransformer
from utils.get_opt import get_opt
from utils.metrics import *
from datasets import EvaluatorModelWrapper
from collections import OrderedDict
from utils.plot_script import *
from utils import paramUtil
from utils.utils import *
from trainers import DDPMTrainer
from utils.motion_process import recover_from_ric


def build_models(opt, dim_pose):
    encoder = MotionTransformer(
        input_feats=dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim,
        no_clip=opt.no_clip,
        no_eff=opt.no_eff)
    return encoder



torch.multiprocessing.set_sharing_strategy('file_system')


def _infer_dim_pose(motion_dir, default_dim):
    try:
        files = glob.glob(os.path.join(motion_dir, "**/*.npy"), recursive=True)
        if not files:
            return default_dim
        arr = np.load(files[0])
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim == 2:
            return arr.shape[1]
    except Exception:
        return default_dim
    return default_dim


def _setup_dataset(opt):
    if opt.dataset_name == 't2m':
        opt.data_root = pjoin(ROOT, 'datasets', 'HumanML3D', 'HumanML3D')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.max_motion_length = 196
        opt.dim_pose = 263
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = pjoin(ROOT, 'datasets', 'KIT-ML')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.max_motion_length = 196
        opt.dim_pose = 251
        kinematic_chain = paramUtil.kit_kinematic_chain
    elif opt.dataset_name == 'beat':
        opt.data_root = pjoin(ROOT, 'datasets', 'BEAT_numpy')
        opt.motion_dir = pjoin(opt.data_root, 'npy')
        opt.text_dir = pjoin(opt.data_root, 'txt')
        opt.joints_num = 88
        opt.max_motion_length = 360
        opt.dim_pose = _infer_dim_pose(opt.motion_dir, 264)
        kinematic_chain = paramUtil.beat_kinematic_chain
    else:
        raise KeyError(f'Unknown dataset {opt.dataset_name}')
    return kinematic_chain


def _resolve_checkpoint(opt):
    direct = pjoin(opt.model_dir, f"{opt.which_epoch}.tar")
    if os.path.exists(direct):
        return direct
    ckpts = sorted(glob.glob(pjoin(opt.model_dir, "ckpt_e*.tar")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {opt.model_dir}")
    return ckpts[-1]


def MAE_score(groundtruth_loader, trainers, gt_dataset, mean, std, dim=3, mm_num_samples=100, mm_num_repeats=10):
    """Compute Mean Absolute Error (MAE), Velocity, and Jerk

    Args:
        original:     array containing joint positions of original gesture
        predicted:    array containing joint positions of predicted gesture
        dim:          gesture dimensionality

    Returns:
        mae:          MAE between original and predicted for each joint
        velocity:     Velocity error between original and predicted for each joint
        jerk:         Jerk error between original and predicted for each joint
    """
    dataloader = DataLoader(gt_dataset, batch_size=1, num_workers=1, shuffle=True)
    trainers.eval_mode()
    trainers.to(opt.device)

    mm_generated_motions = []
    mm_num_samples = min(mm_num_samples, len(gt_dataset))
    mm_idxs = np.random.choice(len(gt_dataset), mm_num_samples, replace=False)
    mm_idxs = np.sort(mm_idxs)
    all_caption = []
    all_m_lens = []
    all_data = []
    all_motion = []
    min_mov_length = 10
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
            all_data.append(data)
            tokens = tokens[0].split('_')
            mm_num_now = len(mm_generated_motions)
            is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
            repeat_times = mm_num_repeats if is_mm else 1
            m_lens = (m_lens // opt.unit_length) * opt.unit_length
            m_lens = torch.clamp(m_lens, min=min_mov_length * opt.unit_length, max=opt.max_motion_length)
            if isinstance(m_lens, int):
                m_lens = torch.LongTensor([m_lens]).to(opt.device)
            else:
                m_lens = m_lens.to(opt.device)
            for t in range(repeat_times):
                all_m_lens.append(m_lens)
                all_caption.extend(caption)
                all_motion.append(motions.cpu().numpy())
            if is_mm:
                mm_generated_motions.append(0)
    all_m_lens = torch.stack(all_m_lens)

    # Generate all sequences
    with torch.no_grad():
        predicted = trainers.generate(all_caption, all_m_lens, opt.dim_pose)
        predicted = torch.stack(predicted, dim=0).cpu().numpy()
        predicted = predicted * std + mean
        if opt.dataset_name in ['t2m', 'kit']:
            predicted = recover_from_ric(torch.from_numpy(predicted).float(), opt.joints_num).numpy()
        print(len(all_motion))
        # original = np.concatenate(all_motion, axis=0)
        
        original = np.concatenate(all_motion, axis=0)
        print(original.shape)
        original = original * std + mean
        # Truncate original to match predicted length
        # original = np.stack(all_motion, axis=0)
        # print(original.shape)
        if opt.dataset_name in ['t2m', 'kit']:
            original = recover_from_ric(torch.from_numpy(original).float(), opt.joints_num).numpy()
        print(original.shape) 
        # num_frames, num_joints, dim = predicted.shape[1], predicted.shape[2], predicted.shape[3]
        assert not np.isnan(predicted).any(), "NaN values detected in predicted array"
        assert not np.isnan(original).any(), "NaN values detected in original array"

        # MAE calculation: support both (N,T,D) and (N,T,J,3)
        reduce_axes_np = tuple(range(1, predicted.ndim))
        mae = np.mean(np.abs(predicted - original), axis=reduce_axes_np)

        # Velocity calculation (first derivative)
        # velocity_original = np.gradient(original, axis=1)  # Gradient along time
        # velocity_predicted = np.gradient(predicted, axis=1)
        # velocity_error = np.mean(np.abs(velocity_original - velocity_predicted), axis=(0, 1, 2))

        # # Jerk calculation (third derivative)
        # jerk_original = np.gradient(np.gradient(np.gradient(original, axis=1), axis=1), axis=1)
        # jerk_predicted = np.gradient(np.gradient(np.gradient(predicted, axis=1), axis=1), axis=1)
        # jerk_error = np.mean(np.abs(jerk_original - jerk_predicted), axis=(0, 1, 2))
        velocity_predicted = torch.diff(torch.from_numpy(predicted), dim=1)
        velocity_original = torch.diff(torch.from_numpy(original), dim=1)

        # Calculate Velocity Error (MAE between velocity_predicted and velocity_original)
        reduce_axes_t = tuple(range(1, velocity_predicted.ndim))
        velocity_error = torch.mean(torch.abs(velocity_predicted - velocity_original), dim=reduce_axes_t)
        velocity_error = torch.mean(velocity_error)  # Averaging over the batch

        # Jerk Calculation (Third Derivative)
        # Apply diff three times to get the third derivative
        jerk_predicted = torch.diff(velocity_predicted, dim=1)
        jerk_original = torch.diff(velocity_original, dim=1)

        # Calculate Jerk Error (MAE between jerk_predicted and jerk_original)
        reduce_axes_j = tuple(range(1, jerk_predicted.ndim))
        jerk_error = torch.mean(torch.abs(jerk_predicted - jerk_original), dim=reduce_axes_j)
        jerk_error = torch.mean(jerk_error)  # Averaging over the batch

    # mae, velocity_error, and jerk_error now contain the respective metrics for each joint
    return mae, velocity_error, jerk_error

# def PAE_score(groundtruth_loader, trainers, gt_dataset, dim=3, mm_num_samples=100, mm_num_repeats=10):
#     """Compute Position Average Error (PAE) for each joint

#     Args:
#         groundtruth_loader: DataLoader for ground truth data
#         trainers: Model trainer instance
#         gt_dataset: Ground truth dataset
#         dim: Dimensionality of gestures (default: 3 for 3D gestures)

#     Returns:
#         np.ndarray: PAE between original and predicted for each joint
#     """
#     dataloader = DataLoader(gt_dataset, batch_size=1, num_workers=1, shuffle=True)
#     epoch, it = trainers.load('/home/ltdoanh/jupyter/jupyter/ldtan/MotionDiffuse/t2m/t2m_new_ver2/model/ckpt_e030.tar')
#     trainers.eval_mode()
#     trainers.to(opt.device)
#     mm_generated_motions = []
#     mm_idxs = np.random.choice(len(gt_dataset), mm_num_samples, replace=False)
#     mm_idxs = np.sort(mm_idxs)
#     all_caption = []
#     all_m_lens = []
#     all_motion = []

#     min_mov_length = 10
#     with torch.no_grad():
#         for i, data in enumerate(dataloader):
#             word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
#             tokens = tokens[0].split('_')
#             mm_num_now = len(mm_generated_motions)
#             is_mm = (mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])
#             repeat_times = mm_num_repeats if is_mm else 1
#             m_lens = max(m_lens // opt.unit_length * opt.unit_length, min_mov_length * opt.unit_length)
#             m_lens = min(m_lens, opt.max_motion_length)
#             m_lens = m_lens.to(opt.device) if not isinstance(m_lens, int) else torch.LongTensor([m_lens]).to(opt.device)
#             for t in range(repeat_times):
#                 all_m_lens.append(m_lens)
#                 all_caption.extend(caption)
#                 all_motion.extend(motions.numpy())
#             if is_mm:
#                 mm_generated_motions.append(0)

#     all_m_lens = torch.stack(all_m_lens)

#     # Generate all sequences
#     # Generate all sequences
#     with torch.no_grad():
#         predicted = trainers.generate(all_caption, all_m_lens, opt.dim_pose)
#         predicted  = predicted[0].cpu().numpy()
#         predicted  = recover_from_ric(torch.from_numpy(predicted).float(), opt.joints_num).numpy()
#         original = np.stack(all_motion, axis=0)

#         # Ensure that `original` and `predicted` have compatible shapes.
#         # Expected shape: (num_frames, num_joints, 3)
#         print("Original shape:", original.shape)
#         print("Predicted shape:", predicted.shape)

#         if original.shape != predicted.shape:
#             raise ValueError(f"Shape mismatch: original {original.shape} vs predicted {predicted.shape}")

#         num_frames = predicted.shape[0]
#         num_joints = predicted.shape[1]

#         diffs = np.zeros((num_frames, num_joints))

#         for i in range(num_frames):
#             for j in range(num_joints):
#                 orig_joint = original[i, j, :]
#                 pred_joint = predicted[i, j, :]
#                 # Ensure both `orig_joint` and `pred_joint` are of shape `(3,)`
#                 if orig_joint.shape != pred_joint.shape:
#                     raise ValueError(f"Joint shape mismatch: orig_joint {orig_joint.shape} vs pred_joint {pred_joint.shape}")

#                 diffs[i, j] = np.linalg.norm(orig_joint - pred_joint)

#         return np.mean(diffs, axis=0)
def PAE_score(groundtruth_loader, trainers, gt_dataset, mean, std, dim=3, mm_num_samples=100, mm_num_repeats=10):
    """Compute Position Average Error (PAE) for each joint

    Args:
        groundtruth_loader: DataLoader for ground truth data
        trainers: Model trainer instance
        gt_dataset: Ground truth dataset
        dim: Dimensionality of gestures (default: 3 for 3D gestures)

    Returns:
        np.ndarray: PAE between original and predicted for each joint
    """
    dataloader = DataLoader(gt_dataset, batch_size=1, num_workers=1, shuffle=True)
    trainers.eval_mode()
    trainers.to(opt.device)
    mm_generated_motions = []
    mm_num_samples = min(mm_num_samples, len(gt_dataset))
    mm_idxs = np.random.choice(len(gt_dataset), mm_num_samples, replace=False)
    mm_idxs = np.sort(mm_idxs)
    all_caption = []
    all_m_lens = []
    all_motion = []
    min_mov_length = 10

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            word_emb, pos_ohot, caption, cap_lens, motions, m_lens, tokens = data
            tokens = tokens[0].split('_')
            mm_num_now = len(mm_generated_motions)
            is_mm = (mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])
            repeat_times = mm_num_repeats if is_mm else 1
            m_lens = (m_lens // opt.unit_length) * opt.unit_length
            m_lens = torch.clamp(m_lens, min=min_mov_length * opt.unit_length, max=opt.max_motion_length)
            if isinstance(m_lens, int):
                m_lens = torch.LongTensor([m_lens]).to(opt.device)
            else:
                m_lens = m_lens.to(opt.device)
            for t in range(repeat_times):
                all_m_lens.append(m_lens)
                all_caption.extend(caption)
                all_motion.append(motions.cpu().numpy())
                # all_motion.extend(motions)
            if is_mm:
                mm_generated_motions.append(0)

    all_m_lens = torch.stack(all_m_lens)
    
    # Generate all sequences
    with torch.no_grad():
        predicted = trainers.generate(all_caption, all_m_lens, opt.dim_pose)
        predicted = torch.stack(predicted, dim=0).cpu().numpy()
        predicted = predicted * std + mean
        if opt.dataset_name in ['t2m', 'kit']:
            predicted = recover_from_ric(torch.from_numpy(predicted).float(), opt.joints_num).numpy()
        
        # Concatenate all ground truth motions and recover them
        original = np.concatenate(all_motion, axis=0)
        original = original * std + mean
        if opt.dataset_name in ['t2m', 'kit']:
            original = recover_from_ric(torch.from_numpy(original).float(), opt.joints_num).numpy()

        # Get the shapes for joint-wise error computation
        # num_frames = predicted.shape[1]  # Number of time steps (196)
        # num_joints = predicted.shape[2]  # Number of joints (22)
        # dim = predicted.shape[3]         # Dimensionality (3 for x, y, z)

        # # Compute Position Average Error (PAE)
        # pae = np.empty(num_joints)
        # for i in range(num_joints):
        #     # Compute the distance between original and predicted joints for each time step
        #     distance = np.linalg.norm(original[:, :, i, :] - predicted[:, :, i, :], axis=-1)
        #     # Compute the mean distance across all frames and samples for this joint
        #     pae[i] = np.mean(distance)
        pred_t = torch.from_numpy(predicted)
        orig_t = torch.from_numpy(original)
        reduce_axes = tuple(range(1, pred_t.ndim))
        pae = torch.mean(torch.abs(pred_t - orig_t), dim=reduce_axes)
        pae = torch.mean(pae)

    # pae now contains the PAE for each joint
    return pae


def evaluate_matching_score(motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    # print(motion_loaders.keys())
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                assert not np.isnan(text_embeddings.cpu().numpy()).any(), "Text embeddings contain NaN values"
                assert not np.isnan(motion_embeddings.cpu().numpy()).any(), "Motion embeddings contain NaN values"
                assert not np.isinf(text_embeddings.cpu().numpy()).any(), "Text embeddings contain Inf values"
                assert not np.isinf(motion_embeddings.cpu().numpy()).any(), "Motion embeddings contain Inf values"

                # Ensure distance matrix is valid
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy())
                assert not np.isnan(dist_mat).any(), "Distance matrix contains NaN values"
                assert not np.isinf(dist_mat).any(), "Distance matrix contains Inf values"

                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []
    print('========== Evaluating FID ==========')  # Start of FID evaluation

    with torch.no_grad():
        for idx, batch in enumerate(groundtruth_loader):
            try:
                # Extract the batch data
                _, _, _, sent_lens, motions, m_lens, _ = batch

                # Debug: Check shapes and data types of inputs
                print(f"Batch {idx} - motions shape: {motions.shape}, m_lens: {m_lens}")

                # Compute motion embeddings
                motion_embeddings = eval_wrapper.get_motion_embeddings(
                    motions=motions,
                    m_lens=m_lens
                )

                # Debug: Check for NaN or Inf values in embeddings
                assert not torch.isnan(motion_embeddings).any(), f"NaN detected in motion_embeddings for batch {idx}"
                assert not torch.isinf(motion_embeddings).any(), f"Inf detected in motion_embeddings for batch {idx}"

                # Collect embeddings
                gt_motion_embeddings.append(motion_embeddings.cpu().numpy())

            except Exception as e:
                print(f"Error processing batch {idx}: {e}")
                continue  # Skip this batch if there's an error

    try:
        # Concatenate all motion embeddings
        gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)

        # Debug: Check shape of concatenated embeddings
        print(f"Concatenated gt_motion_embeddings shape: {gt_motion_embeddings.shape}")

        # Calculate mean and covariance of ground truth embeddings
        gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

        # Debug: Check for NaN or Inf values in statistics
        assert not np.isnan(gt_mu).any(), "NaN detected in gt_mu"
        assert not np.isinf(gt_mu).any(), "Inf detected in gt_mu"
        assert not np.isnan(gt_cov).any(), "NaN detected in gt_cov"
        assert not np.isinf(gt_cov).any(), "Inf detected in gt_cov"

    except Exception as e:
        print(f"Error calculating ground truth statistics: {e}")
        return eval_dict  # Return empty dict if there's an error

    # Iterate through each model's motion embeddings
    for model_name, motion_embeddings in activation_dict.items():
        try:
            # Debug: Check shape of model embeddings
            print(f"Model '{model_name}' - motion_embeddings shape: {motion_embeddings.shape}")

            # Calculate mean and covariance of model embeddings
            mu, cov = calculate_activation_statistics(motion_embeddings)

            # Debug: Check for NaN or Inf values in statistics
            assert not np.isnan(mu).any(), f"NaN detected in mu for model '{model_name}'"
            assert not np.isinf(mu).any(), f"Inf detected in mu for model '{model_name}'"
            assert not np.isnan(cov).any(), f"NaN detected in cov for model '{model_name}'"
            assert not np.isinf(cov).any(), f"Inf detected in cov for model '{model_name}'"

            # Calculate FID
            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
            print(f'---> [{model_name}] FID: {fid:.4f}')
            print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)
            eval_dict[model_name] = fid

        except Exception as e:
            print(f"Error calculating FID for model '{model_name}': {e}")
            continue  # Skip this model if there's an error

    return eval_dict



def evaluate_diversity(activation_dict, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)
    return eval_dict


def evaluate_multimodality(mm_motion_loaders, file):
    eval_dict = OrderedDict({})
    print('========== Evaluating MultiModality ==========')
    for model_name, mm_motion_loader in mm_motion_loaders.items():
        # === THÊM ĐOẠN CHECK NÀY ===
        if mm_motion_loader is None:
            print(f'---> [{model_name}] Multimodality: Skipped (No Loader)')
            continue
        # ===========================
        mm_motion_embeddings = []
        with torch.no_grad():
            for idx, batch in enumerate(mm_motion_loader):
                # (1, mm_replications, dim_pos)
                motions, m_lens = batch
                motion_embedings = eval_wrapper.get_motion_embeddings(motions[0], m_lens[0])
                mm_motion_embeddings.append(motion_embedings.unsqueeze(0))
        if len(mm_motion_embeddings) == 0:
            multimodality = 0
        else:
            mm_motion_embeddings = torch.cat(mm_motion_embeddings, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(mm_motion_embeddings, mm_num_times)
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}')
        print(f'---> [{model_name}] Multimodality: {multimodality:.4f}', file=file, flush=True)
        eval_dict[model_name] = multimodality
    return eval_dict


def get_metric_statistics(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times=1):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({'Matching Score': OrderedDict({}),
                                   'R_precision': OrderedDict({}),
                                   'FID': OrderedDict({}),
                                   'Diversity': OrderedDict({}),
                                   'MultiModality': OrderedDict({})})
        for replication in range(replication_times):
            motion_loaders = {}
            mm_motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, mm_motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader
                mm_motion_loaders[motion_loader_name] = mm_motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(motion_loaders, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            mm_score_dict = evaluate_multimodality(mm_motion_loaders, f)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                if key not in all_metrics['Matching Score']:
                    all_metrics['Matching Score'][key] = [item]
                else:
                    all_metrics['Matching Score'][key] += [item]

            for key, item in R_precision_dict.items():
                if key not in all_metrics['R_precision']:
                    all_metrics['R_precision'][key] = [item]
                else:
                    all_metrics['R_precision'][key] += [item]

            for key, item in fid_score_dict.items():
                if key not in all_metrics['FID']:
                    all_metrics['FID'][key] = [item]
                else:
                    all_metrics['FID'][key] += [item]

            for key, item in div_score_dict.items():
                if key not in all_metrics['Diversity']:
                    all_metrics['Diversity'][key] = [item]
                else:
                    all_metrics['Diversity'][key] += [item]

            for key, item in mm_score_dict.items():
                if key not in all_metrics['MultiModality']:
                    all_metrics['MultiModality'][key] = [item]
                else:
                    all_metrics['MultiModality'][key] += [item]


        # print(all_metrics['Diversity'])
        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                # print(metric_name, model_name)
                mean, conf_interval = get_metric_statistics(np.array(values))
                # print(mean, mean.dtype)
                if isinstance(mean, np.float64) or isinstance(mean, np.float32):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}]'
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f;' % (i+1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python tools/evaluation.py <opt.txt>")

    mm_num_samples = 20 # 100
    mm_num_repeats = 2 # 30
    mm_num_times = 8 # 10
    diversity_times = 300
    replication_times = 1
    batch_size = 64

    opt_path = sys.argv[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(0)

    opt = get_opt(opt_path, device)
    _setup_dataset(opt)

    ckpt_path = _resolve_checkpoint(opt)
    print(f"[INFO] Using checkpoint: {ckpt_path}")

    mean = np.load(pjoin(opt.meta_dir, 'mean.npy'))
    std = np.load(pjoin(opt.meta_dir, 'std.npy'))

    gt_loader, gt_dataset = get_dataset_motion_loader(opt_path, batch_size, device)

    run_text_metrics = opt.dataset_name in ['t2m', 'kit']
    eval_wrapper = None
    if run_text_metrics:
        wrapper_opt = get_opt(opt_path, device)
        _setup_dataset(wrapper_opt)
        if not hasattr(wrapper_opt, 'evaluator_path'):
            wrapper_opt.evaluator_path = pjoin(ROOT, 'checkpoints', 't2m', 'text_mot_match', 'model', 'finest.tar')
        if not os.path.exists(wrapper_opt.evaluator_path):
            raise FileNotFoundError(f"Evaluator checkpoint not found: {wrapper_opt.evaluator_path}")
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    encoder = build_models(opt, opt.dim_pose)
    trainer = DDPMTrainer(opt, encoder)
    trainer.load(ckpt_path)

    log_path = pjoin(opt.save_root, 'evaluation_metrics.txt')
    with open(log_path, 'a') as f:
        print("BEGIN_____________________________________________________BEGIN", file=f)
        print("MAE score", file=f)
        mae, velocity_error, jerk_error = MAE_score(
            gt_loader, trainer, gt_dataset, mean, std, dim=3,
            mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats
        )
        print(mae, file=f)
        print("velocity_error", file=f)
        print(velocity_error, file=f)
        print("jerk_error", file=f)
        print(jerk_error, file=f)
        print("PAE score", file=f)
        print(PAE_score(
            gt_loader, trainer, gt_dataset, mean, std, dim=3,
            mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats
        ), file=f)
        print("END_____________________________________________________END", file=f)

    if run_text_metrics:
        eval_motion_loaders = {
            'text2motion': lambda: get_motion_loader(
                opt,
                batch_size,
                trainer,
                gt_dataset,
                mm_num_samples,
                mm_num_repeats
            )
        }

        log_file = pjoin(opt.save_root, 'evaluation_log.txt')
        evaluation(eval_wrapper, gt_loader, eval_motion_loaders, log_file, replication_times=replication_times)
    else:
        print("[WARN] Skipping Matching/FID/Diversity/Multimodality for BEAT. "
              "Evaluator model is trained on HumanML3D (263 dims) and is not compatible with BEAT (264 dims).")

    print(MAE_score(
        gt_loader, trainer, gt_dataset, mean, std, dim=3,
        mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats
    ))
    print(PAE_score(
        gt_loader, trainer, gt_dataset, mean, std, dim=3,
        mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats
    ))
