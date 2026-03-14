import sys
import os
import numpy as np
import multiprocessing
import torch
from textgrid import TextGrid
from sklearn.pipeline import Pipeline

PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from datasets.emage_utils.rotation_conversions import axis_angle_to_rotation_6d

def time_to_frame(t, fps=60):  
    return int(round(t * fps))


def axis_angle_numpy_to_rot6d(
    motion_axis_angle: np.ndarray,
    preserve_root_translation: bool = True,
) -> np.ndarray:
    """
    Convert motion from axis-angle layout (T, J*3) to rotation-6d layout (T, J*6).
    """
    if motion_axis_angle.ndim != 2:
        raise ValueError(
            f"Expected 2D motion array, got {motion_axis_angle.shape}"
        )
    T, D = motion_axis_angle.shape
    has_root_translation = False
    root_trans = None

    # Common BEAT expmap layout may include root translation at the first 3 dims.
    if preserve_root_translation and D >= 6 and (D - 3) % 3 == 0:
        has_root_translation = True
        root_trans = motion_axis_angle[:, :3]
        rot_part = motion_axis_angle[:, 3:]
    elif D % 3 == 0:
        rot_part = motion_axis_angle
    else:
        raise ValueError(
            f"Expected axis-angle-like motion with D % 3 == 0 (or (D-3) % 3 == 0 with root translation), got {motion_axis_angle.shape}"
        )

    J = rot_part.shape[1] // 3
    aa = torch.from_numpy(rot_part.astype(np.float32)).view(T, J, 3)
    rot6d = axis_angle_to_rotation_6d(aa).reshape(T, J * 6).cpu().numpy()
    if has_root_translation:
        return np.concatenate([root_trans.astype(np.float32), rot6d], axis=1)
    return rot6d

def extract_sentences_with_text(textgrid_path, motion_data, output_dir, fps=30, pause_threshold=0.5,
    split_parts=1, use_first_part_only=True, output_rep='position'
):
    basename = os.path.splitext(os.path.basename(textgrid_path))[0]
    print(f"Processing {basename}, shape: {motion_data.shape}") 
    
    if motion_data.ndim == 3: 
        split_data = np.split(motion_data, split_parts, axis=1)  
        
        if use_first_part_only:  
            motion = split_data[0].squeeze(0)    
        else:  
            reshaped_data = [data.squeeze(0) for data in split_data]  
            motion = reshaped_data[0]    
    else:  
        motion = motion_data[0] if motion_data.ndim == 2 else motion_data  
      
    if output_rep == 'rot6d':
        motion = axis_angle_numpy_to_rot6d(motion)
    elif output_rep not in ('position', 'axis_angle'):
        raise ValueError(f"Unsupported output_rep: {output_rep}")
    motion = motion.astype(np.float32, copy=False)

    max_frames = motion.shape[0]  
      
    tg = TextGrid.fromFile(textgrid_path)  
    tier = tg[0]  
    
    os.makedirs(output_dir, exist_ok=True)  
  
    sentence_start = None  
    sentence_end = None  
    sentence_text = []  
    sentence_idx = 0  
      
    def save_sentence():  
        nonlocal sentence_idx, sentence_start, sentence_end, sentence_text  
          
        if not sentence_text or sentence_start is None or sentence_end is None:  
            return  
              
        start_frame = time_to_frame(sentence_start, fps)  
        end_frame = time_to_frame(sentence_end, fps)  
          
        start_frame = max(0, start_frame)  
        end_frame = min(max_frames, end_frame)  
          
        if end_frame <= start_frame:  
            print(f"Skipped sentence {sentence_idx}: invalid frame range [{start_frame}, {end_frame}]")  
            return  
              
        motion_segment = motion[start_frame:end_frame, :]  
          
        fname_base = f"{basename}_sentence_{sentence_idx:03d}"  
        np.save(os.path.join(output_dir, fname_base + ".npy"), motion_segment)  
          
        with open(os.path.join(output_dir, fname_base + ".txt"), 'w') as ftxt:  
            ftxt.write(" ".join(sentence_text))  
          
        print(f"Saved: {fname_base}.npy & .txt (frames: {start_frame}-{end_frame})")  
        sentence_idx += 1  
  
    for interval in tier.intervals:  
        word = interval.mark.strip()  
        xmin = float(interval.minTime)  
        xmax = float(interval.maxTime)  
  
        if word != "":  
            if sentence_start is None:  
                sentence_start = xmin  
            sentence_end = xmax  
            sentence_text.append(word)  
        else:  
            pause_duration = xmax - xmin  
            if pause_duration >= pause_threshold and sentence_text:  
                save_sentence()  
                sentence_start = None  
                sentence_end = None  
                sentence_text = []  
  
    if sentence_text:  
        save_sentence()  
  
    print(f" Extracted {sentence_idx} sentences from {basename}")

def preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir, output_rep='position'):
    parser = BVHParser()

    if output_rep == 'position':
        data_pipe = Pipeline([
            ('param', MocapParameterizer('position')),
            ('rcpn', RootCentricPositionNormalizer()),
            ('delta', RootTransformer('absolute_translation_deltas')),
            ('const', ConstantsRemover()),
            ('np', Numpyfier()),
            ('down', DownSampler(2)),
            ('stdscale', ListStandardScaler())
        ])
    elif output_rep in ('axis_angle', 'rot6d'):
        # IMPORTANT:
        # Keep axis-angle in its geometric domain (radians) before converting to rot6d.
        # Do NOT apply StandardScaler here; otherwise axis-angle values lose their
        # rotational meaning and rot6d conversion becomes invalid.
        data_pipe = Pipeline([
            ('param', MocapParameterizer('expmap')),
            ('np', Numpyfier()),
            ('down', DownSampler(2)),
        ])
    else:
        raise ValueError(f"Unsupported output_rep: {output_rep}")

    if not os.path.exists(base_dir):
        print(f"Folder not found: {base_dir}")
        return

    files = [f for f in os.listdir(base_dir) if f.endswith(".bvh")]
    
    for fname in files:
        basename = fname.replace(".bvh", "")
        bvh_path = os.path.join(base_dir, fname)
        textgrid_path = os.path.join(base_dir, basename + ".TextGrid")
        text_path = os.path.join(base_dir, basename + ".txt")
        
        if not os.path.exists(textgrid_path) or not os.path.exists(text_path):
            continue  

        try:
            parsed_data = parser.parse(bvh_path)
            piped_data = data_pipe.fit_transform([parsed_data])

            extract_sentences_with_text(
                textgrid_path=textgrid_path,
                motion_data=piped_data,
                output_dir=npy_out_dir,
                split_parts=1,
                use_first_part_only=True,
                output_rep=output_rep,
            )

            for f in os.listdir(npy_out_dir):
                if f.endswith(".txt"):
                    src = os.path.join(npy_out_dir, f)
                    dst = os.path.join(txt_out_dir, f)
                    if os.path.exists(src):
                        os.rename(src, dst)
                        
        except Exception as e:
            print(f"Error processing {bvh_path}: {e}")
            continue

def process_folder_wrapper(i):
    try:
        print(f"--> [Start] Processing folder {i}...")
        
        base_dir = f"./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1/{i}/"
        npy_out_dir = f"./datasets/BEAT_numpy/npy/{i}/"
        txt_out_dir = f"./datasets/BEAT_numpy/txt/{i}/"

        os.makedirs(npy_out_dir, exist_ok=True)
        os.makedirs(txt_out_dir, exist_ok=True)
        
        output_rep = os.environ.get("BEAT_OUTPUT_REP", "position").strip().lower()
        preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir, output_rep=output_rep)
        
        print(f"<-- [Done] Finished folder {i}")
        return f"Folder {i}: Success"
    except Exception as e:
        print(f"!!! Error in folder {i}: {e}")
        return f"Folder {i}: Failed - {e}"

def main():
    folder_indices = range(1, 31)
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    output_rep = os.environ.get("BEAT_OUTPUT_REP", "position").strip().lower()
    print(f"Using {num_workers} processes for {len(folder_indices)} folders. output_rep={output_rep}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_folder_wrapper, folder_indices)
        
    print("Results:", results)

if __name__ == "__main__":
    print("=============================================================================================== ")
    main()
    print("=============================================================================================== ")
    print("Preprocessing completed.")
# BEAT_OUTPUT_REP=rot6d PYTHONPATH=. /srv/conda/envs/serverai/motiondiff/bin/python datasets/preprocess_data.py
