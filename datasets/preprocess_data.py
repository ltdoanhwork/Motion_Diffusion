import sys, os
PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import tqdm
import os  
import numpy as np  
from textgrid import TextGrid 
import argparse
import re

  
def time_to_frame(t, fps=60):  
    return int(round(t * fps))

def extract_sentences_with_text(textgrid_path, motion_data, npy_out_dir, txt_out_dir,
    fps=30, pause_threshold=0.5, split_parts=1, use_first_part_only=True
):
    basename = os.path.splitext(os.path.basename(textgrid_path))[0]
    print(motion_data.shape)
    if motion_data.ndim == 3:  # Shape: (1, frames, features)  
        split_data = np.split(motion_data, split_parts, axis=1)  
        
        if use_first_part_only:  
            motion = split_data[0].squeeze(0)    
        else:  
            reshaped_data = [data.squeeze(0) for data in split_data]  
            motion = reshaped_data[0]  # Sử dụng phần đầu tiên  
    else:  
        # Nếu đã là 2D, sử dụng trực tiếp  
        motion = motion_data[0] if motion_data.ndim == 2 else motion_data  
      
    max_frames = motion.shape[0]  
      
    tg = TextGrid.fromFile(textgrid_path)  
    tier = tg[0]  
  
    # with open(text_path, 'r') as f:  
    #     full_text = f.read().strip()  
  
    os.makedirs(npy_out_dir, exist_ok=True)
    os.makedirs(txt_out_dir, exist_ok=True)
  
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
          
        # Đảm bảo không vượt quá bounds của motion data  
        start_frame = max(0, start_frame)  
        end_frame = min(max_frames, end_frame)  
          
        if end_frame <= start_frame:  
            print(f"⚠️ Skipped sentence {sentence_idx}: invalid frame range [{start_frame}, {end_frame}]")  
            return  
              
        motion_segment = motion[start_frame:end_frame, :]

        fname_base = f"{basename}_sentence_{sentence_idx:03d}"
        np.save(os.path.join(npy_out_dir, fname_base + ".npy"), motion_segment)

        with open(os.path.join(txt_out_dir, fname_base + ".txt"), 'w', encoding='utf-8') as ftxt:
            ftxt.write(" ".join(sentence_text))

        print(f"✅ Saved: {fname_base}.npy & .txt (frames: {start_frame}-{end_frame}, shape: {motion_segment.shape})")
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
                  
                # Reset cho câu tiếp theo  
                sentence_start = None  
                sentence_end = None  
                sentence_text = []  
  
    # Lưu câu cuối cùng  
    if sentence_text:  
        save_sentence()  
  
    print(f"🎉 Extracted {sentence_idx} sentences from {basename}")

def preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir):
    print(f"\n📋 Looking for .bvh files in {base_dir}")
    parser = BVHParser()

    data_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
        ('const', ConstantsRemover()),
        ('np', Numpyfier()),
        ('down', DownSampler(2)),
        ('stdscale', ListStandardScaler())
    ])

    for fname in os.listdir(base_dir):
        if fname.endswith(".bvh"):
            basename = fname.replace(".bvh", "")
            bvh_path = os.path.join(base_dir, fname)
            textgrid_path = os.path.join(base_dir, basename + ".TextGrid")
            text_path = os.path.join(base_dir, basename + ".txt")
            if not os.path.exists(textgrid_path) or not os.path.exists(text_path):
                continue  # Bỏ qua nếu thiếu file

            # Parse BVH và chạy pipeline
            if not os.path.exists(textgrid_path) or not os.path.exists(text_path):
                continue  # Bỏ qua nếu thiếu file

            try:
                parsed_data = parser.parse(bvh_path)
            except Exception as e:
                print(f"❌ Lỗi khi parse {bvh_path}: {e}")
                continue
            piped_data = data_pipe.fit_transform([parsed_data])

            # Gọi hàm tách segment (ghi .npy vào npy_out_dir, .txt vào txt_out_dir)
            extract_sentences_with_text(
                textgrid_path=textgrid_path,
                motion_data=piped_data,
                npy_out_dir=npy_out_dir,
                txt_out_dir=txt_out_dir,
                split_parts=1,
                use_first_part_only=True
            )

def main_preprocess_data(base_dir, npy_out_dir, txt_out_dir):
    base_dir = base_dir
    npy_out_dir = npy_out_dir
    txt_out_dir = txt_out_dir

    os.makedirs(npy_out_dir, exist_ok=True)
    os.makedirs(txt_out_dir, exist_ok=True)
    print("preprocess")
    preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir)


def process_parent_dir(parent_dir, out_root, folders=None, start=None, end=None):
    """Process multiple numeric subfolders under parent_dir.

    - If folders is provided (list of folder names), those are used.
    - Else finds subfolders whose names are digits and optionally filters by start/end.
    Writes outputs into out_root/npy/<foldername>/ and out_root/txt/<foldername>/.
    """
    print(f"\n🔍 Scanning parent directory: {parent_dir}")
    if folders:
        print(f"Using explicit folder list: {folders}")
        to_process = folders
    else:
        entries = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        print(f"Found directories: {entries}")
        # select purely numeric folder names
        numeric = [d for d in entries if re.fullmatch(r"\d+", d)]
        print(f"Numeric folders found: {numeric}")
        # sort numerically
        numeric_sorted = sorted(numeric, key=lambda x: int(x))
        if start is not None or end is not None:
            s = int(start) if start is not None else None
            e = int(end) if end is not None else None
            def in_range(name):
                v = int(name)
                if s is not None and v < s:
                    return False
                if e is not None and v > e:
                    return False
                return True
            to_process = [d for d in numeric_sorted if in_range(d)]
        else:
            to_process = numeric_sorted

    if not to_process:
        print("No folders found to process in", parent_dir)
        return

    for folder in to_process:
        src = os.path.join(parent_dir, folder)
        if not os.path.isdir(src):
            print(f"Skipping {src}: not a directory")
            continue
        npy_out_dir = os.path.join(out_root, 'npy', folder)
        txt_out_dir = os.path.join(out_root, 'txt', folder)
        os.makedirs(npy_out_dir, exist_ok=True)
        os.makedirs(txt_out_dir, exist_ok=True)
        print(f"\n=== Processing folder {folder}: {src} -> {npy_out_dir}, {txt_out_dir} ===")
        preprocess_motion_data(src, npy_out_dir, txt_out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess BVH/TextGrid folders. Can process a single folder or iterate numeric subfolders.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--base-dir', type=str, help='Single input folder containing .bvh/.TextGrid/.txt files')
    group.add_argument('--parent-dir', type=str, help='Parent directory containing numeric subfolders to process')
    parser.add_argument('--npy-out', type=str, help='Output directory for .npy files (single run)')
    parser.add_argument('--txt-out', type=str, help='Output directory for .txt files (single run)')
    parser.add_argument('--out-root', type=str, help='Root output dir when using --parent-dir (default: ./outputs)')
    parser.add_argument('--folders', type=str, help='Comma-separated list of folder names to process under parent-dir')
    parser.add_argument('--start', type=int, help='Start index (inclusive) for numeric folder processing')
    parser.add_argument('--end', type=int, help='End index (inclusive) for numeric folder processing')

    args = parser.parse_args()

    if args.base_dir:
        if not args.npy_out or not args.txt_out:
            parser.error('--base-dir requires --npy-out and --txt-out')
        main_preprocess_data(args.base_dir, args.npy_out, args.txt_out)
    else:
        parent = args.parent_dir
        out_root = args.out_root or os.path.join(os.getcwd(), 'outputs')
        folders = None
        if args.folders:
            folders = [f.strip() for f in args.folders.split(',') if f.strip()]
        process_parent_dir(parent, out_root, folders=folders, start=args.start, end=args.end)