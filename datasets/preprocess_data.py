import sys, os
import joblib
PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import os  
import numpy as np  
from textgrid import TextGrid 
import argparse
import re
import multiprocessing
from tqdm import tqdm

# --- BIẾN TOÀN CỤC CHO WORKER ---
g_parser = None
g_pipeline = None

def init_worker(pipeline_path):
    """Khởi tạo parser và pipeline cho mỗi worker process"""
    global g_parser, g_pipeline
    print(f"Worker {os.getpid()} initializing...")
    g_parser = BVHParser()
    g_pipeline = joblib.load(pipeline_path)

def time_to_frame(t, fps=60):  
    return int(round(t * fps))

def extract_sentences_with_text(textgrid_path, motion_data, npy_out_dir, txt_out_dir,
    fps=30, pause_threshold=0.5, split_parts=1, use_first_part_only=True
):
    """Tách motion data thành các câu dựa trên TextGrid"""
    basename = os.path.splitext(os.path.basename(textgrid_path))[0]
    
    # Xử lý shape của motion_data
    if motion_data.ndim == 3:  # Shape: (1, frames, features)  
        split_data = np.split(motion_data, split_parts, axis=1)  
        
        if use_first_part_only:  
            motion = split_data[0].squeeze(0)    
        else:  
            reshaped_data = [data.squeeze(0) for data in split_data]  
            motion = reshaped_data[0]
    else:  
        motion = motion_data[0] if motion_data.ndim == 2 else motion_data  
      
    max_frames = motion.shape[0]  
      
    tg = TextGrid.fromFile(textgrid_path)  
    tier = tg[0]  
  
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
          
        start_frame = max(0, start_frame)  
        end_frame = min(max_frames, end_frame)  
          
        if end_frame <= start_frame:  
            return  
          
        motion_segment = motion[start_frame:end_frame, :]

        fname_base = f"{basename}_sentence_{sentence_idx:03d}"
        np.save(os.path.join(npy_out_dir, fname_base + ".npy"), motion_segment)

        with open(os.path.join(txt_out_dir, fname_base + ".txt"), 'w', encoding='utf-8') as ftxt:
            ftxt.write(" ".join(sentence_text))

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
  
    return sentence_idx  # Trả về số câu đã extract

def worker_process_file(args):
    """
    Worker function: Parse 1 file BVH + transform + extract sentences
    Args: tuple(bvh_path, textgrid_path, npy_out_dir, txt_out_dir)
    """
    bvh_path, textgrid_path, npy_out_dir, txt_out_dir = args
    global g_parser, g_pipeline
    
    basename = os.path.splitext(os.path.basename(bvh_path))[0]
    
    try:
        # 1. Parse BVH
        parsed_data = g_parser.parse(bvh_path)
        
        # 2. Transform qua pipeline (đã fit)
        piped_data = g_pipeline.transform([parsed_data])
        
        # 3. Extract sentences từ TextGrid
        num_sentences = extract_sentences_with_text(
            textgrid_path=textgrid_path,
            motion_data=piped_data,
            npy_out_dir=npy_out_dir,
            txt_out_dir=txt_out_dir,
            split_parts=1,
            use_first_part_only=True
        )
        
        return (basename, num_sentences, None)  # (filename, số câu, error)
        
    except Exception as e:
        return (basename, 0, str(e))

def preprocess_motion_data_multiprocess(base_dir, npy_out_dir, txt_out_dir, pipeline_path, num_workers=None):
    """
    Xử lý tất cả file BVH trong folder bằng multiprocessing
    """
    print(f"\n📋 Scanning for BVH files in {base_dir}")
    
    # Thu thập tất cả các file cần xử lý
    tasks = []
    for fname in os.listdir(base_dir):
        if fname.endswith(".bvh"):
            basename = fname.replace(".bvh", "")
            bvh_path = os.path.join(base_dir, fname)
            textgrid_path = os.path.join(base_dir, basename + ".TextGrid")
            text_path = os.path.join(base_dir, basename + ".txt")
            
            # Chỉ xử lý nếu có đủ TextGrid và txt
            if os.path.exists(textgrid_path) and os.path.exists(text_path):
                tasks.append((bvh_path, textgrid_path, npy_out_dir, txt_out_dir))
    
    if not tasks:
        print(f"⚠️ No valid BVH files found in {base_dir}")
        return
    
    print(f"✅ Found {len(tasks)} files to process")
    
    # Xác định số workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"🚀 Processing with {num_workers} workers...")
    
    # Tạo output directories
    os.makedirs(npy_out_dir, exist_ok=True)
    os.makedirs(txt_out_dir, exist_ok=True)
    
    # Multiprocessing
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(pipeline_path,)
    ) as pool:
        results = list(tqdm(
            pool.imap(worker_process_file, tasks),
            total=len(tasks),
            desc=f"Processing {os.path.basename(base_dir)}"
        ))
    
    # Thống kê kết quả
    success_count = sum(1 for r in results if r[2] is None)
    total_sentences = sum(r[1] for r in results if r[2] is None)
    
    print(f"\n📊 Results for {os.path.basename(base_dir)}:")
    print(f"   ✅ Success: {success_count}/{len(tasks)} files")
    print(f"   📝 Total sentences extracted: {total_sentences}")
    
    # In lỗi nếu có
    errors = [(r[0], r[2]) for r in results if r[2] is not None]
    if errors:
        print(f"   ❌ Errors: {len(errors)}")
        for fname, err in errors[:5]:  # Chỉ in 5 lỗi đầu
            print(f"      - {fname}: {err}")
        if len(errors) > 5:
            print(f"      ... and {len(errors)-5} more errors")

def process_parent_dir(parent_dir, out_root, pipeline_path, folders=None, start=None, end=None, num_workers=None):
    """
    Xử lý nhiều folders con trong parent directory
    """
    print(f"\n🔍 Scanning parent directory: {parent_dir}")
    
    # Kiểm tra pipeline tồn tại
    if not os.path.exists(pipeline_path):
        print(f"❌ ERROR: Pipeline not found at '{pipeline_path}'")
        print("You must run 'step1_fit_scaler.py' first!")
        return
    
    print(f"✅ Pipeline found: {pipeline_path}")
    
    # Xác định folders cần xử lý
    if folders:
        print(f"Using explicit folder list: {folders}")
        to_process = folders
    else:
        entries = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
        numeric = [d for d in entries if re.fullmatch(r"\d+", d)]
        numeric_sorted = sorted(numeric, key=lambda x: int(x))
        
        if start is not None or end is not None:
            s = int(start) if start is not None else None
            e = int(end) if end is not None else None
            to_process = [d for d in numeric_sorted 
                         if (s is None or int(d) >= s) and (e is None or int(d) <= e)]
        else:
            to_process = numeric_sorted

    if not to_process:
        print("❌ No folders found to process")
        return

    print(f"📂 Will process {len(to_process)} folders: {to_process[:5]}{'...' if len(to_process) > 5 else ''}")
    
    # Xử lý từng folder
    for i, folder in enumerate(to_process, 1):
        src = os.path.join(parent_dir, folder)
        if not os.path.isdir(src):
            print(f"⚠️ Skipping {src}: not a directory")
            continue
        
        npy_out_dir = os.path.join(out_root, 'npy', folder)
        txt_out_dir = os.path.join(out_root, 'txt', folder)
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(to_process)}] Processing folder: {folder}")
        print(f"{'='*60}")
        
        preprocess_motion_data_multiprocess(
            base_dir=src,
            npy_out_dir=npy_out_dir,
            txt_out_dir=txt_out_dir,
            pipeline_path=pipeline_path,
            num_workers=num_workers
        )
    
    print(f"\n🎉 All folders processed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess BVH/TextGrid folders with MULTIPROCESSING'
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--base-dir', type=str, 
                      help='Single input folder containing .bvh/.TextGrid/.txt files')
    group.add_argument('--parent-dir', type=str, 
                      help='Parent directory containing numeric subfolders to process')
    
    parser.add_argument('--npy-out', type=str, 
                       help='Output directory for .npy files (single run)')
    parser.add_argument('--txt-out', type=str, 
                       help='Output directory for .txt files (single run)')
    parser.add_argument('--out-root', type=str, 
                       help='Root output dir when using --parent-dir (default: ./outputs)')
    
    parser.add_argument('--pipeline', type=str, default='global_pipeline.pkl',
                       help='Path to fitted pipeline (default: global_pipeline.pkl)')
    
    parser.add_argument('--folders', type=str, 
                       help='Comma-separated list of folder names to process under parent-dir')
    parser.add_argument('--start', type=int, 
                       help='Start index (inclusive) for numeric folder processing')
    parser.add_argument('--end', type=int, 
                       help='End index (inclusive) for numeric folder processing')
    
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')

    args = parser.parse_args()

    if args.base_dir:
        # Single folder mode
        if not args.npy_out or not args.txt_out:
            parser.error('--base-dir requires --npy-out and --txt-out')
        
        print("🚀 Single folder mode with multiprocessing")
        preprocess_motion_data_multiprocess(
            base_dir=args.base_dir,
            npy_out_dir=args.npy_out,
            txt_out_dir=args.txt_out,
            pipeline_path=args.pipeline,
            num_workers=args.workers
        )
    else:
        # Multiple folders mode
        out_root = args.out_root or os.path.join(os.getcwd(), 'outputs')
        folders_list = None
        if args.folders:
            folders_list = [f.strip() for f in args.folders.split(',') if f.strip()]
        
        process_parent_dir(
            parent_dir=args.parent_dir,
            out_root=out_root,
            pipeline_path=args.pipeline,
            folders=folders_list,
            start=args.start,
            end=args.end,
            num_workers=args.workers
        )