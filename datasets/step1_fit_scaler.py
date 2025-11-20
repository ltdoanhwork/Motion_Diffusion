import sys, os
PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

import joblib
import pandas as pd
from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import re
import argparse
import numpy as np
from tqdm import tqdm
import multiprocessing
import warnings

# Tắt các warning từ pandas về DataFrame fragmentation
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', message='.*DataFrame is highly fragmented.*')

# --- CUSTOM DOWNSAMPLER (Fix bug trong pymo) ---
class FixedDownSampler(BaseEstimator, TransformerMixin):
    """DownSampler hoạt động đúng với MocapData objects"""
    def __init__(self, rate):
        self.rate = rate
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Q = []
        for track in X:
            # Clone và downsample DataFrame
            new_track = track.clone()
            new_track.values = track.values.iloc[::self.rate]
            Q.append(new_track)
        return Q
    
    def inverse_transform(self, X, copy=None):
        return X

# --- BIẾN TOÀN CỤC ---
g_parser = None
g_partial_pipe = None

def init_worker():
    """
    Khởi tạo worker: parse + transform qua các bước TRƯỚC ConstantsRemover
    """
    global g_parser, g_partial_pipe

    g_parser = BVHParser()
    
    # Pipeline chỉ tới TRƯỚC ConstantsRemover
    g_partial_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
    ])

def worker_parse_and_transform(bvh_path):
    """
    Worker: Parse + transform tới trước ConstantsRemover
    Trả về: MocapData object (DataFrame)
    """
    global g_parser, g_partial_pipe
    
    try:
        parsed_data = g_parser.parse(bvh_path)
        processed = g_partial_pipe.transform([parsed_data])[0]
        return processed
    except Exception as e:
        return None

def process_folder_multiprocessing(bvh_paths, folder_name):
    """
    Xử lý TẤT CẢ files trong 1 folder bằng multiprocessing
    Trả về: List[MocapData objects]
    """
    if not bvh_paths:
        return []
    
    num_cores = multiprocessing.cpu_count()
    
    print(f"\n📁 Processing folder: {folder_name}")
    print(f"   Files: {len(bvh_paths)} | Cores: {num_cores}")
    
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(worker_parse_and_transform, bvh_paths),
            total=len(bvh_paths),
            desc=f"   → {folder_name}",
            ncols=100
        ))
    
    # Lọc kết quả hợp lệ
    valid_results = [r for r in results if r is not None]
    failed_count = len(results) - len(valid_results)
    
    if failed_count > 0:
        print(f"   ⚠️  {failed_count} files failed")
    
    return valid_results

def main_fit_scaler(parent_dir, folders=None, start=None, end=None, mode='hybrid'):
    """
    CHIẾN LƯỢC HYBRID (Khuyến nghị):
    1. Duyệt TUẦN TỰ qua từng folder
    2. Với MỖI folder: Multiprocessing xử lý TẤT CẢ files trong folder đó
    3. Sau khi xong tất cả folders: Fit ConstantsRemover + Scaler trên toàn bộ dữ liệu
    
    Mode options:
    - 'hybrid': Tuần tự folders, multiprocessing files trong mỗi folder (KHUYẾN NGHỊ)
    - 'full_mp': Multiprocessing toàn bộ (có thể lag máy)
    - 'sequential': Tuần tự hoàn toàn (chậm nhất)
    """
    
    # --- THU THẬP FOLDERS ---
    if folders:
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
        print("❌ No folders found")
        return

    print(f"\n{'='*60}")
    print(f"🎯 MODE: {mode.upper()}")
    print(f"📂 Folders to process: {len(to_process)}")
    print(f"{'='*60}")

    # --- THU THẬP FILES THEO FOLDER ---
    folder_file_map = {}
    total_files = 0
    
    for folder in to_process:
        folder_path = os.path.join(parent_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        bvh_files = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(".bvh")
        ]
        
        if bvh_files:
            folder_file_map[folder] = bvh_files
            total_files += len(bvh_files)

    if not folder_file_map:
        print("❌ No BVH files found")
        return

    print(f"✅ Found {total_files} BVH files across {len(folder_file_map)} folders")
    
    # Hiển thị thống kê
    file_counts = [len(files) for files in folder_file_map.values()]
    print(f"   Min files/folder: {min(file_counts)}")
    print(f"   Max files/folder: {max(file_counts)}")
    print(f"   Avg files/folder: {sum(file_counts)//len(file_counts)}")

    # --- BƯỚC 1: XỬ LÝ PARSE + TRANSFORM ---
    print(f"\n{'='*60}")
    print(f"🚀 PASS 1: Parse + Transform (to before ConstantsRemover)")
    print(f"{'='*60}")
    
    all_mocap_objects = []
    
    if mode == 'hybrid':
        # HYBRID: Tuần tự folders, multiprocessing files trong mỗi folder
        for idx, (folder, bvh_paths) in enumerate(folder_file_map.items(), 1):
            print(f"\n[{idx}/{len(folder_file_map)}] ", end="")
            folder_results = process_folder_multiprocessing(bvh_paths, folder)
            all_mocap_objects.extend(folder_results)
            
            print(f"   ✓ Accumulated: {len(all_mocap_objects)} files")
    
    elif mode == 'full_mp':
        # FULL MULTIPROCESSING: Xử lý tất cả files cùng lúc
        all_bvh_paths = []
        for bvh_paths in folder_file_map.values():
            all_bvh_paths.extend(bvh_paths)
        
        num_cores = multiprocessing.cpu_count()
        print(f"   Using {num_cores} CPU cores for ALL {len(all_bvh_paths)} files")
        
        with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
            results = list(tqdm(
                pool.imap(worker_parse_and_transform, all_bvh_paths),
                total=len(all_bvh_paths),
                desc="   Processing all files",
                ncols=100
            ))
        
        all_mocap_objects = [r for r in results if r is not None]
    
    elif mode == 'sequential':
        # SEQUENTIAL: Tuần tự hoàn toàn
        parser = BVHParser()
        partial_pipe = Pipeline([
            ('param', MocapParameterizer('position')),
            ('rcpn', RootCentricPositionNormalizer()),
            ('delta', RootTransformer('absolute_translation_deltas')),
        ])
        
        for idx, (folder, bvh_paths) in enumerate(folder_file_map.items(), 1):
            print(f"\n[{idx}/{len(folder_file_map)}] 📁 {folder} ({len(bvh_paths)} files)")
            
            for bvh_path in tqdm(bvh_paths, desc=f"   → {folder}", ncols=100):
                try:
                    parsed_data = parser.parse(bvh_path)
                    processed = partial_pipe.transform([parsed_data])[0]
                    all_mocap_objects.append(processed)
                except Exception as e:
                    pass
    
    if not all_mocap_objects:
        print("\n❌ No valid data after Pass 1")
        return
    
    print(f"\n{'='*60}")
    print(f"✅ PASS 1 COMPLETE")
    print(f"   Successfully processed: {len(all_mocap_objects)}/{total_files} files")
    print(f"   Sample shape: {all_mocap_objects[0].values.shape}")
    print(f"{'='*60}")
    
    # --- BƯỚC 2: FIT ConstantsRemover TRÊN TOÀN BỘ DỮ LIỆU ---
    print(f"\n📊 PASS 2: Fitting ConstantsRemover on ALL {len(all_mocap_objects)} files...")
    
    const_remover = ConstantsRemover()
    const_remover.fit(all_mocap_objects)
    
    print(f"✅ ConstantsRemover fitted!")
    print(f"   Constant columns found: {len(const_remover.const_dims_)}")
    if const_remover.const_dims_:
        print(f"   Examples: {list(const_remover.const_dims_)[:5]}")
    
    # --- BƯỚC 3: APPLY CÁC BƯỚC CÒN LẠI + FIT SCALER ---
    print(f"\n📈 PASS 3: Applying remaining steps + fitting Scaler...")
    
    # Apply ConstantsRemover
    print("   → Applying ConstantsRemover...")
    after_const = const_remover.transform(all_mocap_objects)
    print(f"     Shape after: {after_const[0].values.shape}")
    
    # Apply DownSampler
    print("   → Downsampling by factor of 2...")
    downsampler = FixedDownSampler(2)
    downsampled_mocap = downsampler.transform(after_const)
    print(f"     Shape after: {downsampled_mocap[0].values.shape}")
    
    # Convert to numpy arrays
    print("   → Converting to numpy arrays...")
    numpyfier = Numpyfier()
    numpyfier.fit(downsampled_mocap)
    numpy_arrays = [track.values.values for track in downsampled_mocap]
    print(f"     Converted {len(numpy_arrays)} files")
    
    # Fit Scaler
    print("   → Fitting ListStandardScaler...")
    scaler = ListStandardScaler()
    scaler.fit(numpy_arrays)
    
    print(f"✅ Scaler fitted!")
    print(f"   Mean shape: {scaler.data_mean_.shape}")
    print(f"   Std shape: {scaler.data_std_.shape}")
    
    # --- BƯỚC 4: TẠO VÀ LƯU PIPELINE ---
    print(f"\n💾 Creating complete pipeline...")
    
    downsampler = DownSampler(2)
    full_pipeline = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
        ('const', const_remover),
        ('np', numpyfier),
        ('down', downsampler),
        ('stdscale', scaler)
    ])
    
    output_filename = "global_pipeline.pkl"
    joblib.dump(full_pipeline, output_filename)
    
    print(f"\n{'='*60}")
    print(f"🎉 SUCCESS!")
    print(f"{'='*60}")
    print(f"   Saved: {output_filename}")
    print(f"   Files processed: {len(all_mocap_objects)}/{total_files}")
    print(f"   Input features: {all_mocap_objects[0].values.shape[1]}")
    print(f"   After ConstantsRemover: {after_const[0].values.shape[1]}")
    print(f"   Final features: {scaler.data_mean_.shape[0]}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit global scaler with flexible processing modes'
    )
    parser.add_argument('--parent-dir', type=str, required=True,
                        help='Parent directory containing numbered folders')
    parser.add_argument('--folders', type=str,
                        help='Comma-separated list of specific folders to process')
    parser.add_argument('--start', type=int,
                        help='Start folder number (inclusive)')
    parser.add_argument('--end', type=int,
                        help='End folder number (inclusive)')
    parser.add_argument('--mode', type=str, default='hybrid',
                        choices=['hybrid', 'full_mp', 'sequential'],
                        help='Processing mode: hybrid (recommended), full_mp, or sequential')
    args = parser.parse_args()

    folders_list = None
    if args.folders:
        folders_list = [f.strip() for f in args.folders.split(',') if f.strip()]

    main_fit_scaler(
        args.parent_dir,
        folders=folders_list,
        start=args.start,
        end=args.end,
        mode=args.mode
    )