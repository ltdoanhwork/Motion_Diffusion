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
    Trả về MocapData object (KHÔNG phải numpy) để có thể fit ConstantsRemover sau
    """
    global g_parser, g_partial_pipe

    g_parser = BVHParser()
    
    # Pipeline chỉ tới TRƯỚC ConstantsRemover
    g_partial_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
        # DỪNG Ở ĐÂY - chưa có ConstantsRemover, Numpyfier, DownSampler
    ])

def worker_parse_and_transform(bvh_path):
    """
    Worker: Parse + transform tới trước ConstantsRemover
    Trả về: MocapData object (DataFrame)
    """
    global g_parser, g_partial_pipe
    
    try:
        # Parse
        parsed_data = g_parser.parse(bvh_path)
        
        # Transform qua các bước đầu
        # Kết quả vẫn là MocapData object với DataFrame
        processed = g_partial_pipe.transform([parsed_data])[0]
        
        return processed
        
    except Exception as e:
        print(f"    ❌ {os.path.basename(bvh_path)}: {e}")
        return None

def main_fit_scaler(parent_dir, folders=None, start=None, end=None):
    """
    CHIẾN LƯỢC TỐI ƯU:
    1. Multiprocessing: Parse + transform TẤT CẢ tới TRƯỚC ConstantsRemover
       → Kết quả: List[MocapData] với DataFrame
    2. Fit ConstantsRemover trên TẤT CẢ MocapData objects (nhanh vì không parse)
    3. Apply các bước còn lại + fit Scaler
    """
    
    # --- THU THẬP FILE BVH ---
    print(f"\n🔍 Scanning: {parent_dir}")
    
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

    all_bvh_paths = []
    for folder in to_process:
        src = os.path.join(parent_dir, folder)
        if not os.path.isdir(src):
            continue
        print(f"  Scanning folder {folder}...")
        for fname in os.listdir(src):
            if fname.endswith(".bvh"):
                all_bvh_paths.append(os.path.join(src, fname))

    if not all_bvh_paths:
        print("❌ No BVH files found")
        return

    print(f"\n✅ Found {len(all_bvh_paths)} BVH files")

    # --- BƯỚC 1: MULTIPROCESSING PARSE + PARTIAL TRANSFORM ---
    print(f"\n🚀 PASS 1: Multiprocessing parse + transform (to before ConstantsRemover)...")
    
    num_cores = multiprocessing.cpu_count()
    print(f"   Using {num_cores} CPU cores")
    
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
        results = list(tqdm(
            pool.imap(worker_parse_and_transform, all_bvh_paths),
            total=len(all_bvh_paths),
            desc="Pass 1: Parsing & transforming"
        ))
    
    # Lọc kết quả hợp lệ (MocapData objects)
    mocap_objects = [r for r in results if r is not None]
    
    if not mocap_objects:
        print("❌ No valid data after Pass 1")
        return
    
    print(f"✅ Pass 1 done: {len(mocap_objects)}/{len(all_bvh_paths)} files successful")
    print(f"   Sample shape: {mocap_objects[0].values.shape}")
    
    # --- BƯỚC 2: FIT ConstantsRemover TRÊN TẤT CẢ DỮ LIỆU ---
    print(f"\n📊 PASS 2: Fitting ConstantsRemover on ALL {len(mocap_objects)} files...")
    print("   (This is fast - no parsing, just computing std on DataFrames)")
    
    const_remover = ConstantsRemover()
    const_remover.fit(mocap_objects)
    
    print(f"✅ ConstantsRemover fitted!")
    print(f"   Constant columns found: {len(const_remover.const_dims_)}")
    if const_remover.const_dims_:
        print(f"   Examples: {list(const_remover.const_dims_)[:5]}")
    
    # --- BƯỚC 3: APPLY CÁC BƯỚC CÒN LẠI + FIT SCALER ---
    print(f"\n📈 PASS 3: Applying remaining steps + fitting Scaler...")
    
    # Apply ConstantsRemover
    print("   Applying ConstantsRemover...")
    after_const = const_remover.transform(mocap_objects)
    print(f"   Shape after removing constants: {after_const[0].values.shape}")
    
    # Apply DownSampler (dùng FixedDownSampler)
    print("   Downsampling by factor of 2...")
    downsampler = FixedDownSampler(2)
    downsampled_mocap = downsampler.transform(after_const)
    print(f"   Shape after downsampling: {downsampled_mocap[0].values.shape}")
    
    # Convert to numpy arrays
    print("   Converting to numpy arrays...")
    numpyfier = Numpyfier()
    numpyfier.fit(downsampled_mocap)  # Cần fit để lưu org_mocap_
    
    # Convert manually to list of arrays (không stack)
    numpy_arrays = [track.values.values for track in downsampled_mocap]
    print(f"   Converted {len(numpy_arrays)} files")
    
    # Fit Scaler
    print("   Fitting ListStandardScaler...")
    scaler = ListStandardScaler()
    scaler.fit(numpy_arrays)
    
    print(f"✅ Scaler fitted!")
    print(f"   Mean shape: {scaler.data_mean_.shape}")
    print(f"   Std shape: {scaler.data_std_.shape}")
    
    # --- BƯỚC 4: TẠO VÀ LƯU PIPELINE HOÀN CHỈNH ---
    print(f"\n💾 Creating complete pipeline...")
    
    # Tạo DownSampler wrapper để pipeline hoàn chỉnh
    # (Trong thực tế, downsampling đã được áp dụng thủ công ở trên)
    downsampler = DownSampler(2)
    
    full_pipeline = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
        ('const', const_remover),      # ✅ Fitted
        ('np', numpyfier),              # ✅ Fitted
        ('down', downsampler),          # ⚠️ Wrapper (manual downsampling applied)
        ('stdscale', scaler)            # ✅ Fitted
    ])
    
    output_filename = "global_pipeline.pkl"
    joblib.dump(full_pipeline, output_filename)
    
    print(f"\n🎉 SUCCESS!")
    print(f"   Saved: {output_filename}")
    print(f"   Files processed: {len(mocap_objects)}/{len(all_bvh_paths)}")
    print(f"   Input features: {mocap_objects[0].values.shape[1]}")
    print(f"   After ConstantsRemover: {after_const[0].values.shape[1]}")
    print(f"   Final features: {scaler.data_mean_.shape[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit global scaler (OPTIMIZED multiprocessing)'
    )
    parser.add_argument('--parent-dir', type=str, required=True)
    parser.add_argument('--folders', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    args = parser.parse_args()

    folders_list = None
    if args.folders:
        folders_list = [f.strip() for f in args.folders.split(',') if f.strip()]

    main_fit_scaler(
        args.parent_dir,
        folders=folders_list,
        start=args.start,
        end=args.end
    )