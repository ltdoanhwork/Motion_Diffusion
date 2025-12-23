import sys, os
import joblib
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse
import re
import gc

# --- CONFIG ---
PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------------------------------------------------------------
# HELPER CLASSES
# ------------------------------------------------------------------------------

class OnlineStatsCalculator:
    """Tính Mean/Std theo phương pháp trực tuyến (Streaming) để tiết kiệm RAM."""
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
    
    def update(self, batch_data):
        if batch_data.size == 0:
            return
            
        # Nếu data là (Batch, Time, Feat) hoặc (Time, Feat) -> Flatten về (N, Feat)
        if batch_data.ndim == 3:
            batch_data = batch_data.reshape(-1, batch_data.shape[-1])
        
        n_b = len(batch_data)
        if n_b == 0: return

        mean_b = np.mean(batch_data, axis=0)
        M2_b = np.sum((batch_data - mean_b) ** 2, axis=0)
        
        if self.n == 0:
            self.n = n_b
            self.mean = mean_b
            self.M2 = M2_b
        else:
            n_a = self.n
            mean_a = self.mean
            M2_a = self.M2
            
            delta = mean_b - mean_a
            
            # Update n
            self.n = n_a + n_b
            
            # Update mean
            self.mean = mean_a + delta * n_b / self.n
            
            # Update M2 (Welford's algorithm)
            self.M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b / self.n)

    def finalize(self):
        """Trả về mean và std cuối cùng."""
        if self.n < 2:
            if self.mean is None: return None, None
            return self.mean, np.ones_like(self.mean)
        
        variance = self.M2 / self.n
        std = np.sqrt(variance) + 1e-8  
        return self.mean, std

# ------------------------------------------------------------------------------
# WORKER FUNCTIONS
# ------------------------------------------------------------------------------

# Global vars cho multiprocessing
g_parser = None
g_partial_pipe = None

def init_worker():
    global g_parser, g_partial_pipe
    g_parser = BVHParser()
    
    # Pipeline sơ bộ: Chỉ biến đổi toạ độ, chưa cắt bớt hay scale
    g_partial_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
    ])

def worker_parse_and_transform(bvh_path):
    global g_parser, g_partial_pipe
    try:
        parsed_data = g_parser.parse(bvh_path)
        # Transform trả về list, lấy phần tử đầu tiên
        processed = g_partial_pipe.transform([parsed_data])[0]
        return processed
    except Exception as e:
        # print(f"Error: {e}")
        return None

def process_files_streaming(bvh_paths, stats_calculator, const_remover, 
                            downsample_rate, batch_size=100):
    num_cores = min(multiprocessing.cpu_count(), 16)
    
    # Sử dụng DownSampler chuẩn của pymo
    downsampler = DownSampler(downsample_rate)
    
    for i in tqdm(range(0, len(bvh_paths), batch_size), 
                  desc="Processing batches (Pass 2 - Statistics)", ncols=100):
        batch_paths = bvh_paths[i:i + batch_size]
        
        # 1. Parse & Partial Transform song song
        with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
            results = pool.map(worker_parse_and_transform, batch_paths)
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            continue
        
        # 2. Remove Constants (trên Main Process)
        after_const = const_remover.transform(valid_results)
        
        # 3. Downsample (trên Main Process) - Dùng Pymo DownSampler chuẩn
        downsampled = downsampler.transform(after_const)
        
        # 4. Convert to Numpy để tính toán Stats
        # Lưu ý: Pymo DownSampler trả về list các MocapData
        batch_arrays = [track.values.values for track in downsampled]
        
        if batch_arrays:
            batch_data = np.concatenate(batch_arrays, axis=0)
            stats_calculator.update(batch_data)
        
        # Giải phóng RAM
        del results, valid_results, after_const, downsampled, batch_arrays
        gc.collect()


def main_fit_scaler_efficient(parent_dir, folders=None, start=None, end=None, 
                              batch_size=100, sample_size_limit=1000,
                              downsample_rate=4):
    
    # --- 1. SCAN FILES ---
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
    print(f"🚀 MEMORY-EFFICIENT PIPELINE FITTER (BEAT Optimized)")
    print(f"   Folders: {len(to_process)}")
    print(f"   Downsample Rate: {downsample_rate} (Input 120fps -> Output {120/downsample_rate:.0f}fps)")
    print(f"{'='*60}")

    all_bvh_paths = []
    for folder in to_process:
        folder_path = os.path.join(parent_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        bvh_files = [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(".bvh")
        ]
        all_bvh_paths.extend(bvh_files)

    if not all_bvh_paths:
        print("❌ No BVH files found")
        return

    print(f"📂 Found total {len(all_bvh_paths)} BVH files")

    # --- 2. FIT CONSTANTS REMOVER (PASS 1) ---
    print(f"\n{'='*60}")
    actual_sample_size = min(len(all_bvh_paths), sample_size_limit) 
    
    print(f"🔹 PASS 1: Fitting ConstantsRemover")
    print(f"   Sampling: {actual_sample_size} random files")
    print(f"{'='*60}")
    
    sample_paths = np.random.choice(all_bvh_paths, actual_sample_size, replace=False)
    
    num_cores = min(multiprocessing.cpu_count(), 16)
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
        sample_results = list(tqdm(
            pool.imap(worker_parse_and_transform, sample_paths),
            total=len(sample_paths),
            desc="Sampling",
            ncols=100
        ))
    
    sample_mocap = [r for r in sample_results if r is not None]
    
    if not sample_mocap:
        print("❌ Failed to process sample data")
        return
    
    const_remover = ConstantsRemover()
    const_remover.fit(sample_mocap)
    print(f"✅ ConstantsRemover fitted. Removed dims: {len(const_remover.const_dims_)}")
    
    # Giữ lại một ít mẫu để fit Numpyfier sau này (tránh lỗi logic của pymo)
    sample_for_numpyfier_raw = sample_mocap[0:10] 

    del sample_results, sample_mocap
    gc.collect()

    # --- 3. COMPUTE STATISTICS (PASS 2 - STREAMING) ---
    print(f"\n{'='*60}")
    print(f"🔹 PASS 2: Computing Mean/Std on ALL {len(all_bvh_paths)} FILES")
    print(f"{'='*60}")
    
    stats_calculator = OnlineStatsCalculator()
    
    process_files_streaming(
        all_bvh_paths, 
        stats_calculator, 
        const_remover,
        downsample_rate, # Truyền rate vào đây
        batch_size=batch_size
    )
    
    mean, std = stats_calculator.finalize()
    
    print(f"\n📊 Statistics computed!")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Std shape: {std.shape}")

    # --- 4. BUILD & SAVE FINAL PIPELINE ---
    print(f"\n💾 Creating complete pipeline...")
    
    from pymo.preprocessing import ListStandardScaler
    scaler = ListStandardScaler()
    scaler.data_mean_ = mean
    scaler.data_std_ = std
    
    numpyfier = Numpyfier()
    # Trick: Fit numpyfier bằng dữ liệu mẫu nhỏ đã qua xử lý hằng số
    sample_transformed = const_remover.transform(sample_for_numpyfier_raw)
    
    # Quan trọng: Fit Numpyfier
    if sample_transformed:
        numpyfier.fit(sample_transformed)
    
    # --- PIPELINE CHUẨN CHO BEAT ---
    full_pipeline = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
        ('const', const_remover),
        ('down', DownSampler(downsample_rate)), # Downsample PHẢI TRƯỚC Numpyfier
        ('np', numpyfier),                      # Numpyfier chuyển về Matrix
        ('stdscale', scaler)                    # Scaler làm việc trên Matrix
    ])
    
    output_filename = "global_pipeline.pkl"
    joblib.dump(full_pipeline, output_filename)
    
    print(f"\n{'='*60}")
    print(f"🎉 SUCCESS! Pipeline saved to: {output_filename}")
    print(f"   Use this pipeline for 'process_data.py'")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-dir', type=str, required=True, help="Path to 'beat_english_v2.0.0'")
    parser.add_argument('--folders', type=str, help="Comma separated folders (e.g. '1,2,3')")
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--downsample', type=int, default=4, 
                        help="Downsample rate. If BEAT is 120fps: 4=30fps, 2=60fps, 1=120fps.")
    
    args = parser.parse_args()

    folders_list = None
    if args.folders:
        folders_list = [f.strip() for f in args.folders.split(',') if f.strip()]

    main_fit_scaler_efficient(
        args.parent_dir,
        folders=folders_list,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        downsample_rate=args.downsample
    )