import sys, os
import joblib
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse
import re

PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class OnlineStatsCalculator:
    """
    Tính mean/std theo cách streaming (Welford's algorithm)
    để tránh load toàn bộ data vào RAM.
    """
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
    
    def update(self, batch_data):
        """
        Update statistics với một batch mới.
        batch_data: numpy array shape (n_samples, n_features)
        """
        if batch_data.size == 0:
            return
            
        # Flatten nếu là 3D (batch, time, features) -> (n_samples, features)
        if batch_data.ndim == 3:
            batch_data = batch_data.reshape(-1, batch_data.shape[-1])
        
        for sample in batch_data:
            self.n += 1
            if self.mean is None:
                self.mean = np.zeros_like(sample, dtype=np.float64)
                self.M2 = np.zeros_like(sample, dtype=np.float64)
            
            delta = sample - self.mean
            self.mean += delta / self.n
            delta2 = sample - self.mean
            self.M2 += delta * delta2
    
    def finalize(self):
        """Trả về mean và std cuối cùng."""
        if self.n < 2:
            return self.mean, np.ones_like(self.mean)
        
        variance = self.M2 / self.n
        std = np.sqrt(variance) + 1e-8  # Thêm epsilon để tránh chia 0
        return self.mean, std


class FixedDownSampler(BaseEstimator, TransformerMixin):
    """DownSampler tương thích với MocapData."""
    def __init__(self, rate):
        self.rate = rate
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Q = []
        for track in X:
            new_track = track.clone()
            new_track.values = track.values.iloc[::self.rate]
            Q.append(new_track)
        return Q
    
    def inverse_transform(self, X, copy=None):
        return X


# --- WORKER CHO MULTIPROCESSING ---
g_parser = None
g_partial_pipe = None

def init_worker():
    """Khởi tạo parser và pipeline cho worker."""
    global g_parser, g_partial_pipe
    g_parser = BVHParser()
    g_partial_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
    ])

def worker_parse_and_transform(bvh_path):
    """Worker: Parse + transform BVH file."""
    global g_parser, g_partial_pipe
    try:
        parsed_data = g_parser.parse(bvh_path)
        processed = g_partial_pipe.transform([parsed_data])[0]
        return processed
    except Exception:
        return None


def process_files_streaming(bvh_paths, stats_calculator, const_remover, 
                            downsampler, batch_size=100):
    """
    Xử lý files theo batch và update statistics streaming.
    Không giữ toàn bộ data trong RAM.
    
    Args:
        bvh_paths: List đường dẫn files BVH
        stats_calculator: OnlineStatsCalculator instance
        const_remover: ConstantsRemover đã fit
        downsampler: DownSampler instance
        batch_size: Số files xử lý mỗi lần
    """
    num_cores = min(multiprocessing.cpu_count(), 8)  # Giới hạn cores
    
    for i in tqdm(range(0, len(bvh_paths), batch_size), 
                  desc="Processing batches", ncols=100):
        batch_paths = bvh_paths[i:i + batch_size]
        
        # Multiprocessing cho batch này
        with multiprocessing.Pool(processes=num_cores, 
                                 initializer=init_worker) as pool:
            results = pool.map(worker_parse_and_transform, batch_paths)
        
        # Lọc kết quả hợp lệ
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            continue
        
        # Apply ConstantsRemover
        after_const = const_remover.transform(valid_results)
        
        # Apply DownSampler
        downsampled = downsampler.transform(after_const)
        
        # Convert sang numpy và update stats
        batch_arrays = [track.values.values for track in downsampled]
        if batch_arrays:
            batch_data = np.concatenate(batch_arrays, axis=0)  # (total_frames, features)
            stats_calculator.update(batch_data)
        
        # Xóa để giải phóng RAM
        del results, valid_results, after_const, downsampled, batch_arrays
        import gc
        gc.collect()


def main_fit_scaler_efficient(parent_dir, folders=None, start=None, end=None, 
                              batch_size=100):
    """
    Fit scaler với memory-efficient approach.
    
    CHIẾN LƯỢC:
    1. Pass đầu: Fit ConstantsRemover (cần toàn bộ data để xác định constant dims)
    2. Pass hai: Tính mean/std theo streaming (không cần load toàn bộ vào RAM)
    """
    
    # --- THU THẬP FOLDERS ---
    if folders:
        to_process = folders
    else:
        entries = [d for d in os.listdir(parent_dir) 
                  if os.path.isdir(os.path.join(parent_dir, d))]
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
    print(f"🎯 MEMORY-EFFICIENT MODE")
    print(f"📂 Folders to process: {len(to_process)}")
    print(f"{'='*60}")

    # --- THU THẬP TẤT CẢ FILES ---
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

    print(f"✅ Found {len(all_bvh_paths)} BVH files")

    # ============================================================
    # PASS 1: FIT ConstantsRemover (cần sample nhỏ, không phải toàn bộ)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"🔍 PASS 1: Fitting ConstantsRemover (sampling 1000 files)")
    print(f"{'='*60}")
    
    # Lấy sample để fit ConstantsRemover (không cần toàn bộ data)
    sample_size = min(1000, len(all_bvh_paths))
    sample_paths = np.random.choice(all_bvh_paths, sample_size, replace=False)
    
    num_cores = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(processes=num_cores, 
                             initializer=init_worker) as pool:
        sample_results = list(tqdm(
            pool.imap(worker_parse_and_transform, sample_paths),
            total=len(sample_paths),
            desc="Sampling for ConstantsRemover",
            ncols=100
        ))
    
    sample_mocap = [r for r in sample_results if r is not None]
    
    if not sample_mocap:
        print("❌ Failed to process sample data")
        return
    
    # Fit ConstantsRemover
    const_remover = ConstantsRemover()
    const_remover.fit(sample_mocap)
    print(f"✅ ConstantsRemover fitted on {len(sample_mocap)} samples")
    print(f"   Constant columns: {len(const_remover.const_dims_)}")
    
    del sample_results, sample_mocap
    import gc
    gc.collect()

    # ============================================================
    # PASS 2: TÍNH MEAN/STD THEO STREAMING
    # ============================================================
    print(f"\n{'='*60}")
    print(f"📊 PASS 2: Computing mean/std (streaming mode)")
    print(f"   Batch size: {batch_size} files")
    print(f"{'='*60}")
    
    downsampler = FixedDownSampler(2)
    stats_calculator = OnlineStatsCalculator()
    
    process_files_streaming(
        all_bvh_paths, 
        stats_calculator, 
        const_remover,
        downsampler,
        batch_size=batch_size
    )
    
    # Finalize statistics
    mean, std = stats_calculator.finalize()
    
    print(f"\n✅ Statistics computed!")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Std shape: {std.shape}")

    # ============================================================
    # PASS 3: TẠO VÀ LƯU PIPELINE
    # ============================================================
    print(f"\n💾 Creating complete pipeline...")
    
    # Tạo scaler từ mean/std đã tính
    from pymo.preprocessing import ListStandardScaler
    scaler = ListStandardScaler()
    scaler.data_mean_ = mean
    scaler.data_std_ = std
    
    # Tạo Numpyfier (cần fit một lần)
    numpyfier = Numpyfier()
    # Fit với sample nhỏ
    sample_for_numpyfier = const_remover.transform(
        [sample_mocap[0]] if 'sample_mocap' in locals() else []
    )
    if sample_for_numpyfier:
        numpyfier.fit(sample_for_numpyfier)
    
    full_pipeline = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
        ('const', const_remover),
        ('np', numpyfier),
        ('down', DownSampler(2)),
        ('stdscale', scaler)
    ])
    
    output_filename = "global_pipeline.pkl"
    joblib.dump(full_pipeline, output_filename)
    
    print(f"\n{'='*60}")
    print(f"🎉 SUCCESS!")
    print(f"{'='*60}")
    print(f"   Saved: {output_filename}")
    print(f"   Files processed: {len(all_bvh_paths)}")
    print(f"   Final features: {mean.shape[0]}")
    print(f"   Memory usage: STREAMING (no full load)")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit global scaler with streaming (memory-efficient)'
    )
    parser.add_argument('--parent-dir', type=str, required=True,
                       help='Parent directory containing numbered folders')
    parser.add_argument('--folders', type=str,
                       help='Comma-separated list of specific folders')
    parser.add_argument('--start', type=int,
                       help='Start folder number (inclusive)')
    parser.add_argument('--end', type=int,
                       help='End folder number (inclusive)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of files to process per batch (default: 100)')
    
    args = parser.parse_args()

    folders_list = None
    if args.folders:
        folders_list = [f.strip() for f in args.folders.split(',') if f.strip()]

    main_fit_scaler_efficient(
        args.parent_dir,
        folders=folders_list,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size
    )