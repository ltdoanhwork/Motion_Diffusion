import sys, os
import joblib
import numpy as np
from tqdm import tqdm
import multiprocessing
import argparse
import re
import gc

PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class OnlineStatsCalculator:
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
    
    def update(self, batch_data):
        if batch_data.size == 0:
            return
            
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
            
            # Update M2
            self.M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b / self.n)

    def finalize(self):
        """Trả về mean và std cuối cùng."""
        if self.n < 2:
            if self.mean is None: return None, None
            return self.mean, np.ones_like(self.mean)
        
        variance = self.M2 / self.n
        std = np.sqrt(variance) + 1e-8  
        return self.mean, std


class FixedDownSampler(BaseEstimator, TransformerMixin):
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


g_parser = None
g_partial_pipe = None

def init_worker():
    global g_parser, g_partial_pipe
    g_parser = BVHParser()
    g_partial_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('absolute_translation_deltas')),
    ])

def worker_parse_and_transform(bvh_path):
    global g_parser, g_partial_pipe
    try:
        parsed_data = g_parser.parse(bvh_path)
        processed = g_partial_pipe.transform([parsed_data])[0]
        return processed
    except Exception as e:
        return None


def process_files_streaming(bvh_paths, stats_calculator, const_remover, 
                            downsampler, batch_size=100):
    num_cores = min(multiprocessing.cpu_count(), 16)
    
    for i in tqdm(range(0, len(bvh_paths), batch_size), 
                  desc="Processing batches (Pass 2 - Statistics)", ncols=100):
        batch_paths = bvh_paths[i:i + batch_size]
        
        with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
            results = pool.map(worker_parse_and_transform, batch_paths)
        
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            continue
        
        after_const = const_remover.transform(valid_results)
        
        downsampled = downsampler.transform(after_const)
        
        batch_arrays = [track.values.values for track in downsampled]
        if batch_arrays:
            batch_data = np.concatenate(batch_arrays, axis=0)
            stats_calculator.update(batch_data)
        
        del results, valid_results, after_const, downsampled, batch_arrays
        gc.collect()


def main_fit_scaler_efficient(parent_dir, folders=None, start=None, end=None, 
                              batch_size=100, sample_size_limit=1000):
    
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
        print(" No folders found")
        return

    print(f"\n{'='*60}")
    print(f" MEMORY-EFFICIENT MODE (SAFE FOR LARGE DATASET)")
    print(f" Folders to process: {len(to_process)}")
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
        print(" No BVH files found")
        return

    print(f" Found total {len(all_bvh_paths)} BVH files")

    print(f"\n{'='*60}")
    actual_sample_size = min(len(all_bvh_paths), sample_size_limit) 
    
    print(f" PASS 1: Fitting ConstantsRemover")
    print(f"   Sampling: {actual_sample_size} random files (checking for dead joints)")
    print(f"   Note: This step only detects structural constants.")
    print(f"{'='*60}")
    
    sample_paths = np.random.choice(all_bvh_paths, actual_sample_size, replace=False)
    
    num_cores = min(multiprocessing.cpu_count(), 16)
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
        sample_results = list(tqdm(
            pool.imap(worker_parse_and_transform, sample_paths),
            total=len(sample_paths),
            desc="Sampling for ConstantsRemover",
            ncols=100
        ))
    
    sample_mocap = [r for r in sample_results if r is not None]
    
    if not sample_mocap:
        print(" Failed to process sample data")
        return
    
    # Fit ConstantsRemover
    const_remover = ConstantsRemover()
    const_remover.fit(sample_mocap)
    print(f" ConstantsRemover fitted.")
    print(f"   Constant columns removed: {len(const_remover.const_dims_)}")
    
    sample_for_numpyfier_raw = sample_mocap[0:10] 

    del sample_results, sample_mocap
    gc.collect()
    print(f"\n{'='*60}")
    print(f" PASS 2: Computing Mean/Std on ALL {len(all_bvh_paths)} FILES")
    print(f"   Note: Computing exact statistics on 100% of data.")
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
    
    mean, std = stats_calculator.finalize()
    
    print(f"\n Statistics computed!")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Std shape: {std.shape}")
    print(f"\n Creating complete pipeline...")
    
    from pymo.preprocessing import ListStandardScaler
    scaler = ListStandardScaler()
    scaler.data_mean_ = mean
    scaler.data_std_ = std
    
    numpyfier = Numpyfier()
    sample_transformed = const_remover.transform(sample_for_numpyfier_raw)
    if sample_transformed:
        numpyfier.fit(sample_transformed)
    
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
    print(f"   Total Files processed: {len(all_bvh_paths)}")
    print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-dir', type=str, required=True)
    parser.add_argument('--folders', type=str)
    parser.add_argument('--start', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--sample-size', type=int, default=1000, 
                        help='Number of files to sample for ConstantsRemover (Pass 1). Default: 1000')
    
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
        sample_size_limit=args.sample_size
    )