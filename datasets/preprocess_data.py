import sys
import os
import numpy as np
import multiprocessing
from textgrid import TextGrid
from sklearn.pipeline import Pipeline

PYOM_DIR = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/pymo"
if PYOM_DIR not in sys.path:
    sys.path.insert(0, PYOM_DIR)

from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *

def time_to_frame(t, fps=60):  
    return int(round(t * fps))

def extract_sentences_with_text(textgrid_path, motion_data, output_dir, fps=30, pause_threshold=0.5,
    split_parts=1, use_first_part_only=True
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

def preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir):
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
                use_first_part_only=True
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
        
        preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir)
        
        print(f"<-- [Done] Finished folder {i}")
        return f"Folder {i}: Success"
    except Exception as e:
        print(f"!!! Error in folder {i}: {e}")
        return f"Folder {i}: Failed - {e}"

def main():
    folder_indices = range(1, 31)
    
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_workers} processes for {len(folder_indices)} folders.")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_folder_wrapper, folder_indices)
        
    print("Results:", results)

if __name__ == "__main__":
    print("=============================================================================================== ")
    main()
    print("=============================================================================================== ")
    print("Preprocessing completed.")