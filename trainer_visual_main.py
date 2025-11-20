import os

base_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT"
out_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT_numpy"

print("Đang chạy step1_fit_scaler.py...")
command1 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/datasets/step1_fit_scaler.py" --parent-dir "{base_dir}" --start 1 --end 30 --mode hybrid'
os.system(command1)

print("Đang chạy preprocess_data.py...")
command2 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/datasets/preprocess_data.py" --parent-dir "{base_dir}" --start 1 --end 30 --mode hybrid'
os.system(command2)

print("Đang chạy train_vq.py...")
command3 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/tools/train_vq.py" --dataset_name beat --codebook_size 512'
os.system(command3)

print("Đang chạy train_vq_diffusion.py...")
command4 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/tools/train_vq_diffusion.py" --dataset_name beat --vqvae_name VQVAE_BEAT'
os.system(command4)