import os

base_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT"
out_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT_numpy"

# print("Đang chạy step1_fit_scaler.py...")
# command1 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/datasets/step1_fit_scaler.py" --parent-dir "{base_dir}" --start 1 --end 30 --batch-size 32'
# os.system(command1)

# start = 5
# end = 31
# for i in range(start, end):
#     print("Đang chạy preprocess_data.py...")
#     command2 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/datasets/preprocess_data.py" --parent-dir "{base_dir}" --out-root "{out_dir}" --folders "{i}"'
#     os.system(command2)

# print("Đang chạy train_vq.py...")
# command3 = f'python "/home/serverai/ltdoanh/Motion_Diffusion/tools/train_vq.py" --dataset_name beat --codebook_size 512'
# os.system(command3)

print("Đang chạy train_vq_diffusion.py...")

command4 = f'python tools/train_vq_diffusion.py \
            --name vqkl_diffusion \
            --vqkl_name VQKL_GAN_BEAT \
            --scale_factor 1.39714289 \
            --use_kl_posterior \
            --hand_boost_factor 2.0'

os.system(command4)