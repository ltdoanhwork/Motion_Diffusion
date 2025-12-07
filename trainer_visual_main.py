import os

base_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT"
out_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT_numpy"

# print(f"Đang chạy step1_fit_scaler.py...")
# command1 = f'python "./datasets/step1_fit_scaler.py" --parent-dir "{base_dir}" --start 1 --end 30 --batch-size 32'
# os.system(command1)

# for i in range(1, 31):
#     print("Đang chạy preprocess_data.py...")
#     command2 = f'python "./datasets/preprocess_data.py" --parent-dir "{base_dir}" --out-root "{out_dir}" --folders "{i}" --fps 60'
#     os.system(command2)

print("Đang chạy train_vq_with_discriminator.py...")
command3 = f'python ./tools/train_vq_with_discriminator.py \
            --name VQKL_GAN_BEAT_SO \
            --batch_size 32 \
            --max_epoch 50 \
            --disc_start 7500 \
            --disc_weight 0.75 \
            --perceptual_weight 0.1 \
            --kl_weight 1e-6'

os.system(command3)

print("Đang chạy scale_factor.py...")
command4 = f'python ./tools/scale_factor.py \
            --vqvae_name VQKL_GAN_BEAT_SO'

os.system(command4)

scale_factor_path = "./checkpoints/beat/VQKL_GAN_BEAT_SO/scale_factor.txt" 
scale_val = None

print(f"Đang đọc giá trị từ {scale_factor_path}...")

if os.path.exists(scale_factor_path):
    with open(scale_factor_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip().startswith("scale_factor"):
                parts = line.split('=')
                if len(parts) > 1:
                    scale_val = parts[1].strip() 
                    print(f"--> Tìm thấy scale_factor: {scale_val}")
                break
else:
    print(f"LỖI: Không tìm thấy file {scale_factor_path}. Vui lòng kiểm tra lại đường dẫn.")
    exit(1) 

if scale_val is None:
    print("LỖI: Không đọc được giá trị scale_factor trong file.")
    exit(1)

print("Đang chạy train_vq_diffusion.py...")

command5 = f'python tools/train_vq_diffusion.py \
            --dataset_name beat \
            --name vqkl_diffusion_hierarchical \
            --vqkl_name VQKL_GAN_BEAT_SO \
            --batch_size 64 \
            --max_epoch 50 \
            --lr 5e-5 \
            --weight_decay 0.0 \
            --grad_clip 0.5 \
            --loss_type l1 \
            --scale_factor {scale_val} \
            --noise_schedule linear \
            --use_kl_posterior \
            --hand_loss_weight 10.0 \
            --hand_boost_factor 5.0 \
            --sobolev_loss_weight 1.0 \
            --sobolev_depth 2'

print(f"Command thực thi: {command5}")
os.system(command5)

