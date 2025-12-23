import os

base_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT"
out_dir = "/home/serverai/ltdoanh/Motion_Diffusion/datasets/BEAT_numpy"

print(f"Đang chạy step1_fit_scaler.py...")
command1 = f'python "./datasets/step1_fit_scaler.py" --parent-dir "./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1" --folders "1"'
os.system(command1)

print("Đang chạy preprocess_data.py...")
command2 = f'python "./datasets/preprocess_data.py" --parent-dir "./datasets/BEAT/beat_english_v0.2.1/beat_english_v0.2.1" --out-root "./datasets/BEAT_171dims/" --pipeline "./global_pipeline.pkl"  --folders "1"'
os.system(command2)

# print("Đang chạy train_vq_with_discriminator.py...") tmux a -t 5
# command3 = f'python ./tools/train_vq_with_discriminator.py \
#                 --name VQKL_Sobolev \
#                 --max_epoch 50 \
#                 --batch_size 32 \
#                 --lr 4.5e-6 \
#                 --use_hierarchical_loss \
#                 --hand_loss_weight 10.0 \
#                 --lambda_vel 0.5 \
#                 --lambda_acc 0.5 \
#                 --lambda_spectral 0.1 \
#                 --use_bone_loss \
#                 --lambda_bone 0.15 \
#                 --disc_start 7500'

# os.system(command3)

# print("Đang chạy scale_factor.py...")
# command4 = f'python ./tools/scale_factor.py \
#             --vqvae_name VQKL_Sobolev'

# os.system(command4)

# scale_factor_path = "./checkpoints/beat/VQKL_Sobolev/scale_factor.txt" 
# scale_val = None

# print(f"Đang đọc giá trị từ {scale_factor_path}...")

# if os.path.exists(scale_factor_path):
#     with open(scale_factor_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             if line.strip().startswith("scale_factor"):
#                 parts = line.split('=')
#                 if len(parts) > 1:
#                     scale_val = parts[1].strip() 
#                     print(f"--> Tìm thấy scale_factor: {scale_val}")
#                 break
# else:
#     print(f"LỖI: Không tìm thấy file {scale_factor_path}. Vui lòng kiểm tra lại đường dẫn.")
#     exit(1) 

# if scale_val is None:
#     print("LỖI: Không đọc được giá trị scale_factor trong file.")
#     exit(1)

# print("Đang chạy train_vq_diffusion.py...")

# command5 = f'python tools/train_vq_diffusion.py \
#             --dataset_name beat \
#             --name vqkl_diff \
#             --vqkl_name VQKL_Sobolev \
#             --batch_size 64 \
#             --max_epoch 200 \
#             --lr 1e-4 \
#             --weight_decay 0.01 \
#             --dropout 0.1 \
#             --grad_clip 1.0 \
#             --loss_type rescaled_mse \
#             --scale_factor {scale_val} \
#             --noise_schedule linear \
#             --schedule_sampler uniform \
#             --use_kl_posterior \
#             --fft_loss_weight 0.05 \
#             --hand_loss_weight 2.0 \
#             --hand_boost_factor 1.0 \
#             --sobolev_loss_weight 0.1 \
#             --sobolev_depth 2 \
#             --physical_loss_weight 1.0 \
#             --foot_contact_threshold 0.005 \
#             --use_bone_loss \
#             --lambda_bone 0.5'

# print(f"Command thực thi: {command5}")
# os.system(command5)

