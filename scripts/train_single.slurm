#!/bin/bash
#SBATCH --job-name=motion_single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --output=motion_single_%j.out
#SBATCH --error=motion_single_%j.err
#SBATCH --partition=a100

# ============================================================
# Single GPU Motion Diffusion Training Script (Improved)
# For testing or single GPU training
# ============================================================

echo "==============================================================================="
echo "Job started on $(date)"
echo "Job ID          : $SLURM_JOB_ID"
echo "==============================================================================="

cd /home/serverai/ltdoanh/Motion_Diffusion

# ============================================================
# Training with improvements (single GPU)
# ============================================================
python tools/train.py \
    --dataset_name beat \
    --name beat_improved_single \
    --num_epochs 200 \
    --batch_size 64 \
    --lr 2e-4 \
    --beta_schedule cosine \
    --use_velocity_loss \
    --use_geometric_loss \
    --velocity_weight 0.5 \
    --geometric_weight 0.3 \
    --ema_decay 0.9999 \
    --cfg_dropout 0.1 \
    --log_every 50 \
    --save_every_e 5

echo "==============================================================================="
echo "Job finished on $(date)"
echo "==============================================================================="
