#!/bin/bash
#SBATCH --job-name=motion_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --partition=a100
#SBATCH --time=72:00:00
#SBATCH --output=motion_ddp_%j.out
#SBATCH --error=motion_ddp_%j.err

# ============================================================
# Multi-GPU Motion Diffusion Training Script
# For 4x A100 80GB GPUs with SLURM
# ============================================================

echo "==============================================================================="
echo "Job started on $(date)"
echo "Job ID          : $SLURM_JOB_ID"
echo "Job name        : $SLURM_JOB_NAME"
echo "WorkDir         : $(pwd)"
echo "Partition       : $SLURM_JOB_PARTITION"
echo "Num GPUs        : 4 x A100 80GB"
echo "Hosts allocated : $SLURM_NODELIST"
echo "==============================================================================="

# Load modules (adjust for your cluster)
# module load cuda/11.8
# module load python/3.10

# Activate conda environment if needed
# source ~/.bashrc
# conda activate motion_diffusion

# Set environment variables for DDP
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=4
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Change to project directory
cd /home/serverai/ltdoanh/Motion_Diffusion

# ============================================================
# Training Configuration
# ============================================================
DATASET="beat"
NAME="beat_improved_v1"
EPOCHS=300
BATCH_SIZE=32                    # Per GPU, total effective = 32*4 = 128
LR=1e-4
BETA_SCHEDULE="cosine"

# Motion losses
USE_VELOCITY="--use_velocity_loss"
USE_GEOMETRIC="--use_geometric_loss"
VELOCITY_WEIGHT=0.5
GEOMETRIC_WEIGHT=0.3

# Training improvements
EMA_DECAY=0.9999
CFG_DROPOUT=0.1                  # Classifier-free guidance training
GRAD_ACCUM=1                     # Gradient accumulation steps
# USE_AMP="--use_amp"            # Uncomment to enable mixed precision

# ============================================================
# Run Training with torchrun (DDP)
# ============================================================
echo ""
echo "🚀 Starting DDP Training..."
echo "   Dataset: $DATASET"
echo "   Epochs: $EPOCHS"
echo "   Batch size per GPU: $BATCH_SIZE"
echo "   Total effective batch size: $((BATCH_SIZE * 4))"
echo "   Beta schedule: $BETA_SCHEDULE"
echo ""

torchrun \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    tools/train_ddp.py \
    --dataset_name $DATASET \
    --name $NAME \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --beta_schedule $BETA_SCHEDULE \
    $USE_VELOCITY \
    $USE_GEOMETRIC \
    --velocity_weight $VELOCITY_WEIGHT \
    --geometric_weight $GEOMETRIC_WEIGHT \
    --ema_decay $EMA_DECAY \
    --cfg_dropout $CFG_DROPOUT \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --log_every 50 \
    --save_every_e 10

echo ""
echo "==============================================================================="
echo "Job finished on $(date)"
echo "==============================================================================="
