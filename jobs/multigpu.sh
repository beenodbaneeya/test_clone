#!/bin/bash
#SBATCH --job-name=pytorch_multigpu
#SBATCH --account=<project_number>
#SBATCH --output=logs/multigpu_%j.out
#SBATCH --error=logs/multigpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel            # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=40           # Reserve 40 CPU cores 
#SBATCH --mem=128G                   # Request 128 GB RAM
#SBATCH --gpus=4                     # Request 4 GPU


ml NRIS/GPU
ml PyTorch/2.12.0

# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Training command for wideresnet on CIFAR-100 dataset with DDP (Distributed Data Parallel)
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py --model wideresnet --dataset cifar100 --batch-size 1024 --epochs 100 --base-lr 0.04 --target-accuracy 0.95 --patience 2 --seed 42"

# Training command for ViT on Tiny-ImageNet dataset with DDP (Distributed Data Parallel)
# TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py --model vit --dataset tiny-imagenet --batch-size 1024 --epochs 100 --optimizer adamw --base-lr 0.0003 --target-accuracy 0.95 --patience 2 --seed 42 --num-workers 8 --amp"

# Check GPU availability
echo "Checking GPU availability ..."
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/multigpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > "${GPU_LOG_FILE}" &
NVIDIA_MONITOR_PID=$!

# Run the training script with torchrun 
torchrun --standalone --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node="$SLURM_GPUS_ON_NODE" $TRAINING_SCRIPT

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill "${NVIDIA_MONITOR_PID}" 2>/dev/null || true
