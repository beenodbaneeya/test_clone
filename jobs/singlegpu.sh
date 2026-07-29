#!/bin/bash
#SBATCH --job-name=pytorch_singlegpu
#SBATCH --account=<project_number>
#SBATCH --output=logs/singlegpu_%j.out
#SBATCH --error=logs/singlegpu_%j.err
#SBATCH --time=00:50:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=1                    # Single compute node
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=16           # Reserve 16 CPU cores (Right-sized for WideResNet + CIFAR-100)
#SBATCH --mem=48G                    # Request 48 GB RAM (Right-sized for WideResNet + CIFAR-100)
#SBATCH --gpus-per-node=1            # Request 1 GPU
#SBATCH --reservation=software

# This will be replaced by new module
ml NRIS/GPU
ml use /cluster/projects/nn9999k/jorn/easybuild-gpu/modules/all
ml PyTorch/2.12.0


# Get the absolute path to the project directory.
PROJECT_DIR=$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)

# Training command
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train.py --model wideresnet --dataset cifar100 --seed 42 --batch-size 256 --epochs 100"

# Check GPU availability
echo "Checking GPU availability..."
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${PROJECT_DIR}/jobs/logs/singlegpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > "${GPU_LOG_FILE}" &
NVIDIA_MONITOR_PID=$!

# Run the training script
python $TRAINING_SCRIPT

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill "${NVIDIA_MONITOR_PID}" 2>/dev/null || true
