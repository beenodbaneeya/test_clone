#!/bin/bash
#SBATCH --job-name=pytorch_singlegpu
#SBATCH --account=hidden
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

ml NRIS/GPU
ml use /cluster/projects/hidden/jorn/easybuild-gpu/modules/all
ml PyTorch/2.12.0


# Resolve script and project directories independent of submit location.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

# Ensure logs are always written under jobs/logs
LOCAL_LOGS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOCAL_LOGS_DIR}"

# Training command
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train.py"
TRAINING_ARGS=(
  --model wideresnet
  --dataset cifar100
  --seed 42
  --batch-size 256
  --epochs 100
)

# Change working directory to project root
cd "${PROJECT_DIR}"


# Check GPU availability
echo "Checking GPU availability..."
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())'

# Start GPU utilization monitoring in the background
GPU_LOG_FILE="${LOCAL_LOGS_DIR}/singlegpu.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > "${GPU_LOG_FILE}" &
NVIDIA_MONITOR_PID=$!

# Run the training script
python "${TRAINING_SCRIPT}" "${TRAINING_ARGS[@]}"

# Stop GPU utilization monitoring specifically by PID
echo "Stopping GPU utilization monitoring..."
kill "${NVIDIA_MONITOR_PID}"
