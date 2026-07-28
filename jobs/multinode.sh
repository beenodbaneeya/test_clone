#!/bin/bash
#SBATCH --job-name=pytorch_multinode
#SBATCH --account=hidden
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=accel           # GPU partition
#SBATCH --nodes=2                    # Request 2 compute nodes
#SBATCH --ntasks-per-node=1          # One task (process) on the node
#SBATCH --cpus-per-task=40           # Reserve 40 CPU cores (Right-sized for multi-node WideResNet)
#SBATCH --mem=128G                   # Request 128 GB RAM (Right-sized for multi-node WideResNet)
#SBATCH --gpus-per-node=4            # Number of GPUs per node
#SBATCH --reservation=software

ml purge
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
TRAINING_SCRIPT="${PROJECT_DIR}/scripts/train_ddp.py"
TRAINING_ARGS=(
  --model wideresnet
  --dataset cifar100
  --epochs 100
  --batch-size 2048
  --base-lr 0.02
  --target-accuracy 0.95
  --patience 2
  --seed 42
)

# Change working directory to project root
cd "${PROJECT_DIR}"



export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET


# Get head node IP
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"


# Start GPU utilization monitoring
GPU_LOG_FILE="${LOCAL_LOGS_DIR}/multinode.log"
echo "Starting GPU utilization monitoring..."
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > "${GPU_LOG_FILE}" &
NVIDIA_MONITOR_PID=$!

# Run training script with torchrun
srun torchrun \
  --nnodes="$SLURM_JOB_NUM_NODES" \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$head_node_ip:29500" \
  "${TRAINING_SCRIPT}" "${TRAINING_ARGS[@]}"

# Stop GPU utilization monitoring
echo "Stopping GPU utilization monitoring..."
kill "${NVIDIA_MONITOR_PID}" 2>/dev/null || true
