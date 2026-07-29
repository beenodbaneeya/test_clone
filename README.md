# PyTorch Scaling Test on Olivia

This project is a minimal starter for users who want to run quick scaling checks on **Olivia** with PyTorch.

The main goal is to verify that deep learning training scales when moving from:

- single GPU,
- multiple GPUs on one node,
- multiple nodes with multiple GPUs.

To keep the workflow simple and practical, this repo uses two model/data regimes:

- **WideResNet on CIFAR-100** for fast scaling runs (small images, shorter iteration time),
- **ViT on Tiny-ImageNet** in the training scripts for a larger vision workload.

In deep learning terms, these runs demonstrate **data-parallel training**: each GPU processes a different mini-batch shard, gradients are synchronized, and training throughput should increase as resources increase.

## Project Layout

```text
olivia_pytorch/
  scripts/
    train.py
    train_ddp.py
    train_utils.py
    dataset_utils.py
    device_utils.py
    model.py
  jobs/
    singlegpu.sh
    multigpu.sh
    multinode.sh
    logs/
  datasets/
```


## Quick Start

1. Clone the repository.
2. Edit `#SBATCH --account=...` (and other site-specific directives if needed) in the job script you want to run.
3. Submit from the `jobs/` directory:

```bash
cd jobs
sbatch singlegpu.sh
```

You can similarly run:

- `sbatch multigpu.sh`
- `sbatch multinode.sh`

## File Overview

### `jobs/`

- `jobs/singlegpu.sh`: single-node, single-GPU training run (baseline).
- `jobs/multigpu.sh`: single-node, multi-GPU DDP run.
- `jobs/multinode.sh`: multi-node DDP run with rendezvous setup.
- `jobs/logs/`: Slurm `.out/.err` and GPU utilization logs (`nvidia-smi`) are written here.

### `scripts/`

- `scripts/train.py`: single-GPU training entry point (argument parsing, setup, train/validate loop, throughput reporting).
- `scripts/train_ddp.py`: distributed training entry point using PyTorch DDP (`torchrun`, rank setup, global metric reduction).
- `scripts/train_utils.py`: core train/eval loop functions used by training scripts.
- `scripts/dataset_utils.py`: dataset download/loading and transforms for CIFAR-100 and Tiny-ImageNet.
- `scripts/device_utils.py`: compute device selection helper (CUDA vs CPU).
- `scripts/model.py`: model definitions (WideResNet and ViT wrapper).


## Notes on Scaling Interpretation

- `singlegpu.sh` gives a baseline for wall-clock time and throughput.
- `multigpu.sh` and `multinode.sh` test distributed scaling efficiency.
- For deep learning scaling studies, compare both:
  - **throughput** (images/second), and
  - **optimization behavior** (loss/accuracy trends),

because faster hardware utilization is useful only if model convergence remains healthy.
