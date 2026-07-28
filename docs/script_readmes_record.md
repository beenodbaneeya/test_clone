# Script README Record
This file preserves the original per-script README files that were removed from `scripts/`.
Archived on: 2026-07-29

## scripts/dataset_utils_README.md
```markdown
# README for `dataset_utils.py`

This document explains `scripts/dataset_utils.py` line by line and the deep learning data pipeline concepts behind it.

## Why this file matters

In deep learning, model quality depends heavily on data handling. This file is responsible for:

- locating dataset directories,
- downloading Tiny-ImageNet if needed,
- applying train/eval transforms,
- building DataLoaders for single-GPU and distributed runs,
- fixing Tiny-ImageNet validation layout to `ImageFolder` format.

---

## Imports (lines 1-11)

- `os`, `zipfile`, `Path`: filesystem and archive operations.
- `Optional`: typing for optional arguments.
- `requests`: HTTP download of Tiny-ImageNet zip.
- `tarfile`: extraction of CIFAR-100 tar archive.
- `torch`, `torchvision`, `torchvision.transforms`: tensors, datasets, and augmentations.
- `DataLoader`, `DistributedSampler`: batch loading and DDP-aware sharding.

The file also defines shared normalization constants:

- `IMAGENET_MEAN = (0.485, 0.456, 0.406)`
- `IMAGENET_STD = (0.229, 0.224, 0.225)`

These are used consistently in both Tiny-ImageNet train and validation transforms.

Deep learning concept:
- The data pipeline is not just I/O. It controls augmentation, statistical normalization, and throughput, all of which affect optimization and final generalization.

---

## `_data_dir_default()` (lines 14-18)

- Resolves repo root relative to the current file.
- Creates `datasets/` directory if missing.
- Returns a `Path` object.

Why:
- Keeps script self-contained and portable: first run can auto-create storage.

---

## `load_cifar100(...)` (lines 21-48)

### Function role

- Loads CIFAR-100 train and test sets.
- Applies transform pipeline.
- Returns `(train_loader, test_loader)`.

### Key arguments

- `batch_size`: number of samples per optimizer step.
- `num_workers`: CPU workers for background batch preparation.
- `sampler`: optional distributed sampler.
- `data_dir`: optional dataset root override.

### Important code behavior

- `root = Path(data_dir)... if data_dir else _data_dir_default()`
  - user-defined dataset path or default repo path.
- `_download_cifar100(root)` ensures CIFAR-100 is present before dataset construction.
- CIFAR-100 download uses a custom downloader and extraction flow, then `torchvision.datasets.CIFAR100(..., download=False, ...)`.
  - This avoids the `torchvision` internal HTTPS download path that can fail on some cluster certificate setups.
- Train transform uses:
  - `RandomHorizontalFlip()`
  - `RandomCrop(32, padding=4)`
  - `ToTensor()`
  - CIFAR-100 normalization stats.
- Test transform uses:
  - `ToTensor()`
  - CIFAR-100 normalization stats.
- Dataset creation uses `download=False` because download is already handled explicitly.
- Train loader:
  - `drop_last=True` (stable batch shape for training)
  - shuffle unless sampler is provided.
- Test loader:
  - `drop_last=False` (correct full-dataset evaluation)
  - `shuffle=False`.

Deep learning concept:
- Random crop/flip are training-time augmentation. They increase effective data diversity and help generalization.
- Validation/test should be deterministic so metrics are comparable across epochs and runs.
- Normalization aligns input scale with what optimizers and pretrained-style parameter regimes expect.

---

## `_download_tiny_imagenet(...)` (lines 51-75)

### Function role

- Downloads and extracts Tiny-ImageNet once.

### Behavior

- Checks if extracted directory already exists.
- Streams zip download in chunks.
- Extracts archive to `data_dir`.
- Deletes zip afterward to save disk.

Why:
- Makes cluster workflow robust and repeatable without manual dataset prep.

---

## `_download_cifar100(...)`

### Function role

- Downloads and extracts CIFAR-100 into the configured datasets directory if missing.

### Behavior

- Checks for extracted folder `cifar-100-python`.
- Attempts download URL list (HTTP first, HTTPS second).
- Streams archive to disk, extracts with `tarfile`, deletes archive.
- Raises a clear runtime error if all download attempts fail.

Why this is useful in your cluster:

- Avoids dependency on `torchvision`'s urllib HTTPS certificate path.
- Keeps download behavior aligned with your Tiny-ImageNet custom flow.
- Ensures datasets are placed under the repo root `datasets/` by default.

---

## `load_imagenet(...)` (lines 78-120)

### Function role

- Loads Tiny-ImageNet using `ImageFolder`.
- Applies separate train and validation transforms.
- Supports distributed training mode.

### Transform logic

- Train transform:
  - `RandomResizedCrop(224)`
  - `RandomHorizontalFlip()`
  - `ToTensor()`
  - `Normalize(IMAGENET_MEAN, IMAGENET_STD)`.
- Validation transform:
  - `Resize(256)` then `CenterCrop(224)`
  - `ToTensor()`
  - `Normalize(IMAGENET_MEAN, IMAGENET_STD)`.

Why this is evaluation-pure:

- Validation uses only deterministic geometric transforms (`Resize` + `CenterCrop`).
- No random transforms are applied on validation split.
- This keeps `val_loss` and `val_acc` stable and comparable across epochs and runs.

Deep learning concept:
- Train and eval transforms differ on purpose.
  - Train is stochastic for regularization.
  - Validation is deterministic to make metrics comparable across epochs.

### Tiny-ImageNet layout fix

- Calls `_fix_tiny_imagenet_val_structure(val_dir)` before loading validation data.
- Needed because Tiny-ImageNet validation images are not originally organized by class folders.

### Distributed behavior

- If `distributed=True` and no sampler provided, creates `DistributedSampler(train_set)`.
- This ensures each DDP process gets a unique shard of the training data.

### DataLoader settings

- Train loader:
  - `drop_last=True`, shuffling when sampler absent.
- Validation loader:
  - `drop_last=False`, no shuffle.

---

## `_fix_tiny_imagenet_val_structure(...)` (lines 123-142)

### Function role

- Converts Tiny-ImageNet val directory into `ImageFolder`-compatible class subfolders.

### Behavior

- Checks for `val/images` folder.
- Reads `val_annotations.txt`.
- For each image, creates class directory and moves file there.
- Deletes old `images` directory and annotation file.

Why:
- `torchvision.datasets.ImageFolder` expects `root/class_name/image.jpg` structure.
- Without this conversion, labels cannot be inferred automatically.

---

## Deep learning and systems notes for users

- `num_workers` and `pin_memory=True` affect throughput:
  - more workers often improve data pipeline speed until CPU/I/O saturates.
  - pinning memory improves host-to-GPU transfer performance.
- `drop_last=False` for validation is important for metric correctness.
- For fair model comparisons, keep transforms and split definitions unchanged.

---

## Recent improvement applied

The CIFAR-100 pipeline now uses separate transforms for train and test:

- Train split: stochastic augmentation (`RandomCrop`, `RandomHorizontalFlip`) + normalization.
- Test split: deterministic transform (`ToTensor` + normalization only).

Why this is better:

- Training still benefits from augmentation-based regularization.
- Evaluation becomes stable and fair across epochs.
- Accuracy comparisons across runs become more trustworthy.

Tiny-ImageNet already followed this train/eval separation and has now been cleaned up further by reusing shared normalization constants for consistency and readability.
```

## scripts/device_utils_README.md
```markdown
# README for `device_utils.py`

This document explains `scripts/device_utils.py` in detail so you can confidently explain why device selection matters in deep learning training.

## File Purpose

`device_utils.py` provides one small helper function, `get_device()`, that decides where tensors and models should run:

- GPU (`cuda:0`) if CUDA is available.
- CPU if CUDA is not available.

In deep learning, this choice is critical because training on GPU is usually much faster than CPU for matrix-heavy operations (convolutions, attention, and large linear layers).

---

## Full Source (for reference)

```python
# device_utils.py
import torch

def get_device():
    """
    Determine the compute device (GPU or CPU).
    Returns:
        torch.device: The device to use for the computations.
    """

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

---

## Line-by-Line Explanation

### Line 1

`# device_utils.py`

- Simple file label comment.
- No runtime effect.

### Line 2

`import torch`

- Imports the PyTorch library.
- We need this import because:
  - `torch.cuda.is_available()` checks if a CUDA-capable GPU is visible to PyTorch.
  - `torch.device(...)` creates a device object used by `.to(device)` calls.

### Line 4

`def get_device():`

- Defines a helper function that centralizes device logic.
- Why centralize? If you repeat this logic in multiple scripts, inconsistencies can appear. A single function keeps behavior consistent across training/evaluation scripts.

### Lines 5-9 (Docstring)

```python
"""
Determine the compute device (GPU or CPU).
Returns:
    torch.device: The device to use for the computations.
"""
```

- Documents what the function does and what it returns.
- Useful for maintainability and for users unfamiliar with PyTorch.

### Line 11 (Core Logic)

`return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`

- This is a conditional expression:
  - If `torch.cuda.is_available()` is `True`, return `torch.device("cuda:0")`.
  - Otherwise, return `torch.device("cpu")`.
- `cuda:0` means first visible GPU in the process.
- This returned object is then used later in training code, typically like:
  - `model.to(device)`
  - `images.to(device)` and `labels.to(device)`

---

## Deep Learning Concept: Why Device Selection Matters

### 1) Tensor Compute Throughput

Deep learning workloads are dominated by dense tensor math:

- Matrix multiplications
- Convolutions
- Attention operations

GPUs are built with many parallel cores and high memory bandwidth, so they process these operations significantly faster than CPUs.

### 2) Model and Data Must Be on the Same Device

In PyTorch, if the model is on GPU but data is on CPU (or vice versa), computation fails with a device mismatch error.

That is why `get_device()` is paired with explicit `.to(device)` calls in training loops.

### 3) Reproducible Behavior Across Environments

By checking availability at runtime, the same script can run on:

- A cluster node with GPUs.
- A local machine without GPUs.

This helps portability and reduces user confusion.

---

## How This Fits Your Training Pipeline

In your project, `get_device()` supports clean single-GPU training by:

- Automatically selecting GPU when available.
- Falling back to CPU for compatibility.
- Keeping training scripts simpler and easier for new users to read.

This design is beginner-friendly and good for hosted educational workflows.

---

## Practical Notes for Cluster Users

- `cuda:0` points to the first GPU visible to the job.
- In scheduler-managed jobs (like your NRIS module), GPU visibility can be restricted per job, and `cuda:0` still works correctly within that visibility scope.
- If no GPU is requested or visible, training falls back to CPU, which is slower but still functional.

---

## Suggested Future Improvement (Optional)

When you are ready, we can slightly improve this script by adding type hints and optional diagnostics:

- Return type hint: `def get_device() -> torch.device:`
- Optional print/helper for reporting device name (`torch.cuda.get_device_name(0)`) when using GPU.

These are not required for correctness; your current logic is valid and minimal.
```

## scripts/model_README.md
```markdown
# README for `model.py`

This document explains every line of `scripts/model.py` and the deep learning concepts behind it. The goal is that you can read the model code and clearly explain both implementation and theory to cluster users.

## What this file does

`model.py` defines two neural network architectures:

- `WideResNet`: a convolutional residual network (CNN family).
- `ViTModel`: a Vision Transformer model using `torchvision` pretrained weights.

These two architectures are intentionally different, so users can compare training behavior, throughput, GPU memory usage, and convergence patterns.

---

## Full Source (current)

```python
import torch
import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.01) -> None:
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        return self.activation(out + residual)


class WideResNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        channels = [3, 16, 160, 320, 640]
        self.stem = ConvBnReLU(channels[0], channels[1])
        self.block1 = ResidualBlock(channels[1], channels[2])
        self.block2 = ResidualBlock(channels[2], channels[2])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.block3 = ResidualBlock(channels[2], channels[3])
        self.block4 = ResidualBlock(channels[3], channels[3])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.block5 = ResidualBlock(channels[3], channels[4])
        self.block6 = ResidualBlock(channels[4], channels[4])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[4], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.classifier(out)


class ViTModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        self.vit = vit_b_16(weights=weights)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
```

---

## Line-by-Line Explanation

### Imports

- **Line 1** `import torch`
  - Imports top-level PyTorch API.
  - Used for `torch.Tensor` type hints and `torch.flatten` in `WideResNet.forward`.

- **Line 2** `import torch.nn as nn`
  - Imports neural network module namespace.
  - Gives access to layers such as `Conv2d`, `BatchNorm2d`, `ReLU`, `Linear`, pooling, etc.

- **Line 3** `from torchvision.models import ViT_B_16_Weights, vit_b_16`
  - Imports the Vision Transformer constructor and pretrained weight enum from `torchvision`.
  - `ViT_B_16_Weights.IMAGENET1K_V1` specifies which pretrained checkpoint to load.

### `ConvBnReLU` block

- **Line 6** `class ConvBnReLU(nn.Module):`
  - Defines a reusable convolutional feature block.

- **Line 7** constructor with type hints.
  - `in_channels`: number of channels entering the layer.
  - `out_channels`: number of channels produced.

- **Line 8** `super().__init__()`
  - Initializes base `nn.Module` internals.

- **Line 9** `self.block = nn.Sequential(...)`
  - Groups multiple layers into one callable block.

- **Line 10** `nn.Conv2d(..., kernel_size=3, stride=1, padding=1, bias=False)`
  - 3x3 convolution keeps spatial size due to `padding=1`.
  - `bias=False` because BatchNorm follows and absorbs offset behavior.

- **Line 11** `nn.BatchNorm2d(out_channels)`
  - Normalizes activation distribution per mini-batch.
  - Helps stable gradients and faster convergence.

- **Line 12** `nn.ReLU(inplace=True)`
  - Adds non-linearity.
  - `inplace=True` slightly reduces memory overhead.

- **Line 15** forward method signature with tensor type hints.

- **Line 16** `return self.block(x)`
  - Applies conv -> batchnorm -> relu.

### `ResidualBlock`

- **Line 19** `class ResidualBlock(nn.Module):`
  - Defines a ResNet-style residual unit.

- **Line 20** constructor includes optional `dropout_p`.
  - Small dropout regularizes training.

- **Line 21** base module init.

- **Line 22** first sub-layer uses `ConvBnReLU`.

- **Line 23** dropout layer.

- **Lines 24-27** second conv stack:
  - Conv 3x3 + BatchNorm, without ReLU here.
  - Activation is applied after residual addition (line 43), a common residual pattern.

- **Lines 28-35** shortcut path definition:
  - If channels match, use identity (`nn.Identity`).
  - If channels differ, project input with 1x1 conv + batchnorm.
  - This ensures shapes match before addition.

- **Line 36** final activation module.

- **Line 38** forward signature.

- **Line 39** `residual = self.shortcut(x)`
  - Computes skip path.

- **Lines 40-42** main path:
  - conv-bn-relu -> dropout -> conv-bn.

- **Line 43** `return self.activation(out + residual)`
  - Adds skip signal and transformed signal.
  - Activation after addition gives non-linear fused output.

### `WideResNet`

- **Line 46** model class definition.

- **Line 47** constructor accepts `num_classes`.

- **Line 48** module init.

- **Line 49** `channels = [3, 16, 160, 320, 640]`
  - Width configuration.
  - `3` input channels for RGB images.
  - Large channel counts create a wide network (higher representational capacity).

- **Line 50** stem block converts RGB input to first feature space.

- **Lines 51-52** first residual stage.

- **Line 53** first downsampling via max-pooling (spatial reduction by 2x).

- **Lines 54-55** second stage with higher channel count.

- **Line 56** second downsampling.

- **Lines 57-58** third stage with highest channels.

- **Line 59** adaptive global pooling to `(1,1)`.
  - Makes model robust to input image size differences.

- **Line 60** linear classifier from final channel dimension to class logits.

- **Line 62** forward signature.

- **Lines 63-71** feature extraction pipeline through stem, residual blocks, and pooling.

- **Line 72** global pooling reduces spatial dimensions.

- **Line 73** `torch.flatten(out, 1)` flattens from channel dimension onward.

- **Line 74** final class logits.

### `ViTModel`

- **Line 77** model class wrapper for ViT.

- **Line 78** constructor args:
  - `num_classes`: output class count.
  - `pretrained`: whether to load ImageNet pretrained weights.

- **Line 79** base module init.

- **Line 80** chooses pretrained weights or random init.

- **Line 81** `self.vit = vit_b_16(weights=weights)`
  - Loads Vision Transformer B/16 architecture.

- **Line 82** replaces classification head to match dataset classes.

- **Line 84** forward signature.

- **Line 85** delegates forward pass to torchvision ViT.

---

## Deep Learning Concepts You Should Know

### 1) What is a model in deep learning?

A model is a parameterized function that maps input tensors (images) to outputs (class logits). During training, optimizer updates model parameters to minimize loss.

### 2) CNN vs Transformer vision models

- **CNN (WideResNet)**
  - Strong local inductive bias (convolution sees local neighborhoods).
  - Efficient and robust on many vision tasks.
  - Often easier to optimize from scratch.

- **ViT (Vision Transformer)**
  - Uses self-attention over image patches.
  - Captures long-range global interactions early.
  - Often benefits heavily from pretraining.

### 3) Residual learning

Residual blocks learn a correction to the input (`F(x)`) and output `F(x) + x`. This improves gradient flow and makes deeper networks easier to train.

### 4) Batch normalization

BatchNorm stabilizes intermediate activations and allows higher learning rates, often improving training speed and reliability.

### 5) Why replace ViT classification head?

Pretrained ViT is trained on ImageNet-1k (1000 classes). Your dataset has different class count (e.g., Tiny-ImageNet = 200). So head replacement is required for shape correctness and task alignment.

### 6) Logits vs probabilities

Final layer outputs logits (unnormalized scores). `CrossEntropyLoss` internally applies log-softmax behavior, so explicit softmax is not needed in model output.

### 7) Transfer learning (important for your experiment)

Using pretrained ViT gives the model strong visual features from prior training. Fine-tuning on Tiny-ImageNet often converges faster than training from random initialization.

---

## Why this implementation is better than before

- Clearer naming (`ConvBnReLU`, `ResidualBlock`) improves readability.
- Type hints improve maintainability and onboarding.
- New ViT API (`weights=...`) avoids deprecated `pretrained=True` style.
- Residual projection includes BatchNorm and shape-safe shortcut behavior.
- Adaptive average pooling removes dependence on a fixed spatial size.

---

## What your latest logs indicate (after this model update)

From `jobs/nris_module/logs/singlegpu_1630603.out` and `jobs/nris_module/logs/singlegpu.log`:

- Training starts correctly with ViT + Tiny-ImageNet and AdamW at lr=0.0003.
- Epoch metrics improve quickly:
  - Epoch 1: `val_loss=3.1769`, `val_acc=0.2786`
  - Epoch 2: `val_loss=1.7078`, `val_acc=0.5743`
  - Epoch 3: `val_loss=1.3265`, `val_acc=0.6573`
- Throughput is stable around `858-875 img/s`, indicating consistent training step timing.
- GPU utilization is near saturation (`99-100%` most samples), which means compute kernels keep the accelerator busy.
- GPU memory used is about `34389 MiB` (~34.4 GiB), stable during training.
- Job ended by scheduler signal (`Terminated`), not by model/runtime crash; this is operational (job cancellation) rather than model failure.

Interpretation: model code and training path are healthy, convergence trend is strong in first epochs, and hardware utilization is good for a single-GPU benchmark run.
```

## scripts/train_README.md
```markdown
# README for `train.py`

This README explains the current `scripts/train.py` in detail, including the new optional flags (`--seed`, `--amp`, `--deterministic`) and how to interpret your three cluster runs.

## What this script does

`train.py` orchestrates one complete single-GPU deep learning experiment:

- Parse CLI arguments.
- Set reproducibility controls.
- Select device (GPU or CPU).
- Build dataset loaders and model.
- Build optimizer using architecture-aware defaults.
- Train and validate each epoch.
- Report quality metrics (loss/accuracy) and systems metrics (throughput).

In deep learning terms, this script is the **experiment controller**; the model file defines architecture, and `train_utils.py` performs batch-wise learning math.

---

## Current full code (reference)

```python
"""Single-GPU training script for CIFAR-100 and Tiny-ImageNet benchmarks."""

import argparse
import random
import time
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset_utils import load_cifar100, load_imagenet
from device_utils import get_device
from model import ViTModel, WideResNet
from train_utils import test as evaluate
from train_utils import train as train_one_epoch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model on a single GPU.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["wideresnet", "vit"],
        default="vit",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar100", "tiny-imagenet"],
        default="tiny-imagenet",
        help="Dataset to use.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--base-lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["auto", "sgd", "adam", "adamw"],
        default="auto",
        help="Optimizer to use.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.95,
        help="Validation accuracy threshold for early stopping.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="How many consecutive epochs must meet target accuracy.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional dataset directory (used for Tiny-ImageNet download/storage).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision (AMP) on CUDA.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic operations when possible.",
    )
    return parser.parse_args()


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, int]:
    if args.dataset == "cifar100":
        train_loader, test_loader = load_cifar100(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
        )
        return train_loader, test_loader, 100

    train_loader, test_loader = load_imagenet(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        distributed=False,
    )
    return train_loader, test_loader, 200


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    if model_name == "wideresnet":
        return WideResNet(num_classes=num_classes).to(device)
    return ViTModel(num_classes=num_classes).to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> Tuple[Optimizer, float]:
    if args.optimizer == "auto":
        if args.model == "wideresnet":
            learning_rate = args.base_lr if args.base_lr is not None else 0.1
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            learning_rate = args.base_lr if args.base_lr is not None else 3e-4
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        return optimizer, learning_rate

    learning_rate = args.base_lr if args.base_lr is not None else 1e-3
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer, learning_rate


def loader_num_samples(loader: DataLoader) -> int:
    sampler: Any = getattr(loader, "sampler", None)
    if sampler is not None:
        try:
            return len(sampler)
        except TypeError:
            pass
    return len(loader.dataset)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def main() -> None:
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    device = get_device()
    train_loader, test_loader, num_classes = build_dataloaders(args)
    model = build_model(args.model, num_classes, device)
    optimizer, learning_rate = build_optimizer(args, model)
    loss_fn = nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(
        f"Training {args.model} on {args.dataset} | "
        f"batch_size={args.batch_size}, lr={learning_rate}, optimizer={optimizer.__class__.__name__}, "
        f"amp={use_amp}, seed={args.seed}, deterministic={args.deterministic}"
    )

    val_accuracies: list[float] = []
    total_time = 0.0
    total_images = 0

    for epoch in range(args.epochs):
        start_time = time.time()
        train_acc, train_loss, epoch_images = train_one_epoch(
            model,
            optimizer,
            train_loader,
            loss_fn,
            device,
            use_amp=use_amp,
            scaler=scaler,
        )
        epoch_time = time.time() - start_time
        total_time += epoch_time

        if epoch_images <= 0:
            epoch_images = loader_num_samples(train_loader)
        total_images += epoch_images
        throughput = epoch_images / epoch_time

        val_accuracy, val_loss = evaluate(model, test_loader, loss_fn, device, use_amp=use_amp)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"time={epoch_time:.3f}s, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_accuracy:.4f}, throughput={throughput:.1f} img/s"
        )

        if len(val_accuracies) >= args.patience and all(
            acc >= args.target_accuracy for acc in val_accuracies[-args.patience:]
        ):
            print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
            break

    final_throughput = total_images / total_time if total_time > 0 else 0.0
    print(f"\nTraining complete. Final val_acc: {val_accuracies[-1]:.4f}")
    print(f"Total time: {total_time:.1f}s, final throughput: {final_throughput:.1f} img/s")


if __name__ == "__main__":
    main()
```

---

## Line-by-line explanation (compact, complete)

### Imports and module scope

- `argparse`, `time`: CLI and timing.
- `random`: seeds Python RNG.
- `Any`, `Tuple`: type hints.
- `torch`, `nn`, `optim`, `Optimizer`, `DataLoader`: training primitives and explicit types.
- Project imports:
  - `load_cifar100`, `load_imagenet` for dataset pipelines.
  - `get_device` for CPU/GPU selection.
  - `ViTModel`, `WideResNet` for architecture selection.
  - `train_one_epoch`, `evaluate` for loop internals.

### `parse_args()`

- Builds a reproducible CLI contract.
- Important arguments and deep learning meaning:
  - `--model`, `--dataset`: experiment identity.
  - `--batch-size`: memory/performance tradeoff.
  - `--epochs`: maximum optimization budget.
  - `--base-lr`: step size override.
  - `--optimizer`: optimization dynamics.
  - `--target-accuracy`, `--patience`: early stopping policy.
  - `--num-workers`: input pipeline parallelism.
  - `--data-dir`: dataset storage location.
  - `--seed`: random initialization/data order control.
  - `--amp`: mixed precision acceleration.
  - `--deterministic`: strict reproducibility mode.

### `build_dataloaders(args)`

- Routes to CIFAR-100 or Tiny-ImageNet loaders.
- Returns `num_classes` (`100` or `200`) so classifier heads are dimension-correct.

### `build_model(model_name, num_classes, device)`

- Instantiates chosen model and moves it to selected device with `.to(device)`.
- In PyTorch, model and tensors must live on same device.

### `build_optimizer(args, model)`

- `auto` mode chooses architecture-aware defaults:
  - WideResNet -> SGD + momentum + weight decay.
  - ViT -> AdamW + weight decay.
- Returns `(optimizer, resolved_lr)` so logging reflects actual settings.

### `loader_num_samples(loader)`

- Safe helper to estimate processed sample count.
- Uses sampler length when available; otherwise dataset length.
- Supports robust throughput reporting.

### `set_seed(seed, deterministic)`

- Seeds Python + Torch RNGs; seeds all CUDA devices if present.
- Deterministic mode:
  - `cudnn.deterministic=True` and `benchmark=False`.
  - Reproducibility improves, but speed often drops.
- Non-deterministic mode:
  - `benchmark=True` lets cuDNN choose fast kernels for the current shapes.

### `main()`

- Calls all builder/helper functions.
- Sets:
  - `loss_fn = CrossEntropyLoss` for multi-class logits.
  - `use_amp` when both `--amp` and CUDA are active.
  - `GradScaler` only for AMP mode.
- Prints config for experiment traceability.
- Epoch loop:
  - Trains one epoch and gets train metrics.
  - Computes epoch time and throughput.
  - Runs validation and logs all metrics.
  - Applies early stopping condition.
- Prints final throughput summary.

---

## Deep learning concepts behind new flags

## 1) `--seed` (reproducibility)

- Controls random initial weights, data order randomness, and stochastic behavior.
- Same seed + same environment tends to produce similar trends.
- Important for fair model comparison and debugging.

## 2) `--amp` (mixed precision)

- Uses lower precision math (mainly float16/bfloat16) where safe.
- Usually increases throughput and may lower memory usage.
- `GradScaler` protects gradient updates from underflow.

## 3) `--deterministic` (strict reproducibility)

- Forces deterministic algorithm choices where possible.
- Great for repeatability, often slower because fastest non-deterministic kernels are disabled.

---

## Interpreting your three new logs

You ran three controlled cases on ViT + Tiny-ImageNet + batch size 256 + AdamW lr 3e-4.

### Case A: `--seed 42` only (`amp=False`, `deterministic=False`)

Observed:

- Epoch 1: `117.892s`, `train_acc=0.2992`, `val_acc=0.6357`, `846.9 img/s`
- Epoch 2: `114.490s`, `train_acc=0.5695`, `val_acc=0.7065`, `872.0 img/s`

What it means:

- This is your baseline speed-quality balance.
- Accuracy rises quickly, loss drops strongly: optimization is healthy.
- Throughput around 850-870 img/s is your non-AMP reference.

### Case B: `--seed 42 --amp` (`deterministic=False`)

Observed:

- Epoch 1: `58.160s`, `train_acc=0.3008`, `val_acc=0.6307`, `1716.6 img/s`
- Epoch 2: `56.941s`, `train_acc=0.5711`, `val_acc=0.7087`, `1753.4 img/s`

What it means:

- Throughput is about 2x higher than baseline.
- Accuracy/loss are very close to baseline, so quality is preserved.
- This is a textbook AMP success: major speedup with negligible metric drift.

### Case C: `--seed 42 --deterministic` (`amp=False`)

Observed:

- Epoch 1: `281.943s`, `train_acc=0.2947`, `val_acc=0.6240`, `354.1 img/s`

What it means:

- Throughput drops heavily (to ~40% of baseline, ~20% of AMP run).
- Accuracy remains in a similar ballpark for early epoch.
- This is expected: strict determinism trades raw speed for repeatability.

### Practical takeaway

- For production benchmarking: use `--amp` (best speed/quality tradeoff).
- For strict reproducibility experiments: use `--deterministic` (accept slower runtime).
- For day-to-day training: keep deterministic off, seed fixed, and AMP on.

---

## Suggested run profiles for users

```bash
# Fast default for cluster users
python scripts/train.py --model vit --dataset tiny-imagenet --seed 42 --amp

# Reproducibility-focused debug run
python scripts/train.py --model vit --dataset tiny-imagenet --seed 42 --deterministic

# Baseline comparison run
python scripts/train.py --model vit --dataset tiny-imagenet --seed 42
```

---

## Note for mentoring users new to deep learning

When users ask why one run is faster/slower, use this short explanation:

- The model quality comes from optimization and data.
- The runtime speed comes from numerical precision and kernel selection.
- AMP changes precision to accelerate hardware.
- Deterministic mode restricts algorithm choice for reproducibility.

So these flags mainly change **how** math is executed, not the task itself.
```

## scripts/train_ddp_README.md
```markdown
# README for `train_ddp.py`

This document explains `scripts/train_ddp.py` and interprets your latest NRIS multi-GPU logs for WideResNet on CIFAR-100.

## What this script does

`train_ddp.py` is the distributed (DDP) training controller. It launches one process per GPU and keeps all processes synchronized while they train the same model on different shards of data.

Core responsibilities:

- Parse distributed experiment arguments.
- Initialize PyTorch distributed (`nccl`) and rank/device mapping.
- Build dataset loaders with `DistributedSampler`.
- Build model and wrap with `DistributedDataParallel`.
- Train and evaluate each epoch.
- Correctly aggregate metrics across ranks.
- Compute global throughput.
- Apply synchronized early stopping and cleanup.

---

## High-level function-by-function explanation

## `parse_args()`

- Defines CLI knobs for model/dataset/optimizer/learning-rate, plus distributed-performance knobs (`--batch-size` global, `--num-workers`, `--amp`, `--seed`, `--deterministic`).
- `--batch-size` is global across all GPUs; per-GPU batch is computed later as `global_batch/world_size`.

## `ddp_setup()`

- Calls `dist.init_process_group(backend="nccl")` for GPU collectives.
- Reads `LOCAL_RANK`, `RANK`, `WORLD_SIZE` from `torchrun`.
- Pins each process to one GPU: `torch.cuda.set_device(local_rank)`.

Deep learning concept:
- DDP uses data parallelism: each GPU processes different mini-batches, gradients are synchronized so all replicas stay equivalent after each step.

## `set_seed(seed, deterministic, rank)`

- Uses `seed + rank` to avoid identical per-rank random streams.
- Controls cuDNN deterministic/benchmark behavior.

Deep learning concept:
- Determinism improves reproducibility, often reducing speed; benchmark mode improves speed for fixed tensor shapes.

## `build_dataloaders(args, global_rank, world_size)`

- Ensures global batch is divisible by number of GPUs.
- Rank 0 prepares dataset first, then `dist.barrier()` avoids race conditions.
- Builds train/validation datasets.
- Attaches `DistributedSampler` for both splits.
- Creates `DataLoader`s with per-GPU batch.

Deep learning concept:
- `DistributedSampler` prevents overlap: each rank sees unique samples.
- For validation, sampler-based partitioning allows global metric aggregation from all ranks.

## `build_model(...)` and `build_optimizer(...)`

- Creates `WideResNet` or `ViTModel`.
- `auto` optimizer defaults:
  - WideResNet -> SGD + momentum + weight decay.
  - ViT -> AdamW.

Deep learning concept:
- Optimizer choice is architecture-sensitive; CNNs and Transformers often prefer different defaults.

## `train_one_epoch(...)`

- Standard training loop with optional AMP.
- Returns local rank stats: `correct`, `loss_sum`, `num_samples`.

Deep learning concept:
- Loss is accumulated as sample-weighted sum, which is important for correct global averaging when batch sizes differ.

## `evaluate(...)`

- Evaluation mode + `torch.no_grad()`.
- Optional AMP for faster eval.
- Returns local validation `correct`, `loss_sum`, `num_samples`.

## `all_reduce_metrics(...)`

- Sums local metrics across all ranks using `dist.all_reduce(SUM)`.
- Computes globally correct:
  - `accuracy = global_correct / global_total`
  - `loss = global_loss_sum / global_total`

This is the mathematically correct DDP metric strategy.

## `main_worker()`

- Orchestrates full run.
- Wraps model with `DDP(model, device_ids=[local_rank])`.
- Logs from rank 0 only.
- Uses max epoch time across ranks for fair global throughput.
- Synchronizes early-stopping decision via broadcasted tensor.
- Ensures process-group cleanup in `finally`.

---

## Interpretation of your latest logs

## Single GPU (`jobs/nris_module/logs/singlegpu_1630603.out`)

- CIFAR-100 download now succeeds from your mirror URL.
- Training starts with `wideresnet`, `cifar100`, SGD `lr=0.1`.
- First 3 epochs show healthy learning:
  - `train_acc`: `0.1071 -> 0.2341 -> 0.3383`
  - `val_acc`: `0.1541 -> 0.2118 -> 0.3328`
  - `val_loss`: `3.5268 -> 3.2748 -> 2.6338`
- Throughput rises after epoch 1 warmup: `4206 -> ~7220 img/s`.
- `singlegpu.log` confirms delayed GPU ramp-up (download + setup first), then near-full utilization (~98-99%) during training.

Conclusion:
- Single-GPU pipeline is healthy; optimizer/model/data path behaves correctly.

## Multi GPU DDP (`jobs/nris_module/logs/multigpu_1630612.out`)

- All ranks detect existing CIFAR and skip download.
- DDP launch configuration is coherent:
  - world size 4
  - global batch 512
  - per-GPU batch 128
  - SGD with `lr=0.04`
- Learning curve is strong over 8 epochs:
  - `train_acc`: `0.1434 -> 0.6338`
  - `val_acc`: `0.1694 -> 0.4929`
  - `val_loss`: `3.8555 -> 1.9964`
- Throughput stabilizes around `~23.3k-23.6k img/s` after first epoch.

DDP scaling insight:
- Single GPU steady throughput: ~7.2k img/s.
- DDP steady throughput: ~23.5k img/s.
- Approx speedup: ~3.25x on 4 GPUs.
- Efficiency ~81% (`3.25/4`), which is good for real distributed training.

## Multi-GPU utilization (`jobs/nris_module/logs/multigpu.log`)

- Initial low utilization during startup/data setup.
- Then all GPUs move to ~92-94% compute with ~4.4 GB memory each.
- Balanced memory/compute across 4 GPUs indicates proper rank-to-device mapping.

## Warning in `multigpu_1630612.err`

- Warning about `barrier()` device context and guessed device ID.
- This is a warning, not a failure.
- Your run still trained successfully and produced consistent metrics.

## GPU Utilization Log Interpretation (Important for Users)

In `multinode.log` and related GPU logs, utilization may occasionally drop to `0%` and then ramp up again. This is expected and does not automatically indicate a training error.

Why this happens:

- Sampling frequency vs epoch speed:
  - `nvidia-smi` is sampled every 5 seconds.
  - WideResNet on CIFAR-100 can complete an epoch in around 1 to 2 seconds in DDP.
  - A 5-second snapshot often captures idle/sync windows (for example, end-of-epoch synchronization or transition between train and validation), so utilization appears bursty.
- Local-node monitoring scope:
  - Current `multinode.sh` monitoring logs GPUs on the launching node.
  - In multi-node training, this does not represent all GPUs across all nodes at every instant.

Practical takeaway:

- For short-epoch distributed runs, occasional `0%` points in sampled logs are normal.
- Use epoch-level throughput and loss/accuracy trends as the primary indicators of training health.

### Optional command to monitor all nodes

If users need per-node GPU utilization instead of a single local log, they can run monitoring through `srun` and write one file per hostname:

```bash
srun --ntasks=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -lc 'nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv -l 5 > "${SLURM_SUBMIT_DIR}/logs/multinode_${SLURMD_NODENAME}.log"'
```

This avoids one very large mixed log and keeps analysis clear per node.

---

## Deep learning concepts demonstrated by these experiments

- **Data parallelism:** each GPU processes different mini-batches, then gradients are reduced.
- **Global metrics:** local stats must be summed then normalized globally for correct val/train reporting.
- **Warmup effect:** first epoch is slower due to startup overhead (data worker startup, kernel caching, graph setup).
- **Throughput vs convergence:** higher throughput in DDP does not guarantee better accuracy per epoch, but usually gives faster wall-clock convergence.
- **Generalization tracking:** train accuracy rises faster than validation accuracy, which is expected; gap monitoring helps detect overfitting later.
- **Mixed precision (AMP):** AMP uses lower-precision arithmetic where it is numerically safe and keeps sensitive updates stable with gradient scaling. This reduces memory pressure and typically increases tensor-core throughput, so users can reach more training steps per second without changing the model objective.

---

## Batch size and scaling interpretation

When comparing single-GPU, multi-GPU, and multi-node runs, there are two valid scaling styles:

- **Strong scaling:** keep global batch fixed, increase GPU count, and measure speedup on the same total work.
- **Weak scaling:** keep per-GPU batch fixed, increase GPU count, and increase global batch proportionally.

For your current NRIS job scripts:

- Single GPU: `batch_size=256` (per GPU = 256)
- 4 GPUs: `global_batch_size=1024` (per GPU = 256)
- 8 GPUs: `global_batch_size=2048` (per GPU = 256)

This is weak scaling, which is appropriate for demonstrating how training scales across more GPUs and nodes while keeping each GPU equally loaded.

If users want strict speedup comparison on identical total work, they should run strong scaling by keeping global batch the same across all runs.

---

## Practical next tests you can run

1. Add `--amp` in DDP to test additional speedup and memory reduction.
2. Compare `global_batch_size=512` vs `1024` to study scaling/generalization tradeoff.
3. Run longer (no manual cancellation) to compare final accuracy single vs DDP at similar optimizer settings.
```

## scripts/train_utils_README.md
```markdown
# README for `train_utils.py`

This document explains `scripts/train_utils.py` line by line and the core optimization concepts used during training and validation.

## Why this file matters

`train_utils.py` contains the inner-loop logic that actually performs learning:

- `train(...)`: forward pass, loss computation, backward pass, optimizer step, and train metrics.
- `test(...)`: evaluation-only forward pass and validation metrics.

In deep learning, these two functions are where model parameters are updated (or intentionally not updated during validation).

---

## Full source overview

The file has two functions:

- `train(...) -> tuple[train_accuracy, train_loss, num_images]`
- `test(...) -> tuple[val_accuracy, val_loss]`

Both support optional AMP (`use_amp`) on CUDA devices.

---

## Imports (lines 1-3)

- `import torch`
  - Tensor operations, autocast, gradient scaler types.
- `from torch.optim import Optimizer`
  - Explicit optimizer type hint.
- `from torch.utils.data import DataLoader`
  - Explicit DataLoader type hint.

Why:
- Type hints make function contracts clearer for users and maintainers.

---

## `train(...)` line-by-line

### Signature (lines 6-14)

- `model`: any `torch.nn.Module` classifier.
- `optimizer`: SGD/Adam/AdamW, etc.
- `train_loader`: batches of `(images, labels)`.
- `loss_fn`: usually `CrossEntropyLoss` for multi-class tasks.
- `device`: CPU or CUDA target.
- `use_amp`: mixed precision toggle.
- `scaler`: `GradScaler` for safe AMP gradient updates.
- Returns `(train_accuracy, train_loss, total_labels)`.

Deep learning concept:
- Function signature defines the training step API. Keeping this explicit helps you swap models/optimizers without changing loop internals.

### Setup (lines 21-24)

- `model.train()` enables training mode.
  - Activates layers like dropout.
  - BatchNorm uses batch statistics.
- Initializes counters for sample-weighted metrics.

Deep learning concept:
- `train()` mode changes layer behavior. Forgetting this causes silent metric issues.

### Batch loop (lines 26-47)

- Moves tensors to device with `non_blocking=True`.
- `optimizer.zero_grad(set_to_none=True)` clears old gradients efficiently.

Why clear gradients?
- In PyTorch, gradients accumulate by default. If not reset each step, updates become incorrect.

#### AMP branch (lines 30-36)

- `autocast` runs selected ops at lower precision.
- `scaler.scale(loss).backward()` prevents fp16 underflow.
- `scaler.step(optimizer)` + `scaler.update()` performs safe optimizer updates.

Deep learning concept:
- Mixed precision speeds tensor math and often improves throughput.
- Gradient scaling preserves numerical stability.

#### Standard precision branch (lines 37-41)

- Full precision forward, loss, backward, optimizer step.

### Metrics update (lines 43-47)

- `predictions = outputs.argmax(dim=1)` picks highest-logit class.
- Accuracy counts correct class predictions.
- `loss_total += loss.item() * batch_size` computes sample-weighted mean loss.

Why weighted loss?
- If final batch size differs, plain average of batch means can bias epoch loss.

### Return (lines 49-51)

- `train_accuracy = correct / total`
- `train_loss = weighted_loss / total`
- returns both + total sample count for throughput logic.

---

## `test(...)` line-by-line

### Signature (lines 54-60)

- Similar to `train`, but no optimizer/scaler needed for parameter updates.
- Returns `(val_accuracy, val_loss)`.

### Setup (lines 67-70)

- `model.eval()` switches to evaluation mode.
  - Dropout is disabled.
  - BatchNorm uses running averages.

Deep learning concept:
- Eval mode gives deterministic inference behavior and fairer validation tracking.

### No-grad evaluation loop (lines 72-87)

- `with torch.no_grad()` disables gradient graph creation.
  - Saves memory.
  - Speeds validation.
- Moves data to device.
- Optional AMP autocast branch for faster eval on CUDA.
- Computes predictions and sample-weighted loss.

### Return (lines 89-90)

- Standard epoch-level validation accuracy and mean loss.

---

## Key deep learning concepts in this file

## 1) Forward pass

- Model maps inputs to logits.

## 2) Loss function

- Measures mismatch between predicted logits and true labels.
- For classification, lower cross-entropy usually means better calibrated predictions.

## 3) Backpropagation

- `loss.backward()` computes gradients of loss w.r.t model parameters.

## 4) Optimizer step

- `optimizer.step()` updates parameters using gradients.
- This is the core learning action.

## 5) Train mode vs eval mode

- `train()` and `eval()` are not cosmetic; they change layer behavior.

## 6) AMP and numerical precision

- AMP trades some precision for speed.
- `GradScaler` protects tiny gradients.
- In your logs, AMP nearly doubled throughput while keeping accuracy similar.

## 7) Metric correctness

- Sample-weighted loss and full validation set (`drop_last=False` in dataset utils) make reported metrics more trustworthy.

---

## How to explain this simply to users

- `train(...)` is where the model learns.
- `test(...)` is where we check how well it generalizes.
- AMP changes arithmetic precision for speed.
- Accuracy tells correctness, loss tells optimization quality.

Both quality and speed are needed for a good cluster benchmark.
```

