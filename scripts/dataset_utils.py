"""Dataset loading utilities for CIFAR-100 and Tiny-ImageNet in PyTorch."""

import os
import shutil
import ssl
import zipfile
from pathlib import Path
from typing import Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Dataset Normalization Statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def _data_dir_default() -> Path:
    """Return default dataset directory at <repo>/datasets."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


# CIFAR-100 Data Loader

def load_cifar100(
    batch_size: int,
    num_workers: int = 4,
    sampler: Optional[DistributedSampler] = None,
    data_dir: Optional[str] = None,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-100 train and test data loaders from local storage."""
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()
    cifar_local_path = root / "cifar-100-python"

    if not cifar_local_path.exists():
        raise FileNotFoundError(
            f"CIFAR-100 dataset not found at '{cifar_local_path}'. "
            "Please run 'bash datasets/download_datasets.sh' first."
        )

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_set = torchvision.datasets.CIFAR100(
        root=str(root), download=False, train=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR100(
        root=str(root), download=False, train=False, transform=transform_test
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader

# Tiny-ImageNet Extractor & Data Loader


def _extract_and_organize_tiny_imagenet(root: Path) -> Path:
    tiny_dir = root / "tiny-imagenet-200"
    zip_path = root / "tiny-imagenet-200.zip"
    complete_marker = tiny_dir / ".complete"

    if complete_marker.exists():
        return tiny_dir

    # Check DDP global rank safely
    rank = int(os.environ.get("RANK", "0"))

    if rank == 0 and zip_path.exists():
        print("Extracting Tiny-ImageNet archive cleanly on Rank 0...", flush=True)
        shutil.rmtree(tiny_dir, ignore_errors=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)

        val_dir = tiny_dir / "val"
        images_dir = val_dir / "images"
        annotations = val_dir / "val_annotations.txt"

        if annotations.exists() and images_dir.exists():
            created_dirs = set()
            with open(annotations, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    img_name, class_name = parts[0], parts[1]

                    target_dir = val_dir / class_name
                    if class_name not in created_dirs:
                        target_dir.mkdir(exist_ok=True)
                        created_dirs.add(class_name)

                    src = images_dir / img_name
                    dst = target_dir / img_name
                    if src.exists():
                        os.replace(src, dst)

            shutil.rmtree(images_dir, ignore_errors=True)
            annotations.unlink(missing_ok=True)

        complete_marker.touch()
        print("Tiny-ImageNet extraction complete.", flush=True)

    # Synchronize multi-GPU processes if running under torch.distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    return tiny_dir

def load_imagenet(
    batch_size: int,
    num_workers: int = 4,
    sampler: Optional[DistributedSampler] = None,
    data_dir: Optional[str] = None,
    distributed: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Load Tiny-ImageNet train and validation data loaders."""
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()
    tiny_imagenet_dir = _extract_and_organize_tiny_imagenet(root)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_set = torchvision.datasets.ImageFolder(root=tiny_imagenet_dir / "train", transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(root=tiny_imagenet_dir / "val", transform=transform_test)

    if distributed and sampler is None:
        sampler = DistributedSampler(train_set)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader