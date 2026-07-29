"""Dataset download/loading utilities for CIFAR-100 and Tiny-ImageNet."""

import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_DIRNAME = "cifar-100-python"


def _data_dir_default() -> Path:
    """Return default dataset directory at <repo>/datasets."""
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _download_cifar100(data_dir: Path, verbose: bool = True) -> Path:
    """Download and extract CIFAR-100 dataset using requests to avoid cluster SSL issues."""
    cifar_dir = data_dir / CIFAR100_DIRNAME
    if cifar_dir.exists():
        if verbose:
            print(f"CIFAR-100 dataset already exists at {cifar_dir}. Skipping download.")
        return cifar_dir

    archive_path = data_dir / CIFAR100_ARCHIVE
    url = "https://data.brainchip.com/dataset-mirror/cifar100/cifar-100-python.tar.gz"

    if verbose:
        print(f"Downloading CIFAR-100 dataset from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    except Exception as error:
        if archive_path.exists():
            archive_path.unlink()
        raise RuntimeError(
            f"Failed to download CIFAR-100: {error}\n"
            "Please check network connection or stage dataset manually in datasets directory."
        ) from error

    if verbose:
        print(f"Extracting CIFAR-100 dataset to {data_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    archive_path.unlink(missing_ok=True)
    return cifar_dir


def load_cifar100(
    batch_size: int,
    num_workers: int = 0,
    sampler: Optional[DistributedSampler] = None,
    data_dir: Optional[str] = None,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Load CIFAR-100 train and test data loaders."""
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()
    _download_cifar100(root, verbose=verbose)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader


def _download_tiny_imagenet(data_dir: Path, verbose: bool = True) -> Path:
    """Download and extract Tiny-ImageNet dataset if not present."""
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    tiny_imagenet_zip = data_dir / "tiny-imagenet-200.zip"
    tiny_imagenet_dir = data_dir / "tiny-imagenet-200"

    if tiny_imagenet_dir.exists():
        if verbose:
            print(f"Tiny-ImageNet dataset already exists at {tiny_imagenet_dir}. Skipping download.")
        return tiny_imagenet_dir

    if verbose:
        print(f"Downloading Tiny-ImageNet dataset from {tiny_imagenet_url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(tiny_imagenet_url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    with open(tiny_imagenet_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if verbose:
        print(f"Extracting Tiny-ImageNet dataset to {data_dir}...")
    with zipfile.ZipFile(tiny_imagenet_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    os.remove(tiny_imagenet_zip)
    return tiny_imagenet_dir


def _fix_tiny_imagenet_val_structure(val_dir: Path, verbose: bool = True) -> None:
    """Reorganize Tiny-ImageNet validation directory into ImageFolder format."""
    images_dir = val_dir / "images"
    if not images_dir.exists():
        return

    if verbose:
        print(f"Fixing Tiny-ImageNet validation directory structure at {val_dir}...")
    with open(val_dir / "val_annotations.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            img_name, class_name = parts[0], parts[1]
            class_dir = val_dir / class_name
            class_dir.mkdir(exist_ok=True)
            (images_dir / img_name).rename(class_dir / img_name)

    images_dir.rmdir()
    (val_dir / "val_annotations.txt").unlink(missing_ok=True)



def load_imagenet(
    batch_size: int,
    num_workers: int = 0,
    sampler: Optional[DistributedSampler] = None,
    data_dir: Optional[str] = None,
    distributed: bool = False,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Load Tiny-ImageNet train and validation data loaders."""
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()

    tiny_imagenet_dir = _download_tiny_imagenet(root, verbose=verbose)
    _fix_tiny_imagenet_val_structure(tiny_imagenet_dir / "val", verbose=verbose)

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

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader