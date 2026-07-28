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
CIFAR100_ARCHIVE = "cifar-100-python.tar.gz"
CIFAR100_DIRNAME = "cifar-100-python"


def _download_cifar100(data_dir: Path, verbose: bool = True) -> Path:
    """Download and extract CIFAR-100 if needed."""
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
            "Please check network connection or stage the dataset manually in the datasets directory."
        ) from error

    if verbose:
        print(f"Extracting CIFAR-100 dataset to {data_dir}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    archive_path.unlink(missing_ok=True)
    return cifar_dir


def _data_dir_default() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_cifar100(
    batch_size: int,
    num_workers: int = 0,
    sampler: Optional[DistributedSampler] = None,
    data_dir: Optional[str] = None,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Loads the CIFAR-100 dataset. Creates the dataset directory to store the dataset during runtime.
    """
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()
    _download_cifar100(root, verbose=verbose)
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    # Load full datasets
    train_set = torchvision.datasets.CIFAR100(
        root=str(root), download=False, train=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(
        root=str(root), download=False, train=False, transform=transform_test)
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, drop_last=True, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def _download_tiny_imagenet(data_dir: Path, verbose: bool = True) -> Path:
    """
    Downloads and extracts the Tiny-ImageNet dataset if not already present.
    """
    tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    tiny_imagenet_zip = data_dir / "tiny-imagenet-200.zip"
    tiny_imagenet_dir = data_dir / "tiny-imagenet-200"
    # Check if the dataset is already downloaded and extracted
    if tiny_imagenet_dir.exists():
        if verbose:
            print(f"Tiny-ImageNet dataset already exists at {tiny_imagenet_dir}. Skipping download.")
        return tiny_imagenet_dir
    # Download the dataset
    if verbose:
        print(f"Downloading Tiny-ImageNet dataset from {tiny_imagenet_url}...")
    response = requests.get(tiny_imagenet_url, stream=True)
    with open(tiny_imagenet_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    # Extract the dataset
    if verbose:
        print(f"Extracting Tiny-ImageNet dataset to {data_dir}...")
    with zipfile.ZipFile(tiny_imagenet_zip, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    # Remove the zip file after extraction
    os.remove(tiny_imagenet_zip)
    return tiny_imagenet_dir


def load_imagenet(
    batch_size: int,
    num_workers: int = 0,
    sampler: Optional[DistributedSampler] = None,
    data_dir: Optional[str] = None,
    distributed: bool = False,
    verbose: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Loads the Tiny-ImageNet dataset. Downloads the dataset if it is not already present.
    """
    root = Path(data_dir).expanduser().resolve() if data_dir else _data_dir_default()
    # Download Tiny-ImageNet if not already present
    tiny_imagenet_dir = _download_tiny_imagenet(root, verbose=verbose)
    # Define transformations for training and validation.
    # Validation is intentionally deterministic (no random transforms).
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
    # Define paths for train and validation datasets
    train_dir = tiny_imagenet_dir / "train"
    val_dir = tiny_imagenet_dir / "val"
    # Fix Tiny-ImageNet validation directory structure if necessary
    _fix_tiny_imagenet_val_structure(val_dir, verbose=verbose)
    # Load datasets
    train_set = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
    test_set = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)
    # Apply DistributedSampler to the training Dataset
    if distributed and sampler is None:
        sampler = DistributedSampler(train_set)
    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, drop_last=True, shuffle=(sampler is None), sampler=sampler, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def _fix_tiny_imagenet_val_structure(val_dir: Path, verbose: bool = True) -> None:
    """
    Fixes the validation directory structure of Tiny-ImageNet to match ImageFolder format.
    """
    images_dir = val_dir / "images"
    if not images_dir.exists():
        return  # Structure is already fixed
    if verbose:
        print(f"Fixing Tiny-ImageNet validation directory structure at {val_dir}...")
    with open(val_dir / "val_annotations.txt", "r") as f:
        annotations = f.readlines()
    for line in annotations:
        parts = line.strip().split("\t")
        img_name, class_name = parts[0], parts[1]
        class_dir = val_dir / class_name
        class_dir.mkdir(exist_ok=True)
        img_path = images_dir / img_name
        img_path.rename(class_dir / img_name)
    # Remove the old "images" directory and annotations file
    images_dir.rmdir()
    (val_dir / "val_annotations.txt").unlink()






