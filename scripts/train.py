"""Single-GPU training script for CIFAR-100 and Tiny-ImageNet benchmarks."""

import argparse
import random
import time

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
    """Parse CLI arguments for single-GPU training."""
    parser = argparse.ArgumentParser(
        description="Train a model on a single GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["wideresnet", "vit"],
        default="wideresnet",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cifar100", "tiny-imagenet"],
        default="cifar100",
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
        help="Optional dataset directory override.",
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


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    """Create train/validation data loaders and return the number of classes."""
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
    """Instantiate and move the selected model to the target device."""
    if model_name == "wideresnet":
        return WideResNet(num_classes=num_classes).to(device)
    return ViTModel(num_classes=num_classes).to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> tuple[Optimizer, float]:
    """Build optimizer with architecture-aware defaults."""
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
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:  # adamw
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return optimizer, learning_rate


def set_seed(seed: int, deterministic: bool) -> None:
    """Set Python/Torch random seeds and optional deterministic cuDNN behavior."""
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
    """Run single-GPU training and print speed metrics."""
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
        total_images += epoch_images

        throughput = epoch_images / epoch_time if epoch_time > 0 else 0.0
        val_accuracy, val_loss = evaluate(model, test_loader, loss_fn, device, use_amp=use_amp)
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"time={epoch_time:.3f}s, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, throughput={throughput:.1f} img/s"
        )

        # Early stopping logic
        if len(val_accuracies) >= args.patience and all(
            acc >= args.target_accuracy for acc in val_accuracies[-args.patience :]
        ):
            print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.")
            break

    final_throughput = total_images / total_time if total_time > 0 else 0.0
    if val_accuracies:
        print(f"\nTraining complete. Final val_acc: {val_accuracies[-1]:.4f}")
        print(f"Total time: {total_time:.1f}s, final throughput: {final_throughput:.1f} img/s")


if __name__ == "__main__":
    main()