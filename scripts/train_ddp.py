"""Distributed Data Parallel (DDP) training script for PyTorch."""

import argparse
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Enable GH200 Tensor Cores for FP32 math operations
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

from dataset_utils import load_cifar100, load_imagenet
from model import ViTModel, WideResNet


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for DDP training."""
    parser = argparse.ArgumentParser(
        description="Distributed Data Parallel training script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, choices=["wideresnet", "vit"], default="wideresnet")
    parser.add_argument("--dataset", type=str, choices=["cifar100", "tiny-imagenet"], default="cifar100")
    parser.add_argument("--batch-size", type=int, default=2048, help="Global batch size across all GPUs")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--base-lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--optimizer", type=str, choices=["auto", "sgd", "adam", "adamw"], default="auto")
    parser.add_argument("--target-accuracy", type=float, default=0.95, help="Early stopping target validation accuracy")
    parser.add_argument("--patience", type=int, default=2, help="Consecutive epochs required for early stopping")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers per process")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional dataset root path. Defaults to <repo>/datasets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic cuDNN behavior")
    return parser.parse_args()


def ddp_setup() -> tuple[int, int, int, torch.device]:
    """Initialize distributed process group cleanly from torchrun environment variables."""
    if not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA GPUs.")

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Explicitly set local CUDA device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Standard clean initialization without forcing early device_id locking
    dist.init_process_group(backend="nccl")
    return local_rank, global_rank, world_size, device


def set_seed(seed: int, deterministic: bool, rank: int) -> None:
    """Set rank-specific seed for reproducible worker streams."""
    final_seed = seed + rank
    random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def build_dataloaders(
    args: argparse.Namespace,
    global_rank: int,
    world_size: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler, int]:
    """Build DDP-aware train/validation loaders safely from local datasets."""
    if args.batch_size % world_size != 0:
        raise ValueError(f"Global batch size ({args.batch_size}) must be divisible by world size ({world_size}).")

    per_gpu_batch_size = args.batch_size // world_size

    if args.dataset == "cifar100":
        base_train_loader, base_val_loader = load_cifar100(
            batch_size=per_gpu_batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
        )
        num_classes = 100
    else:
        base_train_loader, base_val_loader = load_imagenet(
            batch_size=per_gpu_batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            distributed=False,
        )
        num_classes = 200

    train_set = base_train_loader.dataset
    val_set = base_val_loader.dataset

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=False,
        drop_last=False,
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=per_gpu_batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=per_gpu_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, train_sampler, num_classes


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Instantiate and move the selected model to the local CUDA device."""
    if model_name == "wideresnet":
        return WideResNet(num_classes=num_classes).to(device)
    return ViTModel(num_classes=num_classes).to(device)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> tuple[Optimizer, float]:
    """Build optimizer with architecture aware defaults."""
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
    else:
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    return optimizer, learning_rate


def train_one_epoch(
    model: DDP,
    optimizer: Optimizer,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float, int]:
    """Run one local-rank training epoch and return local metric sums."""
    model.train()
    local_correct = 0.0
    local_loss_sum = 0.0
    local_total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        local_total += batch_size
        local_correct += (outputs.argmax(dim=1) == labels).sum().item()
        local_loss_sum += loss.detach().item() * batch_size

    return local_correct, local_loss_sum, local_total


def evaluate(
    model: DDP,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, int]:
    """Run one local-rank validation pass and return local metric sums."""
    model.eval()
    local_correct = 0.0
    local_loss_sum = 0.0
    local_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            local_total += batch_size
            local_correct += (outputs.argmax(dim=1) == labels).sum().item()
            local_loss_sum += loss.detach().item() * batch_size

    return local_correct, local_loss_sum, local_total


def all_reduce_metrics(correct: float, loss_sum: float, total: int, device: torch.device) -> tuple[float, float, int]:
    """All-reduce local metric sums across all GPUs and compute global accuracy/loss."""
    metrics = torch.tensor([correct, loss_sum, float(total)], dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    global_correct, global_loss_sum, global_total = metrics.tolist()
    total_int = int(global_total)
    accuracy = global_correct / global_total if global_total > 0 else 0.0
    mean_loss = global_loss_sum / global_total if global_total > 0 else 0.0
    return accuracy, mean_loss, total_int


def main_worker() -> None:
    """Run distributed training with synchronized reporting and stopping."""
    args = parse_args()
    local_rank, global_rank, world_size, device = ddp_setup()

    try:
        set_seed(args.seed, args.deterministic, global_rank)
        train_loader, val_loader, train_sampler, num_classes = build_dataloaders(args, global_rank, world_size)

        model = build_model(args.model, num_classes, device)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        optimizer, learning_rate = build_optimizer(args, model)
        loss_fn = nn.CrossEntropyLoss()
        use_amp = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        if global_rank == 0:
            gpus_per_node = torch.cuda.device_count()
            nodes = max(1, world_size // gpus_per_node)
            per_gpu_batch_size = args.batch_size // world_size
            print(
                f"Training {args.model} on {args.dataset} with PyTorch DDP | "
                f"world_size={world_size}, nodes={nodes}, gpus_per_node={gpus_per_node}",
                flush=True,
            )
            print(
                f"global_batch_size={args.batch_size}, per_gpu_batch_size={per_gpu_batch_size}, "
                f"lr={learning_rate}, optimizer={optimizer.__class__.__name__}, "
                f"amp={use_amp}, seed={args.seed}, deterministic={args.deterministic}",
                flush=True,
            )

        val_accuracies: list[float] = []
        total_time = 0.0
        total_images = 0

        for epoch in range(args.epochs):
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)

            t0 = time.time()
            train_correct, train_loss_sum, train_total = train_one_epoch(
                model,
                optimizer,
                train_loader,
                loss_fn,
                device,
                use_amp,
                scaler,
            )
            epoch_time_local = time.time() - t0

            train_acc, train_loss, global_train_total = all_reduce_metrics(
                train_correct,
                train_loss_sum,
                train_total,
                device,
            )
            val_correct, val_loss_sum, val_total = evaluate(model, val_loader, loss_fn, device, use_amp)
            val_acc, val_loss, _ = all_reduce_metrics(val_correct, val_loss_sum, val_total, device)

            # Max epoch time across ranks reflects true synchronized step duration
            epoch_time_tensor = torch.tensor(epoch_time_local, dtype=torch.float64, device=device)
            dist.all_reduce(epoch_time_tensor, op=dist.ReduceOp.MAX)
            epoch_time = epoch_time_tensor.item()

            total_time += epoch_time
            total_images += global_train_total
            throughput = global_train_total / epoch_time if epoch_time > 0 else 0.0

            should_stop = False
            if global_rank == 0:
                val_accuracies.append(val_acc)
                print(
                    f"Epoch {epoch + 1}/{args.epochs}: "
                    f"time={epoch_time:.3f}s, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, throughput={throughput:.1f} img/s",
                    flush=True,
                )
                should_stop = len(val_accuracies) >= args.patience and all(
                    acc >= args.target_accuracy for acc in val_accuracies[-args.patience :]
                )
                if should_stop:
                    print(f"Target accuracy reached. Early stopping after epoch {epoch + 1}.", flush=True)

            # Synchronize early-stop decision across all DDP ranks
            stop_tensor = torch.tensor(1 if should_stop else 0, device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() == 1:
                break

        if global_rank == 0:
            final_throughput = total_images / total_time if total_time > 0 else 0.0
            print("\nTraining Summary:", flush=True)
            print(f"Total training time: {total_time:.3f} seconds", flush=True)
            print(f"Throughput: {final_throughput:.3f} images/second", flush=True)
            print(f"Total GPUs used: {world_size}", flush=True)
            print("Training completed successfully.", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main_worker()