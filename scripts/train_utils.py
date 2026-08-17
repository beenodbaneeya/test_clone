"""Shared train/eval loops used by single GPU PyTorch execution."""

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# Enable Tensor Cores for FP32 matrix multiplication
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


def train(
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, int]:
    """Train the model for one epoch on a single device.

    Returns:
        tuple[float, float, int]: Train accuracy, train loss, and total images processed.
    """
    model.train()
    total_labels = 0
    correct_labels = 0
    loss_total = 0.0

    # Evaluate AMP eligibility ONCE outside the batch loop to avoid per-iteration overhead
    is_amp_active = use_amp and scaler is not None and device.type == "cuda"

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if is_amp_active:
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
        total_labels += batch_size
        correct_labels += (outputs.argmax(dim=1) == labels).sum().item()
        loss_total += loss.detach().item() * batch_size

    train_accuracy = correct_labels / total_labels if total_labels > 0 else 0.0
    train_loss = loss_total / total_labels if total_labels > 0 else 0.0
    return train_accuracy, train_loss, total_labels


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    """Evaluate the model on the validation dataset.

    Returns:
        tuple[float, float]: Validation accuracy and validation loss.
    """
    model.eval()
    total_labels = 0
    correct_labels = 0
    loss_total = 0.0

    # Evaluate AMP eligibility ONCE outside the loop
    is_amp_active = use_amp and device.type == "cuda"

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_amp_active:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            batch_size = labels.size(0)
            total_labels += batch_size
            correct_labels += (outputs.argmax(dim=1) == labels).sum().item()
            loss_total += loss.detach().item() * batch_size

    val_accuracy = correct_labels / total_labels if total_labels > 0 else 0.0
    val_loss = loss_total / total_labels if total_labels > 0 else 0.0
    return val_accuracy, val_loss