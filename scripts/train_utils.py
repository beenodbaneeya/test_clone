"""Shared train/eval loops used by single-GPU and DDP entry points."""

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, int]:
    """
    Train the model for one epoch on a single device.

    Deep learning note:
    This loop performs forward pass, loss computation, backpropagation, and
    optimizer updates. Metrics are tracked as sample-weighted values so epoch
    loss/accuracy remain correct even when final batches differ in size.

    Returns:
        tuple[float, float, int]: Train accuracy, train loss, and number of images processed.
    """
    model.train()
    total_labels = 0
    correct_labels = 0
    loss_total = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # AMP executes selected ops in lower precision and uses GradScaler to
        # avoid numerical underflow during gradient updates.
        if use_amp and scaler is not None and device.type == "cuda":
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
        predictions = outputs.argmax(dim=1)
        total_labels += batch_size
        correct_labels += (predictions == labels).sum().item()
        loss_total += loss.detach().item() * batch_size

    train_accuracy = correct_labels / total_labels
    train_loss = loss_total / total_labels
    return train_accuracy, train_loss, total_labels


def test(
    model: torch.nn.Module,
    test_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float]:
    """
    Evaluate the model on the validation dataset.

    Deep learning note:
    Evaluation runs in ``model.eval()`` + ``torch.no_grad()`` mode to disable
    training-time behavior (e.g., dropout randomness) and to avoid gradient
    tracking overhead.

    Returns:
        tuple[float, float]: Validation accuracy and validation loss.
    """
    model.eval()
    total_labels = 0
    correct_labels = 0
    loss_total = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)

            predictions = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            total_labels += batch_size
            correct_labels += (predictions == labels).sum().item()
            loss_total += loss.detach().item() * batch_size

    val_accuracy = correct_labels / total_labels
    val_loss = loss_total / total_labels
    return val_accuracy, val_loss
