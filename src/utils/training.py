"""
Training and evaluation utilities
"""
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import accuracy


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip_norm=1.0, use_tqdm=True):
    """
    Train model for one epoch
    
    Args:
        model: The model to train
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: GradScaler for mixed precision (optional)
        grad_clip_norm: Gradient clipping norm (0 to disable)
        use_tqdm: Whether to show progress bar
        
    Returns:
        avg_loss: Average training loss
        avg_acc: Average training accuracy
    """
    model.train()
    running_loss = 0.0
    running_top1 = 0.0
    
    if use_tqdm:
        pbar = tqdm(loader, desc="Train", leave=False)
    else:
        pbar = loader
    
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        top1, = accuracy(outputs, targets, topk=(1,))
        running_loss += loss.item() * images.size(0)
        running_top1 += top1 * images.size(0)

        if use_tqdm:
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc@1": f"{top1:.2f}"})

    n = len(loader.dataset)
    return running_loss / n, running_top1 / n


def evaluate(model, loader, criterion, device, use_tqdm=True):
    """
    Evaluate model on validation/test set
    
    Args:
        model: The model to evaluate
        loader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        use_tqdm: Whether to show progress bar
        
    Returns:
        avg_loss: Average validation loss
        avg_acc: Average validation accuracy
    """
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    with torch.no_grad():
        if use_tqdm:
            pbar = tqdm(loader, desc="Eval", leave=False)
        else:
            pbar = loader
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            top1, = accuracy(outputs, targets, topk=(1,))
            running_loss += loss.item() * images.size(0)
            running_top1 += top1 * images.size(0)
            
            if use_tqdm:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc@1": f"{top1:.2f}"})

    n = len(loader.dataset)
    return running_loss / n, running_top1 / n
