"""
Training and evaluation utilities
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

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


def evaluate(model, loader, criterion, device, use_tqdm=True, compute_metrics=True):
    """
    Evaluate model on validation/test set
    
    Args:
        model: The model to evaluate
        loader: Validation/test data loader
        criterion: Loss function
        device: Device to evaluate on
        use_tqdm: Whether to show progress bar
        compute_metrics: Whether to compute detailed metrics (F1, Precision, Recall, AUC)
        
    Returns:
        metrics_dict: Dictionary containing:
            - loss: Average validation loss
            - acc: Average validation accuracy (top-1)
            - precision: Macro-averaged precision (if compute_metrics=True)
            - recall: Macro-averaged recall (if compute_metrics=True)
            - f1: Macro-averaged F1-score (if compute_metrics=True)
            - auc: Macro-averaged AUC-ROC (if compute_metrics=True)
    """
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
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
            
            if compute_metrics:
                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            
            if use_tqdm:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc@1": f"{top1:.2f}"})

    n = len(loader.dataset)
    avg_loss = running_loss / n
    avg_acc = running_top1 / n
    
    metrics_dict = {
        "loss": avg_loss,
        "acc": avg_acc,
    }
    
    if compute_metrics:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Compute precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='macro', zero_division=0
        )
        
        # Compute AUC-ROC (macro-averaged, one-vs-rest)
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
        except ValueError:
            # If not all classes are present, set AUC to 0
            auc = 0.0
        
        metrics_dict.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        })
    
    return metrics_dict
