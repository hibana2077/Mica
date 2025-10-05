"""
Data loading and dataset utilities
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.data import resolve_data_config, create_transform


def build_dataloaders(cfg, model: nn.Module) -> Tuple[DataLoader, DataLoader, list]:
    """
    Build training and test data loaders
    
    Args:
        cfg: Configuration object with train_dir, test_dir, batch_size, workers, num_classes
        model: The model to resolve data config from
        
    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing/validation
        class_names: List of class names
    """
    # Resolve timm data config to get correct input size/mean/std/interp
    data_cfg = resolve_data_config({}, model=model)

    train_tfms = create_transform(**data_cfg, is_training=True)
    test_tfms = create_transform(**data_cfg, is_training=False)

    train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(cfg.test_dir, transform=test_tfms)

    class_names = train_ds.classes
    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"Detected {len(class_names)} classes in train_dir, but cfg.num_classes={cfg.num_classes}.\n"
            f"Classes: {class_names}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=max(1, cfg.batch_size * 2),
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, test_loader, class_names


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    Computes the accuracy over the k top predictions
    
    Args:
        output: Model predictions (logits)
        target: Ground truth labels
        topk: Tuple of k values for top-k accuracy
        
    Returns:
        List of top-k accuracies
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res
