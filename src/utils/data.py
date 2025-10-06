"""
Data loading and dataset utilities
"""
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.data import resolve_data_config, create_transform


def build_dataloaders(cfg, model: nn.Module) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """Build training, validation (optional) and test data loaders.

    If cfg.val_split > 0, a portion of Dataset-1 (train_dir) is held out as
    validation (E0 design). The Dataset-2 directory always serves as the
    cross-source test set. If val_split == 0, the validation loader points to
    the test_dir (legacy behavior) and the returned test_loader duplicates the
    validation loader for compatibility.
    """
    # Resolve timm data config to get correct input size/mean/std/interp
    data_cfg = resolve_data_config({}, model=model)

    train_tfms = create_transform(**data_cfg, is_training=True)
    test_tfms = create_transform(**data_cfg, is_training=False)

    full_train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
    test_ds = datasets.ImageFolder(cfg.test_dir, transform=test_tfms)

    class_names = full_train_ds.classes
    if len(class_names) != cfg.num_classes:
        raise ValueError(
            f"Detected {len(class_names)} classes in train_dir, but cfg.num_classes={cfg.num_classes}.\n"
            f"Classes: {class_names}"
        )
    if getattr(cfg, "val_split", 0) and cfg.val_split > 0:
        # Stratified split not directly available for ImageFolder; approximate by per-class indices
        targets = [s[1] for s in full_train_ds.samples]
        targets = torch.tensor(targets)
        val_indices = []
        train_indices = []
        for cls in range(len(class_names)):
            cls_idx = (targets == cls).nonzero(as_tuple=True)[0]
            n_cls = len(cls_idx)
            n_val = max(1, int(n_cls * cfg.val_split))
            perm = torch.randperm(n_cls)
            val_sel = cls_idx[perm[:n_val]]
            train_sel = cls_idx[perm[n_val:]]
            val_indices.extend(val_sel.tolist())
            train_indices.extend(train_sel.tolist())
        train_subset = torch.utils.data.Subset(full_train_ds, train_indices)
        val_subset = torch.utils.data.Subset(full_train_ds, val_indices)
        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=max(1, cfg.batch_size * 2), shuffle=False, num_workers=cfg.workers, pin_memory=True)
    else:
        train_loader = DataLoader(full_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
        # Use test_ds as validation directly
        val_loader = DataLoader(test_ds, batch_size=max(1, cfg.batch_size * 2), shuffle=False, num_workers=cfg.workers, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=max(1, cfg.batch_size * 2), shuffle=False, num_workers=cfg.workers, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names


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
