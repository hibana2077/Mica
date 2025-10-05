"""
Data augmentation utilities
"""
import math
import random

import torch


def apply_mixup(images, targets, mixup_alpha=0.0, cutmix_alpha=0.0):
    """
    Apply MixUp or CutMix augmentation
    
    Args:
        images: Input images tensor
        targets: Target labels tensor
        mixup_alpha: MixUp alpha parameter (0 to disable)
        cutmix_alpha: CutMix alpha parameter (0 to disable)
        
    Returns:
        mixed_images: Augmented images
        mixed_targets: Augmented targets
    """
    if mixup_alpha > 0.0:
        lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
        indices = torch.randperm(images.size(0)).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[indices]
        # For CE with soft labels you'd need BCE or KL; here we approximate by choosing one label
        mixed_targets = targets.clone()
        mask = torch.rand_like(targets.float()) < (1 - lam)
        mixed_targets[mask] = targets[indices][mask]
        return mixed_images, mixed_targets
    elif cutmix_alpha > 0.0:
        lam = torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        indices = torch.randperm(images.size(0)).to(images.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
        images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        mixed_targets = targets.clone()
        # same simple label switch heuristic
        mixed_targets = torch.where(torch.rand_like(targets.float()) < 0.5, targets, targets[indices])
        return images, mixed_targets
    else:
        return images, targets


def rand_bbox(size, lam):
    """
    Generate random bounding box for CutMix
    
    Args:
        size: Size of the image tensor (B, C, H, W)
        lam: Lambda parameter for mixing ratio
        
    Returns:
        bbx1, bby1, bbx2, bby2: Bounding box coordinates
    """
    W = size[3]
    H = size[2]
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform center
    cx = random.randint(0, W)
    cy = random.randint(0, H)
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)
    return bbx1, bby1, bbx2, bby2
