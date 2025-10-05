#!/usr/bin/env python
"""
Kaggle 20-class Image Classification with timm
- Train on /kaggle/input/nb-tcm-chm/Dataset_1_Cleaned
- Test/Val on /kaggle/input/nb-tcm-chm/Dataset_2_Cleaned

This script uses an in-script CONFIG (overridable via CLI args) and
PyTorch + timm to fine-tune an image classification model.

Example (defaults are already set for Kaggle paths):
    python kaggle_timm_20class_train.py \
        --model_name tf_efficientnet_b0_ns \
        --epochs 15 --batch_size 32 --lr 3e-4

Outputs (by default):
- /kaggle/working/best_model.pth          (best val acc checkpoint)
- /kaggle/working/train_log.jsonl         (per-epoch metrics)
- /kaggle/working/run_config.json         (frozen config)

Requires: torch, torchvision, timm, tqdm
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from utils import (
    set_seed,
    get_device,
    save_json,
    build_dataloaders,
    build_model,
    evaluate,
    apply_mixup,
)
from utils.data import accuracy

# ------------------------------
# In-script CONFIG (defaults)
# ------------------------------
@dataclass
class Config:
    train_dir: str = "/kaggle/input/nb-tcm-chm/Dataset_1_Cleaned"
    test_dir: str = "/kaggle/input/nb-tcm-chm/Dataset_2_Cleaned"
    output_dir: str = "/kaggle/working"

    model_name: str = "tf_efficientnet_b0_ns"  # e.g., resnet50, vit_base_patch16_224, tf_efficientnet_b0_ns
    pretrained: bool = True
    num_classes: int = 20

    epochs: int = 15
    batch_size: int = 32
    workers: int = 2  # Kaggle often limits to 2 for reliability; feel free to raise

    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1

    # Regularization / Augment (basic). Mixup/CutMix can be enabled if desired.
    mixup_alpha: float = 0.0  # set >0 to enable
    cutmix_alpha: float = 0.0  # set >0 to enable
    dropout: float = 0.0       # additional dropout on classifier if supported by model

    grad_clip_norm: float = 1.0
    early_stop_patience: int = 7

    seed: int = 42
    amp: bool = True  # Automatic Mixed Precision
    use_tqdm: bool = True  # Enable/disable tqdm progress bars


def parse_args():
    """Parse command line arguments to override Config defaults."""
    parser = argparse.ArgumentParser(
        description="Kaggle 20-class Image Classification with timm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data paths
    parser.add_argument("--train_dir", type=str, default="/kaggle/input/nb-tcm-chm/Dataset_1_Cleaned",
                        help="Path to training dataset directory")
    parser.add_argument("--test_dir", type=str, default="/kaggle/input/nb-tcm-chm/Dataset_2_Cleaned",
                        help="Path to test/validation dataset directory")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working",
                        help="Directory to save outputs (model, logs, config)")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="tf_efficientnet_b0_ns",
                        help="Model name from timm (e.g., resnet50, vit_base_patch16_224)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                        help="Don't use pretrained weights")
    parser.add_argument("--num_classes", type=int, default=20,
                        help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate for classifier")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for optimizer")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    
    # Augmentation & regularization
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup alpha parameter (0 to disable)")
    parser.add_argument("--cutmix_alpha", type=float, default=0.0,
                        help="CutMix alpha parameter (0 to disable)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--early_stop_patience", type=int, default=7,
                        help="Early stopping patience (0 to disable)")
    
    # Others
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision")
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="Disable automatic mixed precision")
    parser.add_argument("--tqdm", action="store_true", default=True,
                        help="Enable tqdm progress bars")
    parser.add_argument("--no_tqdm", dest="use_tqdm", action="store_false",
                        help="Disable tqdm progress bars, only show epoch summary")
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create Config from args
    cfg = Config(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        dropout=args.dropout,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        amp=args.amp,
        use_tqdm=args.use_tqdm,
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_json(os.path.join(cfg.output_dir, "run_config.json"), asdict(cfg))

    set_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Build model + loaders
    model = build_model(cfg)
    model.to(device)

    train_loader, test_loader, class_names = build_dataloaders(cfg, model)
    print(f"Classes ({len(class_names)}): {class_names}")

    # Criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # Optimizer & Cosine LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Cosine schedule to near-zero over all epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    # Optionally set up Mixup/CutMix (manual lightweight version)
    do_mixup = cfg.mixup_alpha > 0.0 or cfg.cutmix_alpha > 0.0

    best_acc = -1.0
    epochs_no_improve = 0
    log_path = os.path.join(cfg.output_dir, "train_log.jsonl")

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        # Training
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        
        if cfg.use_tqdm:
            pbar = tqdm(train_loader, desc=f"Train[{epoch}]", leave=False)
        else:
            pbar = train_loader
        
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if do_mixup:
                images, targets = apply_mixup(images, targets, cfg.mixup_alpha, cfg.cutmix_alpha)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                if cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            top1, = accuracy(outputs, targets, topk=(1,))
            running_loss += loss.item() * images.size(0)
            running_top1 += top1 * images.size(0)
            
            if cfg.use_tqdm:
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc@1": f"{top1:.2f}"})

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_top1 / len(train_loader.dataset)

        # Validation on test_dir
        val_loss, val_acc = evaluate(model, test_loader, criterion, device, cfg.use_tqdm)
        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2f} | val_loss={val_loss:.4f} val_acc={val_acc:.2f}")

        # log
        save_json(
            log_path,
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc},
        )  # this overwrites - keep a single last snapshot
        # also append a jsonl for history
        with open(os.path.join(cfg.output_dir, "train_log.jsonl"), "a", encoding="utf-8") as jf:
            jf.write(json.dumps({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}) + "\n")

        # Early stopping & checkpointing
        improved = val_acc > best_acc
        if improved:
            best_acc = val_acc
            epochs_no_improve = 0
            ckpt_path = os.path.join(cfg.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_acc": best_acc,
                "class_names": class_names,
                "config": asdict(cfg),
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path} (acc={best_acc:.2f})")
        else:
            epochs_no_improve += 1

        if cfg.early_stop_patience > 0 and epochs_no_improve >= cfg.early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs. Best acc={best_acc:.2f}")
            break

    print("Training complete.")


if __name__ == "__main__":
    main()