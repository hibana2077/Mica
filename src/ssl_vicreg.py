#!/usr/bin/env python
"""Self-Supervised Pretraining with VICReg + Simple Domain Alignment

Implements the core of the experimental design described in docs/idea.md:
 - Joint unsupervised pretraining on Dataset-1 (web) and Dataset-2 (pharmacy)
 - VICReg objective (variance / invariance / covariance)
 - Lightweight domain alignment regularizer on batch statistics (mean + covariance)

Produces an encoder checkpoint that can later be consumed by linear probe / k-NN scripts.

Simplifications / Assumptions (documented so future extensions are easy):
 - Domain alignment currently uses feature mean & covariance Frobenius distance
   (||μ_s - μ_t||^2 + ||C_s - C_t||_F^2). This approximates reducing distributional discrepancy.
 - No pseudo-label prototypes yet; hook points left (TODO) for class-conditional alignment.
 - Ensures each training step sees samples from both domains via paired loaders.

CLI Example:
  python src/ssl_vicreg.py \
      --dataset1 dataset/Dataset_1_Cleaned \
      --dataset2 dataset/Dataset_2_Cleaned \
      --output_dir output/SSL_resnet50 \
      --model_name resnet50.a1_in1k --epochs 200

Checkpoints:
  output_dir/ssl_checkpoint.pth : encoder + projector + optimizer states
  output_dir/ssl_best.pth       : best (lowest loss) snapshot
  output_dir/ssl_log.jsonl      : per-epoch metrics
  output_dir/ssl_config.json    : frozen config
  output_dir/feature_stats.json : final variance / offdiag energy diagnostics

"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm

from utils import set_seed, get_device, save_json


# -------------------------------------------------
# Config
# -------------------------------------------------
@dataclass
class SSLConfig:
    dataset1: str = "dataset/Dataset_1_Cleaned"
    dataset2: str = "dataset/Dataset_2_Cleaned"
    output_dir: str = "output/SSL"

    model_name: str = "resnet50.a1_in1k"
    pretrained: bool = True  # Using ImageNet init can accelerate convergence

    batch_size: int = 256          # Effective total (will be split half per domain)
    workers: int = 4
    epochs: int = 200
    lr: float = 2e-3               # Cosine schedule will be applied
    weight_decay: float = 1e-4
    warmup_epochs: int = 10

    proj_dim: int = 2048           # VICReg projector output dimension
    proj_hidden: int = 8192        # Hidden size of MLP projector

    # VICReg loss weights
    lambda_invariance: float = 25.0
    mu_variance: float = 25.0
    nu_covariance: float = 1.0
    var_gamma: float = 1.0         # Target minimum std per-dimension

    # Domain alignment weight
    lambda_align: float = 1.0

    # Optimization / misc
    seed: int = 42
    amp: bool = True
    log_every: int = 50
    save_every: int = 25
    resume: str | None = None

    # Augmentation (keep moderate to preserve fine-grained texture cues)
    jitter: float = 0.4
    blur_prob: float = 0.1
    gray_prob: float = 0.1


# -------------------------------------------------
# Data utilities
# -------------------------------------------------
def build_ssl_dataloaders(cfg: SSLConfig, model_name: str) -> Tuple[DataLoader, DataLoader, int]:
    """Create two domain dataloaders (each returns a single image w/ transform).

    Returns consistent class count for potential prototype extensions.
    """
    # We only need basic image folder reading; later we apply dual-view augmentation in code.
    dummy_model_cfg = resolve_data_config({}, model=None)  # Not strictly using timm model here

    # Custom augmentation pipeline: we rely on torchvision transforms built by timm convenience.
    train_tfms = create_transform(**dummy_model_cfg, is_training=True)

    ds1 = datasets.ImageFolder(cfg.dataset1, transform=train_tfms)
    ds2 = datasets.ImageFolder(cfg.dataset2, transform=train_tfms)
    num_classes = len(ds1.classes)
    assert num_classes == len(ds2.classes), "Dataset-1 and Dataset-2 must have same class set"

    # Half-batch per domain
    half = cfg.batch_size // 2
    loader1 = DataLoader(ds1, batch_size=half, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    loader2 = DataLoader(ds2, batch_size=half, shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=True)
    return loader1, loader2, num_classes


def dual_view(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate two views; currently identity because transforms already random.
    If stronger augmentations desired, add here.
    """
    return images, images.clone()


# -------------------------------------------------
# Model: backbone + projector
# -------------------------------------------------
def build_encoder_and_projector(cfg: SSLConfig):
    from timm import create_model
    encoder = create_model(cfg.model_name, pretrained=cfg.pretrained, num_classes=0, global_pool="avg")
    feat_dim = getattr(encoder, "num_features", 2048)
    projector = nn.Sequential(
        nn.Linear(feat_dim, cfg.proj_hidden, bias=False),
        nn.BatchNorm1d(cfg.proj_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.proj_hidden, cfg.proj_hidden, bias=False),
        nn.BatchNorm1d(cfg.proj_hidden),
        nn.ReLU(inplace=True),
        nn.Linear(cfg.proj_hidden, cfg.proj_dim, bias=False),
    )
    return encoder, projector, feat_dim


# -------------------------------------------------
# VICReg Loss Components
# -------------------------------------------------
def vicreg_loss(z1: torch.Tensor, z2: torch.Tensor, cfg: SSLConfig):
    """Compute VICReg components.
    z1,z2: (B, D)
    Returns dict with individual components and total (excluding external regularizers).
    """
    # invariance (MSE)
    inv = F.mse_loss(z1, z2)

    def variance_term(z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        penalty = torch.mean(F.relu(cfg.var_gamma - std))
        return penalty, std.detach()

    v1, std1 = variance_term(z1)
    v2, std2 = variance_term(z2)
    var = (v1 + v2) / 2

    # covariance suppression (off-diagonal)
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (z1.shape[0] - 1)
    cov2 = (z2c.T @ z2c) / (z2.shape[0] - 1)
    offdiag1 = cov1.flatten()[1:: z1.shape[1] + 1]  # incorrect method; easier utility below

    def off_diagonal(x: torch.Tensor):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    cov_loss = (off_diagonal(cov1).pow(2).sum() / z1.shape[1]) + (off_diagonal(cov2).pow(2).sum() / z2.shape[1])

    total = cfg.lambda_invariance * inv + cfg.mu_variance * var + cfg.nu_covariance * cov_loss
    return {
        "loss_core": total,
        "inv": inv.detach(),
        "var": var.detach(),
        "cov": cov_loss.detach(),
        "std_mean": torch.cat([std1, std2]).mean().detach(),
    }


# -------------------------------------------------
# Domain Alignment Regularizer
# -------------------------------------------------
def domain_alignment(feats_s: torch.Tensor, feats_t: torch.Tensor):
    """Simple alignment: mean + covariance Frobenius difference.
    feats_*: (B, D)
    Returns scalar loss and diagnostics.
    """
    mu_s = feats_s.mean(0)
    mu_t = feats_t.mean(0)
    cs = (feats_s - mu_s).T @ (feats_s - mu_s) / (feats_s.shape[0] - 1)
    ct = (feats_t - mu_t).T @ (feats_t - mu_t) / (feats_t.shape[0] - 1)
    mean_loss = F.mse_loss(mu_s, mu_t)
    cov_loss = F.mse_loss(cs, ct)
    return mean_loss + cov_loss, mean_loss.detach(), cov_loss.detach()


# -------------------------------------------------
# Scheduler (cosine with warmup)
# -------------------------------------------------
def cosine_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float):
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# -------------------------------------------------
# Training Loop
# -------------------------------------------------
def train(cfg: SSLConfig):
    device = get_device()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_json(os.path.join(cfg.output_dir, "ssl_config.json"), asdict(cfg))

    loader1, loader2, _ = build_ssl_dataloaders(cfg, cfg.model_name)
    encoder, projector, feat_dim = build_encoder_and_projector(cfg)
    encoder.to(device)
    projector.to(device)

    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    global_step = 0
    best_loss = float("inf")
    log_path = os.path.join(cfg.output_dir, "ssl_log.jsonl")
    with open(log_path, "w", encoding="utf-8"):
        pass

    total_steps = cfg.epochs * min(len(loader1), len(loader2))

    for epoch in range(1, cfg.epochs + 1):
        encoder.train(); projector.train()
        iter1 = iter(loader1)
        iter2 = iter(loader2)
        n_steps = min(len(loader1), len(loader2))
        pbar = tqdm(range(n_steps), desc=f"SSL Epoch {epoch}/{cfg.epochs}", leave=False)

        running = {k: 0.0 for k in ["loss", "core", "inv", "var", "cov", "align", "align_mean", "align_cov"]}
        count_samples = 0
        for _ in pbar:
            try:
                imgs_s, _ = next(iter1)
            except StopIteration:
                iter1 = iter(loader1)
                imgs_s, _ = next(iter1)
            try:
                imgs_t, _ = next(iter2)
            except StopIteration:
                iter2 = iter(loader2)
                imgs_t, _ = next(iter2)

            imgs_s = imgs_s.to(device, non_blocking=True)
            imgs_t = imgs_t.to(device, non_blocking=True)
            x = torch.cat([imgs_s, imgs_t], dim=0)
            # two views (placeholder; augmentations already random per sample)
            v1, v2 = dual_view(x)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                z1 = projector(encoder(v1))
                z2 = projector(encoder(v2))
                core = vicreg_loss(z1, z2, cfg)

                # For domain alignment use averaged representation (stopgrad not applied)
                z_avg = 0.5 * (z1 + z2)
                z_s = z_avg[: imgs_s.shape[0]]
                z_t = z_avg[imgs_s.shape[0] :]
                align_raw, align_mean, align_cov = domain_alignment(z_s, z_t)
                align_loss = cfg.lambda_align * align_raw
                loss = core["loss_core"] + align_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

            # Manual cosine warmup LR
            global_step += 1
            lr_now = cosine_warmup(global_step, total_steps, cfg.warmup_epochs * n_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            bsz = x.size(0)
            count_samples += bsz
            running["loss"] += loss.item() * bsz
            running["core"] += core["loss_core"].item() * bsz
            running["inv"] += core["inv"].item() * bsz
            running["var"] += core["var"].item() * bsz
            running["cov"] += core["cov"].item() * bsz
            running["align"] += align_loss.item() * bsz
            running["align_mean"] += align_mean.item() * bsz
            running["align_cov"] += align_cov.item() * bsz

            if global_step % cfg.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{running['loss']/count_samples:.3f}",
                    "inv": f"{running['inv']/count_samples:.3f}",
                    "align": f"{running['align']/count_samples:.3f}",
                    "lr": f"{lr_now:.2e}",
                })

        # Epoch metrics
        epoch_metrics = {k: v / count_samples for k, v in running.items()}
        epoch_metrics.update({"epoch": epoch, "lr": lr_now})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_metrics) + "\n")

        # Save periodic
        if epoch % cfg.save_every == 0:
            torch.save({
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cfg": asdict(cfg),
                "feat_dim": feat_dim,
            }, os.path.join(cfg.output_dir, f"ssl_epoch{epoch}.pth"))

        # Best checkpoint (min loss)
        if epoch_metrics["loss"] < best_loss:
            best_loss = epoch_metrics["loss"]
            torch.save({
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "cfg": asdict(cfg),
                "feat_dim": feat_dim,
            }, os.path.join(cfg.output_dir, "ssl_best.pth"))

    # Feature diagnostics on final encoder
    encoder.eval(); projector.eval()
    with torch.no_grad():
        # Take a single pass over dataset1 for stats (lightweight)
        loader_small = DataLoader(datasets.ImageFolder(cfg.dataset1, transform=create_transform(**resolve_data_config({}, model=None), is_training=False)), batch_size=256, shuffle=False)
        feats = []
        for imgs, _ in loader_small:
            imgs = imgs.to(device)
            f = projector(encoder(imgs))
            feats.append(f.cpu())
        feats = torch.cat(feats, 0)
        std = feats.std(0)
        cov = torch.cov(feats.T)
        offdiag_energy = (cov - torch.diag(torch.diag(cov))).pow(2).mean().item()
        stats = {
            "mean_std": std.mean().item(),
            "frac_above_gamma": (std > cfg.var_gamma).float().mean().item(),
            "offdiag_energy": offdiag_energy,
            "n_samples": feats.size(0),
        }
        save_json(os.path.join(cfg.output_dir, "feature_stats.json"), stats)


def parse_args():
    p = argparse.ArgumentParser(description="VICReg + Domain Alignment SSL")
    for field in SSLConfig.__dataclass_fields__.values():
        name = f"--{field.name}"
        if field.type is bool:
            # Provide --no_* flag for bools
            if getattr(SSLConfig, field.name, False):
                p.add_argument(f"--no_{field.name}", action="store_false", dest=field.name)
            else:
                p.add_argument(name, action="store_true")
        else:
            p.add_argument(name, type=type(field.default), default=field.default)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SSLConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
