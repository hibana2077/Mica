"""Self-Supervised & Manifold Consistency Regularization training script.

Implements the (A) self-supervised backbone with a SimCLR-like objective and
optionally the (B) Manifold Consistency Regularization (MCR) losses described
in `docs/idea.md`.

Three modes controlled by --mode:
  supervised  -> fall back to standard classification (delegates to run.py style)
  ssl         -> SimCLR only (no labels except for evaluation)
  mcr         -> SimCLR + intra-source geodesic preservation + cross-source alignment

Outputs geometry metrics report after training if --eval-geometry is set.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np

from utils import set_seed, get_device, save_json, build_model, compute_model_complexity
from utils.geometry import evaluate_geometry


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SSLConfig:
    train_dir: str = "./Dataset_1_Cleaned"  # web
    test_dir: str = "./Dataset_2_Cleaned"   # phone
    output_dir: str = "./outputs_ssl"
    model_name: str = "resnet18"  # smaller backbone default
    pretrained: bool = False
    num_classes: int = 20
    epochs: int = 50
    batch_size: int = 64
    workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    proj_dim: int = 128
    temperature: float = 0.2
    mode: str = "ssl"  # supervised|ssl|mcr
    # MCR specific
    mcr_intra_k: int = 10
    mcr_intra_mode: str = "distance"  # 'distance' | 'laplacian'
    mcr_lap_sigma: float = 0.0  # if <=0 auto-estimate via median neighbor distance
    mcr_lambda_intra: float = 1.0
    mcr_lambda_cross: float = 1.0
    mcr_cross_k: int = 5
    # Generic
    seed: int = 42
    amp: bool = True
    eval_geometry: bool = True
    geometry_k: int = 10
    geometry_max_nodes: int = 2000


# ---------------------------------------------------------------------------
# Data pipeline for SimCLR style augmentations
# ---------------------------------------------------------------------------
def get_ssl_transforms(size: int = 224):
    aug = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    basic = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return aug, basic


class TwoCropWrapper(torch.utils.data.Dataset):
    """Return two augmented views AND the label (for evaluation)"""
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label


def build_ssl_loaders(cfg: SSLConfig) -> Tuple[DataLoader, DataLoader, list]:
    aug, basic = get_ssl_transforms()
    train_base = datasets.ImageFolder(cfg.train_dir, transform=None)
    test_base = datasets.ImageFolder(cfg.test_dir, transform=basic)
    train_dataset = TwoCropWrapper(train_base, aug)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, drop_last=True)
    test_loader = DataLoader(test_base, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
    return train_loader, test_loader, train_base.classes


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, proj_dim, bias=True),
        )

    def forward(self, x):
        return self.net(x)


def simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy (NT-Xent)."""
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)  # (2b,2b)
    mask = torch.eye(2 * b, device=z.device).bool()
    sim = sim / temperature
    # positives: (i, i+b) and (i+b, i)
    positives = torch.cat([torch.diag(sim, b), torch.diag(sim, -b)])
    logits = sim[~mask].view(2 * b, -1)
    labels = torch.arange(b, device=z.device)
    labels = torch.cat([labels, labels])
    loss = F.cross_entropy(logits, labels)
    return loss


# ---------------------------------------------------------------------------
# MCR Losses
# ---------------------------------------------------------------------------
def compute_intra_geodesic_loss(features: torch.Tensor, k: int = 10, mode: str = "distance", sigma: float = 0.0) -> torch.Tensor:
    """Intra-source manifold regularization.

    Modes:
      - distance: MSE between batch local pairwise distances (simple proxy)
      - laplacian: Graph Laplacian smoothness Tr(F^T L F) encouraging neighbor closeness along k-NN graph.

    Args:
        features: (B, D) feature tensor (will not be grad-detached here; caller can decide)
        k: number of neighbors (excluding self)
        mode: 'distance' or 'laplacian'
        sigma: bandwidth for Laplacian weights; if <=0 auto-estimate via median of neighbor distances
    """
    B = features.size(0)
    if B <= 2:
        return torch.tensor(0.0, device=features.device)
    # Pairwise distances (L2)
    dmat = torch.cdist(features, features, p=2)  # (B,B)
    if mode == "distance":
        with torch.no_grad():
            knn = torch.topk(dmat, min(k + 1, B), largest=False).indices[:, 1:]
        target = dmat.detach()
        losses = []
        for i in range(B):
            j_idx = knn[i]
            losses.append((dmat[i, j_idx] - target[i, j_idx]) ** 2)
        if not losses:
            return torch.tensor(0.0, device=features.device)
        return torch.mean(torch.cat(losses))
    elif mode == "laplacian":
        k_eff = min(k + 1, B)
        dvals, idx = torch.topk(dmat, k_eff, largest=False)  # includes self distance 0 first
        # drop self
        dvals = dvals[:, 1:]
        idx = idx[:, 1:]
        if sigma <= 0:
            # median of neighbor distances
            sigma_est = torch.median(dvals.detach())
            sigma_eff = sigma_est.clamp_min(1e-6)
        else:
            sigma_eff = torch.tensor(sigma, device=features.device)
        # weights w_ij = exp(-d^2 / (2 sigma^2))
        w = torch.exp(- (dvals ** 2) / (2 * sigma_eff ** 2))  # (B,k)
        # Laplacian smoothness ~ sum w_ij ||f_i - f_j||^2
        # Expand neighbor feature differences
        f_exp = features.unsqueeze(1).expand(-1, k_eff - 1, -1)  # (B,k,D)
        f_neigh = features[idx]
        diff2 = (f_exp - f_neigh).pow(2).sum(-1)  # (B,k)
        smooth = (w * diff2).sum() / (B * (k_eff - 1))
        return smooth
    else:
        raise ValueError(f"Unknown intra geodesic mode: {mode}")


def compute_cross_alignment_loss(z: torch.Tensor, labels: torch.Tensor, sources: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Cross-source alignment: for each sample find a same-class sample in other source
    and enforce local tangent alignment via cosine similarity of neighbor displacement.
    Simple proxy: encourage mean feature of class per source to match.
    """
    unique_classes = labels.unique()
    loss = 0.0
    count = 0
    for c in unique_classes:
        mask_c = labels == c
        for s in [0, 1]:
            mask_cs = mask_c & (sources == s)
            if mask_cs.sum() < 1:
                continue
        if not ((mask_c & (sources == 0)).any() and (mask_c & (sources == 1)).any()):
            continue
        mean0 = z[mask_c & (sources == 0)].mean(0)
        mean1 = z[mask_c & (sources == 1)].mean(0)
        loss = loss + (1 - F.cosine_similarity(mean0, mean1, dim=0))
        count += 1
    if count == 0:
        return torch.tensor(0.0, device=z.device)
    return loss / count


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_ssl_mcr(cfg: SSLConfig):
    set_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_json(os.path.join(cfg.output_dir, "ssl_config.json"), asdict(cfg))

    # Data
    train_loader, test_loader, class_names = build_ssl_loaders(cfg)
    # Model
    backbone = build_model(cfg)
    # Remove classifier head; use global pooling output
    if hasattr(backbone, "reset_classifier"):
        backbone.reset_classifier(0)
    backbone.to(device)
    # Determine feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224, device=device)
        feat = backbone(dummy)
    if feat.ndim == 4:  # some models may return (B,C,H,W)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
    feat_dim = feat.shape[1]
    projector = ProjectionHead(feat_dim, cfg.proj_dim).to(device)

    params = list(backbone.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    for epoch in range(1, cfg.epochs + 1):
        backbone.train(); projector.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")
        epoch_loss = 0.0
        for v1, v2, y in pbar:
            v1, v2 = v1.to(device), v2.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                f1 = backbone(v1)
                f2 = backbone(v2)
                if f1.ndim == 4:
                    f1 = F.adaptive_avg_pool2d(f1, (1, 1)).flatten(1)
                    f2 = F.adaptive_avg_pool2d(f2, (1, 1)).flatten(1)
                z1 = projector(f1)
                z2 = projector(f2)
                loss_ssl = simclr_loss(z1, z2, temperature=cfg.temperature)
                loss = loss_ssl
                if cfg.mode == "mcr":
                    # Intra-source geodesic (approx) on concatenated f1+f2
                    feats_cat = torch.cat([f1, f2], dim=0)
                    loss_intra = compute_intra_geodesic_loss(
                        feats_cat.detach(),
                        k=cfg.mcr_intra_k,
                        mode=cfg.mcr_intra_mode,
                        sigma=cfg.mcr_lap_sigma,
                    )
                    # Cross-source alignment: mark first half as source0 second half as source0 (same source) => we need real cross-source pairs
                    # For simplicity sample half batch as source0 (web) half as source1 (phone) assumption -> placeholder.
                    # Real implementation should load true source flags. Here we simulate by splitting.
                    batch_sources = torch.cat([
                        torch.zeros(f1.size(0), device=device, dtype=torch.long),
                        torch.ones(f2.size(0), device=device, dtype=torch.long)
                    ])
                    batch_labels = torch.cat([y, y])
                    loss_cross = compute_cross_alignment_loss(feats_cat, batch_labels, batch_sources, k=cfg.mcr_cross_k)
                    loss = loss + cfg.mcr_lambda_intra * loss_intra + cfg.mcr_lambda_cross * loss_cross
                else:
                    loss_intra = torch.tensor(0.0, device=device)
                    loss_cross = torch.tensor(0.0, device=device)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * v1.size(0)
            pbar.set_postfix({"ssl": f"{loss_ssl.item():.3f}", "intra": f"{loss_intra.item():.3f}", "cross": f"{loss_cross.item():.3f}"})
        print(f"Epoch {epoch} mean loss: {epoch_loss / len(train_loader.dataset):.4f}")

    # Save backbone + projector
    torch.save({
        "backbone": backbone.state_dict(),
        "projector": projector.state_dict(),
        "config": asdict(cfg),
        "classes": class_names,
    }, os.path.join(cfg.output_dir, f"{cfg.mode}_model.pth"))

    # Geometry evaluation (optional)
    if cfg.eval_geometry:
        print("Computing geometry metrics (this may take a while)...")
        backbone.eval(); projector.eval()
        feats = []
        labels = []
        sources = []
        # Source 0: train/web, Source 1: test/phone
        with torch.no_grad():
            # collect web
            base_web = datasets.ImageFolder(cfg.train_dir, transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]))
            web_loader = DataLoader(base_web, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
            for x, y in tqdm(web_loader, desc="Feat-Web"):
                x = x.to(device)
                f = backbone(x)
                if f.ndim == 4:
                    f = F.adaptive_avg_pool2d(f, (1,1)).flatten(1)
                feats.append(f.cpu())
                labels.append(y)
                sources.append(torch.zeros_like(y))
            base_phone = datasets.ImageFolder(cfg.test_dir, transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]))
            phone_loader = DataLoader(base_phone, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers)
            for x, y in tqdm(phone_loader, desc="Feat-Phone"):
                x = x.to(device)
                f = backbone(x)
                if f.ndim == 4:
                    f = F.adaptive_avg_pool2d(f, (1,1)).flatten(1)
                feats.append(f.cpu())
                labels.append(y)
                sources.append(torch.ones_like(y))
        feats = torch.cat(feats, 0).numpy()
        labels = torch.cat(labels, 0).numpy()
        sources = torch.cat(sources, 0).numpy()
        # Reference space: use raw (feats) as both reference & embedded for baseline.
        report = evaluate_geometry(
            reference_features=feats,
            embedded_features=feats,
            labels=labels,
            source_flags=sources,
            k=cfg.geometry_k,
            max_geo_nodes=cfg.geometry_max_nodes,
        )
        save_json(os.path.join(cfg.output_dir, f"geometry_{cfg.mode}.json"), report.to_dict())
        print("Geometry metrics:", report.to_dict())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="SSL / MCR training")
    p.add_argument("--train_dir", type=str, default="./Dataset_1_Cleaned")
    p.add_argument("--test_dir", type=str, default="./Dataset_2_Cleaned")
    p.add_argument("--output_dir", type=str, default="./outputs_ssl")
    p.add_argument("--model_name", type=str, default="resnet18")
    p.add_argument("--pretrained", action="store_true", default=False)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--mode", type=str, choices=["ssl", "mcr"], default="ssl")
    p.add_argument("--mcr_intra_k", type=int, default=10)
    p.add_argument("--mcr_lambda_intra", type=float, default=1.0)
    p.add_argument("--mcr_lambda_cross", type=float, default=1.0)
    p.add_argument("--mcr_cross_k", type=int, default=5)
    p.add_argument("--mcr_intra_mode", type=str, choices=["distance", "laplacian"], default="distance")
    p.add_argument("--mcr_lap_sigma", type=float, default=0.0, help="Bandwidth for Laplacian mode; <=0 for auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--eval_geometry", action="store_true", default=False)
    p.add_argument("--geometry_k", type=int, default=10)
    p.add_argument("--geometry_max_nodes", type=int, default=2000)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = SSLConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        proj_dim=args.proj_dim,
        temperature=args.temperature,
        mode=args.mode,
        mcr_intra_k=args.mcr_intra_k,
    mcr_lambda_intra=args.mcr_lambda_intra,
        mcr_lambda_cross=args.mcr_lambda_cross,
        mcr_cross_k=args.mcr_cross_k,
    mcr_intra_mode=args.mcr_intra_mode,
    mcr_lap_sigma=args.mcr_lap_sigma,
        seed=args.seed,
        amp=args.amp,
        eval_geometry=args.eval_geometry,
        geometry_k=args.geometry_k,
        geometry_max_nodes=args.geometry_max_nodes,
    )
    train_ssl_mcr(cfg)


if __name__ == "__main__":
    main()
