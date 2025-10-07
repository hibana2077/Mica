#!/usr/bin/env python
"""Linear probing script for pretrained SSL encoder.

Loads encoder weights (frozen) from ssl_best.pth and trains a linear classifier
on a specified fraction of labels from one domain, evaluating same-domain and
cross-domain performance. Implements the experimental protocol sections for
label efficiency curves.

Example:
  python src/linear_probe.py \
    --encoder_ckpt output/SSL_resnet50/ssl_best.pth \
    --train_domain dataset/Dataset_1_Cleaned \
    --test_domain dataset/Dataset_2_Cleaned \
    --label_fraction 0.05 --epochs 50 --output_dir output/LP_run
"""
from __future__ import annotations
import argparse, json, os, random
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm

from utils import set_seed, get_device, save_json
from utils.data import accuracy


@dataclass
class LPConfig:
    encoder_ckpt: str
    train_domain: str
    test_domain: str
    output_dir: str = "output/LP"
    label_fraction: float = 1.0  # 0 < f <=1 ; if <1 stratified subset
    per_class_limit: int = 0      # optional absolute cap per class (overrides fraction if >0)
    batch_size: int = 128
    workers: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    amp: bool = True
    eval_knn: bool = True
    knn_k: int = 20


def stratified_subset(ds: datasets.ImageFolder, fraction: float, per_class_limit: int, seed: int) -> Subset:
    if fraction >= 0.999 and per_class_limit <= 0:
        return Subset(ds, list(range(len(ds))))
    random.seed(seed)
    by_class = {}
    for idx, (_p, cls) in enumerate(ds.samples):
        by_class.setdefault(cls, []).append(idx)
    chosen = []
    for cls, idxs in by_class.items():
        random.shuffle(idxs)
        if per_class_limit > 0:
            take = min(per_class_limit, len(idxs))
        else:
            take = max(1, int(len(idxs) * fraction))
        chosen.extend(idxs[:take])
    chosen.sort()
    return Subset(ds, chosen)


def build_encoder(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    from timm import create_model
    encoder = create_model(cfg["model_name"], pretrained=False, num_classes=0, global_pool="avg")
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, cfg


def prepare_loaders(cfg: LPConfig, encoder, seed: int):
    """Create training & validation DataLoaders.

    Uses timm's resolve_data_config on the already constructed encoder so that
    the correct default_cfg (input size, normalization stats, interpolation, etc.)
    is applied. Previously this passed model=None which triggered an assertion
    in timm (needs model / args / pretrained_cfg). Passing the actual encoder
    avoids the crash on systems where no explicit data args are provided.
    """
    try:
        data_cfg = resolve_data_config({}, model=encoder)
        tfm_train = create_transform(**data_cfg, is_training=True)
        tfm_eval = create_transform(**data_cfg, is_training=False)
    except Exception as e:
        # Fallback: basic ImageNet-style transforms if something unexpected occurs.
        from torchvision import transforms
        tfm_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        tfm_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    full_train = datasets.ImageFolder(cfg.train_domain, transform=tfm_train)
    test = datasets.ImageFolder(cfg.test_domain, transform=tfm_eval)
    subset = stratified_subset(full_train, cfg.label_fraction, cfg.per_class_limit, seed)
    # If the supervision subset is smaller than batch_size, setting drop_last=True would
    # drop the only (incomplete) batch, yielding zero iterations and later a div-by-zero.
    dynamic_drop_last = len(subset) >= cfg.batch_size
    train_loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=dynamic_drop_last,
    )
    val_loader = DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    return train_loader, val_loader, full_train.classes


def evaluate_linear(encoder, classifier, loader, device, amp=True):
    encoder.eval(); classifier.eval()
    total_loss = 0.0
    total_top1 = 0.0
    n = 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        pbar = loader
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                feats = encoder(images)
                logits = classifier(feats)
                loss = ce(logits, targets)
            top1, = accuracy(logits, targets, topk=(1,))
            bs = images.size(0)
            n += bs
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
    return {"loss": total_loss / n, "acc": total_top1 / n}


def knn_eval(encoder, train_loader, test_loader, device, k=20, amp=True):
    # Extract train features & labels
    encoder.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                f = encoder(x)
            feats.append(F.normalize(f, dim=1).cpu())
            labels.append(y)
    feats = torch.cat(feats, 0)
    labels = torch.cat(labels, 0)
    # Test
    correct = 0
    n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                f = encoder(x)
            f = F.normalize(f, dim=1)
            sims = f @ feats.T  # (B, Ntrain)
            topk = sims.topk(k, dim=1).indices  # (B, k)
            pred = torch.mode(labels[topk], dim=1).values
            correct += (pred.cpu() == y).sum().item()
            n += y.size(0)
    return correct * 100.0 / n


def train_linear(cfg: LPConfig):
    device = get_device(); set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_json(os.path.join(cfg.output_dir, "lp_config.json"), asdict(cfg))
    encoder, enc_cfg = build_encoder(cfg.encoder_ckpt)
    encoder.to(device)
    feat_dim = getattr(encoder, "num_features", 2048)
    # Build loaders (train domain subset, test domain full)
    train_loader, test_loader, classes = prepare_loaders(cfg, encoder, cfg.seed)
    classifier = nn.Linear(feat_dim, len(classes)).to(device)
    opt = torch.optim.AdamW(classifier.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Use new torch.amp API (deprecation warning otherwise)
    scaler = torch.amp.GradScaler(device_type="cuda", enabled=cfg.amp and device.type == "cuda")
    ce = nn.CrossEntropyLoss()
    best_acc = -1
    log_path = os.path.join(cfg.output_dir, "lp_log.jsonl")
    with open(log_path, "w"):
        pass
    for epoch in range(1, cfg.epochs + 1):
        classifier.train()
        run_loss = 0.0; run_acc = 0.0; seen = 0
        pbar = tqdm(train_loader, desc=f"LP Epoch {epoch}/{cfg.epochs}", leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(device); targets = targets.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                feats = encoder(imgs)
                logits = classifier(feats)
                loss = ce(logits, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            top1, = accuracy(logits, targets, topk=(1,))
            bs = imgs.size(0)
            run_loss += loss.item() * bs
            run_acc += top1 * bs
            seen += bs
            pbar.set_postfix({"loss": f"{run_loss/seen:.3f}", "acc": f"{run_acc/seen:.2f}"})
        if seen == 0:
            # This should not happen anymore due to dynamic_drop_last logic, but guard just in case.
            # Provide actionable feedback about likely cause.
            raise RuntimeError(
                "No training batches were processed (seen=0). Likely the supervision subset is smaller "
                "than batch_size and was dropped. Reduce --batch_size or increase label_fraction/per_class_limit."
            )
        train_loss = run_loss / seen; train_acc = run_acc / seen
        test_metrics = evaluate_linear(encoder, classifier, test_loader, device, amp=scaler.is_enabled())
        log_entry = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, **{f"test_{k}": v for k,v in test_metrics.items()}}
        with open(log_path, "a") as f: f.write(json.dumps(log_entry) + "\n")
        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            torch.save({"classifier": classifier.state_dict(), "epoch": epoch, "best_acc": best_acc, "cfg": asdict(cfg), "encoder_cfg": enc_cfg}, os.path.join(cfg.output_dir, "lp_best.pth"))
    # Optional k-NN evaluation using same supervised subset as memory bank
    if cfg.eval_knn:
        knn_acc = knn_eval(encoder, train_loader, test_loader, device, k=cfg.knn_k, amp=scaler.is_enabled())
        save_json(os.path.join(cfg.output_dir, "knn_metrics.json"), {"knn_acc": knn_acc})


def parse_args():
    p = argparse.ArgumentParser(description="Linear Probe on SSL Encoder")
    from dataclasses import MISSING
    # Resolve (possibly postponed) type annotations so that bools are detected correctly
    from typing import get_type_hints
    type_hints = get_type_hints(LPConfig)
    for field in LPConfig.__dataclass_fields__.values():
        name = f"--{field.name}"
        # Determine the (resolved) python type of the field (handles __future__ annotations)
        resolved_type = type_hints.get(field.name, field.type)
        # Handle booleans with --flag / --no_field semantics
        if resolved_type is bool:
            default_val = getattr(LPConfig, field.name)
            if default_val:  # default True -> create a --no_flag to disable
                p.add_argument(f"--no_{field.name}", action="store_false", dest=field.name,
                               help=f"Disable {field.name} (default: enabled)")
            else:            # default False -> create a --flag to enable
                p.add_argument(name, action="store_true",
                               help=f"Enable {field.name} (default: disabled)")
            continue

        # Non-boolean fields
        if field.default is MISSING and field.default_factory is MISSING:  # required argument
            # Assume string type for required path / directory style args
            p.add_argument(name, required=True, type=str,
                           help=f"(Required) Value for {field.name}")
        else:
            default_val = field.default
            # Infer a callable type for argparse based on the default's Python type
            inferred_type = type(default_val)
            # Edge case: if default is None, fall back to string
            if default_val is None:
                inferred_type = str
            p.add_argument(name, type=inferred_type, default=default_val,
                           help=f"Default: {default_val}")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = LPConfig(**vars(args))
    train_linear(cfg)


if __name__ == "__main__":
    main()
