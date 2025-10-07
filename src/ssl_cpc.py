#!/usr/bin/env python
"""Contrastive Predictive Coding / InfoNCE style SSL with optional domain alignment.

Lightweight implementation to serve as an ablation against VICReg (see ssl_vicreg.py).
Focus: provide comparable interface & logging so that experiments can toggle between methods.

Key differences:
 - Uses standard InfoNCE with cosine similarity on projected features.
 - Temperature parameter tau.
 - Optional mean+cov domain alignment (same helper as VICReg version) for consistency.

Outputs:
  output_dir/ssl_cpc_best.pth  (encoder + projector)
  output_dir/ssl_cpc_log.jsonl (per-epoch metrics)
  output_dir/feature_stats.json (diagnostics)
"""
from __future__ import annotations
import argparse, json, os, math
from dataclasses import dataclass, asdict
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm
from utils import set_seed, get_device, save_json
from ssl_vicreg import build_encoder_and_projector, build_ssl_dataloaders, domain_alignment, cosine_warmup

@dataclass
class CPCConfig:
    dataset1: str = "dataset/Dataset_1_Cleaned"
    dataset2: str = "dataset/Dataset_2_Cleaned"
    output_dir: str = "output/SSL_CPC"
    model_name: str = "resnet50.a1_in1k"
    pretrained: bool = True
    batch_size: int = 256
    workers: int = 4
    epochs: int = 200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    proj_dim: int = 2048
    proj_hidden: int = 8192
    tau: float = 0.2               # temperature for InfoNCE
    lambda_align: float = 1.0
    seed: int = 42
    log_every: int = 50
    save_every: int = 25
    no_amp: bool = False


def two_views(x: torch.Tensor):
    # current pipeline already applies random transforms; duplicate with clone
    return x, x.clone()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    logits = z1 @ z2.t() / tau
    targets = torch.arange(B, device=z1.device)
    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.t(), targets)
    loss = (loss_12 + loss_21) / 2
    # compute alignment stats for monitoring (cosine sim of positives)
    pos_sim = (z1 * z2).sum(1).mean().detach()
    return loss, pos_sim


def train(cfg: CPCConfig):
    device = get_device(); set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_json(os.path.join(cfg.output_dir, "ssl_cpc_config.json"), asdict(cfg))
    encoder, projector, feat_dim = build_encoder_and_projector(cfg)  # reuse VICReg builder
    loader1, loader2, _ = build_ssl_dataloaders(cfg, encoder)
    encoder.to(device); projector.to(device)
    opt = torch.optim.AdamW(list(encoder.parameters())+list(projector.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=not cfg.no_amp and device.type=="cuda")
    best_loss = float('inf'); global_step=0
    total_steps = cfg.epochs * min(len(loader1), len(loader2))
    log_path = os.path.join(cfg.output_dir, "ssl_cpc_log.jsonl")
    with open(log_path,'w'): pass
    for epoch in range(1, cfg.epochs+1):
        encoder.train(); projector.train()
        it1 = iter(loader1); it2 = iter(loader2)
        steps = min(len(loader1), len(loader2))
        pbar = tqdm(range(steps), desc=f"CPC Epoch {epoch}/{cfg.epochs}", leave=False)
        run = {k:0.0 for k in ['loss','align','align_mean','align_cov','pos_sim']}
        seen=0
        for _ in pbar:
            try: s,_ = next(it1)
            except StopIteration: it1 = iter(loader1); s,_ = next(it1)
            try: t,_ = next(it2)
            except StopIteration: it2 = iter(loader2); t,_ = next(it2)
            s = s.to(device); t=t.to(device)
            x = torch.cat([s,t],0)
            v1,v2 = two_views(x)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                z1 = projector(encoder(v1)); z2 = projector(encoder(v2))
                loss_core, pos_sim = info_nce(z1, z2, cfg.tau)
                z_avg = 0.5*(z1+z2)
                z_s = z_avg[:s.size(0)]; z_t = z_avg[s.size(0):]
                align_raw, align_mean, align_cov = domain_alignment(z_s, z_t)
                loss = loss_core + cfg.lambda_align * align_raw
            if scaler.is_enabled():
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            global_step +=1
            lr_now = cosine_warmup(global_step, total_steps, cfg.warmup_epochs*steps, cfg.lr)
            for pg in opt.param_groups: pg['lr']=lr_now
            bs = x.size(0); seen += bs
            run['loss'] += loss.item()*bs; run['align'] += (cfg.lambda_align*align_raw).item()*bs
            run['align_mean'] += align_mean.item()*bs; run['align_cov'] += align_cov.item()*bs
            run['pos_sim'] += pos_sim.item()*bs
            if global_step % cfg.log_every ==0:
                pbar.set_postfix({"loss": f"{run['loss']/seen:.3f}", 'pos': f"{run['pos_sim']/seen:.3f}", 'align': f"{run['align']/seen:.3f}"})
        epoch_metrics = {k: v/seen for k,v in run.items()}; epoch_metrics['epoch']=epoch; epoch_metrics['lr']=lr_now
        with open(log_path,'a') as f: f.write(json.dumps(epoch_metrics)+"\n")
        if epoch_metrics['loss'] < best_loss:
            best_loss = epoch_metrics['loss']
            torch.save({'encoder': encoder.state_dict(), 'projector': projector.state_dict(), 'epoch': epoch, 'cfg': asdict(cfg), 'feat_dim': feat_dim, 'best_loss': best_loss}, os.path.join(cfg.output_dir,'ssl_cpc_best.pth'))
        if epoch % cfg.save_every==0:
            torch.save({'encoder': encoder.state_dict(), 'projector': projector.state_dict(), 'epoch': epoch, 'cfg': asdict(cfg), 'feat_dim': feat_dim}, os.path.join(cfg.output_dir,f'ssl_cpc_epoch{epoch}.pth'))
    # Diagnostics
    encoder.eval(); projector.eval()
    with torch.no_grad():
        ds1 = datasets.ImageFolder(cfg.dataset1, transform=create_transform(**resolve_data_config({}, model=encoder), is_training=False))
        loader = DataLoader(ds1, batch_size=256, shuffle=False)
        feats=[]
        for imgs,_ in loader:
            imgs=imgs.to(device)
            f=projector(encoder(imgs)); feats.append(f.cpu())
        feats = torch.cat(feats,0)
        std = feats.std(0); cov = torch.cov(feats.T)
        offdiag = (cov - torch.diag(torch.diag(cov))).pow(2).mean().item()
        save_json(os.path.join(cfg.output_dir,'feature_stats.json'), {'mean_std': std.mean().item(),'offdiag_energy': offdiag, 'n_samples': feats.size(0)})


def parse_args():
    p = argparse.ArgumentParser(description='CPC/InfoNCE SSL with optional domain alignment')
    for field in CPCConfig.__dataclass_fields__.values():
        name = f"--{field.name}"; default=field.default
        if isinstance(default,bool):
            if default:
                p.add_argument(f"--no_{field.name}", dest=field.name, action='store_false', help=f"Disable {field.name}")
            else:
                p.add_argument(name, action='store_true', help=f"Enable {field.name}")
        else:
            p.add_argument(name, type=type(default), default=default)
    return p.parse_args()


def main():
    args = parse_args(); cfg = CPCConfig(**vars(args)); train(cfg)

if __name__ == '__main__':
    main()
