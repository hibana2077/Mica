#!/usr/bin/env python
"""Embedding visualization (t-SNE / UMAP) dual-colored by class & domain.

Generates two plots:
  - Colored by class label
  - Colored by domain (source vs target)

Also computes simple domain gap metrics (MMD over raw features, cosine center distance)
that can be later correlated with cross-domain accuracy.

Requires: matplotlib, seaborn, umap-learn (optional), scikit-learn
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch, torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.data import resolve_data_config, create_transform
from sklearn.manifold import TSNE
try:
    import umap
except Exception:
    umap = None
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_device


def load_encoder(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['cfg']; from timm import create_model
    encoder = create_model(cfg['model_name'], pretrained=False, num_classes=0, global_pool='avg')
    encoder.load_state_dict(ckpt['encoder'], strict=True)
    encoder.eval()
    return encoder


def gather_features(encoder, root: str, batch: int, device):
    data_cfg = resolve_data_config({}, model=encoder)
    tfm = create_transform(**data_cfg, is_training=False)
    ds = datasets.ImageFolder(root, transform=tfm)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)
    feats=[]; labels=[]
    with torch.no_grad():
        for imgs, y in loader:
            imgs=imgs.to(device)
            f = encoder(imgs).cpu()
            feats.append(f); labels.append(y)
    feats = torch.cat(feats,0); labels = torch.cat(labels,0)
    return feats, labels, ds.classes


def mmd_rbf(x, y, gamma=1.0):
    def k(a,b):
        return torch.exp(-gamma*((a.unsqueeze(1)-b.unsqueeze(0))**2).sum(-1))
    kxx = k(x,x).mean(); kyy = k(y,y).mean(); kxy = k(x,y).mean()
    return (kxx + kyy - 2*kxy).item()


def main():
    ap = argparse.ArgumentParser(description='Embedding visualization & domain metrics')
    ap.add_argument('--encoder_ckpt', required=True)
    ap.add_argument('--dataset1', required=True)
    ap.add_argument('--dataset2', required=True)
    ap.add_argument('--output_dir', default='output/viz')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--max_per_domain', type=int, default=2000)
    ap.add_argument('--use_umap', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    encoder = load_encoder(args.encoder_ckpt).to(device)
    f1, y1, classes = gather_features(encoder, args.dataset1, args.batch, device)
    f2, y2, _ = gather_features(encoder, args.dataset2, args.batch, device)
    # Subsample if necessary
    if f1.size(0) > args.max_per_domain:
        idx = torch.randperm(f1.size(0))[:args.max_per_domain]; f1 = f1[idx]; y1=y1[idx]
    if f2.size(0) > args.max_per_domain:
        idx = torch.randperm(f2.size(0))[:args.max_per_domain]; f2 = f2[idx]; y2=y2[idx]
    feats = torch.cat([f1,f2],0)
    labels = torch.cat([y1,y2],0)
    domains = torch.cat([torch.zeros(f1.size(0),dtype=torch.long), torch.ones(f2.size(0),dtype=torch.long)])
    # Compute domain metrics
    mmd = mmd_rbf(F.normalize(f1,dim=1), F.normalize(f2,dim=1), gamma=1.0/feats.size(1))
    center_gap = F.pairwise_distance(f1.mean(0).unsqueeze(0), f2.mean(0).unsqueeze(0)).item()
    metrics = {'mmd': mmd, 'center_gap': center_gap, 'n1': f1.size(0), 'n2': f2.size(0)}
    with open(os.path.join(args.output_dir,'domain_metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
    # Dim reduction
    feats_np = feats.numpy()
    if args.use_umap and umap is not None:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine', random_state=42)
        emb = reducer.fit_transform(feats_np)
        method='umap'
    else:
        emb = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca', random_state=42).fit_transform(feats_np)
        method='tsne'
    # Plot by class
    plt.figure(figsize=(8,6))
    palette = sns.color_palette('tab20', n_colors=len(classes))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=[classes[i] for i in labels], legend=False, s=8, palette=palette)
    plt.title(f'Embedding ({method}) colored by class')
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir,f'emb_{method}_class.png'), dpi=200); plt.close()
    # Plot by domain
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=domains.numpy(), palette=['#1f77b4','#ff7f0e'], s=10)
    plt.title(f'Embedding ({method}) colored by domain')
    plt.legend(title='Domain', labels=['Dataset1','Dataset2'])
    plt.tight_layout(); plt.savefig(os.path.join(args.output_dir,f'emb_{method}_domain.png'), dpi=200); plt.close()
    print('Saved embeddings & metrics to', args.output_dir)

if __name__ == '__main__':
    main()
