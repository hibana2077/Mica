#!/usr/bin/env python
"""Few-shot evaluation using frozen backbone features.

Loads a trained SSL/MCR backbone checkpoint saved by ssl_mcr.py, extracts
features for (web=train_dir) and (phone=test_dir) datasets, then performs
few-shot classification on the phone set with 1..K shots per class sampled
from the phone domain itself (or optionally from web) using simple k-NN or
linear probing. Reports accuracy for each shot count.

Usage:
  python fewshot_eval.py --ckpt path/to/mcr_model.pth --train_dir dataset/Dataset_1_Cleaned \
      --test_dir dataset/Dataset_2_Cleaned --shots 1 3 5 --method knn --repeats 5

Outputs JSON file in same folder as checkpoint with suffix _fewshot.json
"""
from __future__ import annotations
import argparse, os, json, random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import build_model, get_device


def extract_features(backbone, loader, device):
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = backbone(x)
            if f.ndim == 4:
                f = F.adaptive_avg_pool2d(f, (1,1)).flatten(1)
            feats.append(f.cpu())
            labels.append(y)
    return torch.cat(feats,0), torch.cat(labels,0)


def fewshot_knn(train_feats, train_labels, test_feats, test_labels, k=1):
    # simple L2 distance k-NN
    with torch.no_grad():
        d2 = torch.cdist(test_feats, train_feats, p=2)
        nn_idx = d2.topk(k, largest=False).indices  # (Nt,k)
        preds = []
        for i in range(nn_idx.size(0)):
            labs = train_labels[nn_idx[i]]
            # majority vote
            vals, counts = labs.unique(return_counts=True)
            pred = vals[counts.argmax()].item()
            preds.append(pred)
        preds = torch.tensor(preds)
        acc = (preds == test_labels).float().mean().item()
    return acc


def fewshot_linear(train_feats, train_labels, test_feats, test_labels, epochs=100):
    # simple single hidden layer linear classifier
    num_classes = int(train_labels.max().item()+1)
    clf = torch.nn.Linear(train_feats.size(1), num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = clf(train_feats)
        loss = F.cross_entropy(out, train_labels)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = clf(test_feats).argmax(1)
        acc = (preds == test_labels).float().mean().item()
    return acc


def sample_fewshot(features, labels, shots):
    # returns subset features/labels with exactly `shots` per class
    classes = labels.unique().tolist()
    idxs = []
    for c in classes:
        all_idx = (labels == c).nonzero(as_tuple=True)[0].tolist()
        chosen = random.sample(all_idx, min(shots, len(all_idx)))
        idxs.extend(chosen)
    sel = torch.tensor(idxs)
    return features[sel], labels[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--train_dir', type=str, required=True)
    ap.add_argument('--test_dir', type=str, required=True)
    ap.add_argument('--shots', type=int, nargs='+', default=[1,3,5])
    ap.add_argument('--method', type=str, choices=['knn','linear'], default='knn')
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    model_name = cfg.get('model_name', 'resnet18')
    backbone = build_model(type('tmp', (), {'model_name':model_name, 'pretrained':False, 'num_classes':0}))
    if hasattr(backbone,'reset_classifier'): backbone.reset_classifier(0)
    backbone.load_state_dict(ckpt['backbone'])
    backbone.to(device)

    tfm = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_set = datasets.ImageFolder(args.train_dir, transform=tfm)
    test_set = datasets.ImageFolder(args.test_dir, transform=tfm)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    train_feats, train_labels = extract_features(backbone, train_loader, device)
    test_feats, test_labels = extract_features(backbone, test_loader, device)

    results = { 'method': args.method, 'shots': {}, 'repeats': args.repeats }

    for shots in args.shots:
        accs = []
        for _ in range(args.repeats):
            fs, ls = sample_fewshot(train_feats, train_labels, shots)
            if args.method == 'knn':
                acc = fewshot_knn(fs, ls, test_feats, test_labels, k=1)
            else:
                acc = fewshot_linear(fs, ls, test_feats, test_labels)
            accs.append(acc)
        results['shots'][shots] = {'mean_acc': float(np.mean(accs)), 'std_acc': float(np.std(accs))}
        print(f'Shots {shots}: acc mean={np.mean(accs):.4f} std={np.std(accs):.4f}')

    out_path = args.ckpt.replace('.pth','_fewshot.json')
    with open(out_path,'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2)
    print('Saved few-shot results to', out_path)

if __name__ == '__main__':
    main()
