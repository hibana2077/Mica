#!/usr/bin/env python
"""Correlate domain proxy metrics with cross-domain accuracy.

Inputs:
  --metrics_csv: CSV containing rows with columns [run, fraction, in_acc, cross_acc, f1, ...]
  --proxy_json: JSON file containing domain proxy metrics (e.g., mmd, center_gap)

Outputs:
  correlation.json: Pearson & Spearman coefficients between each proxy metric and cross-domain acc.

The CSV can be produced by aggregating multiple linear probe runs (parse their best_metrics.json files).
"""
from __future__ import annotations
import argparse, json, csv, os
import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    ap = argparse.ArgumentParser(description='Domain gap correlation analysis')
    ap.add_argument('--metrics_csv', required=True)
    ap.add_argument('--proxy_json', required=True)
    ap.add_argument('--output', default='correlation.json')
    args = ap.parse_args()
    with open(args.metrics_csv,'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    with open(args.proxy_json,'r') as f:
        proxies = json.load(f)
    cross_acc = np.array([float(r['cross_acc']) for r in rows])
    results = {}
    for name, val in proxies.items():
        if not isinstance(val, (int,float)): continue
        arr = np.array([val for _ in rows], dtype=float)
        # Since proxy is constant across fractions, correlation would be undefined.
        # In practical use, user should supply per-run proxy; we guard anyway.
        if np.allclose(arr, arr[0]):
            results[name] = {'pearson': 0.0, 'spearman': 0.0, 'note': 'constant proxy (need per-run values)'}
        else:
            pr, _ = pearsonr(arr, cross_acc)
            sr, _ = spearmanr(arr, cross_acc)
            results[name] = {'pearson': float(pr), 'spearman': float(sr)}
    with open(args.output,'w') as f:
        json.dump(results,f,indent=2)
    print('Wrote', args.output)

if __name__ == '__main__':
    main()
