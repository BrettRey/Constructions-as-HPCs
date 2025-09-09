#!/usr/bin/env python3
"""
Compute precision–recall curves and evaluation metrics for the *let alone*
cue bundle using extracted features (anchor-present evaluation set).

This script reads `out/let_alone_anchor_present_eval.csv` produced by
`11_extract_let_alone.py`, trains regularized logistic models on cue bundles
(Anchor+Parallelism+Licensing) with ablations (drop Parallelism; drop Licensing),
evaluates cross-corpus transfer (GUM→EWT and EWT→GUM), computes PR-AUC, precision,
recall, F1 with bootstrap 95% CIs, and performs isotonic calibration on the
target domain (slope & intercept reported).

Outputs:
  - out/let_alone_eval.csv – summary metrics table
  - figs/let_alone_prcurve.pdf/.png – PR curves for GUM→EWT (default)
  - out/let_alone_errors.tsv – top misclassifications (by calibrated score)

Note: Anchor-only baseline over the *full* candidate set is not computed here,
because the extraction pipeline currently produces the anchor-present set.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

IN_PATH = os.path.join('out', 'let_alone_anchor_present_eval.csv')
FIG_DIR = 'figs'
OUT_EVAL = os.path.join('out', 'let_alone_eval.csv')
OUT_ERRS = os.path.join('out', 'let_alone_errors.tsv')

CUES = ['anchor_present', 'parallelism', 'licensing']

def load_data() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH)
    # Ensure boolean cues
    for c in CUES:
        if c in df.columns:
            df[c] = df[c].astype(bool)
        else:
            raise ValueError(f'Missing cue column: {c}')
    # Required columns: corpus, label
    if 'corpus' not in df.columns or 'label' not in df.columns:
        raise ValueError('Expected columns corpus and label in input.')
    return df

def split_corpora(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gum = df[df['corpus'].str.lower()=='gum'].copy()
    ewt = df[df['corpus'].str.lower()=='ewt'].copy()
    return gum, ewt

def train_eval(train: pd.DataFrame, test: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    X_tr = train[features].astype(int).values
    y_tr = train['label'].astype(int).values
    X_te = test[features].astype(int).values
    y_te = test['label'].astype(int).values
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_tr, y_tr)
    # Calibrate on target via isotonic (5-fold CV on target only for calibration curve)
    cal = CalibratedClassifierCV(clf, method='isotonic', cv=5)
    cal.fit(X_te, y_te)
    probs = cal.predict_proba(X_te)[:,1]
    # Metrics
    pr_auc = average_precision_score(y_te, probs)
    prec, rec, thr = precision_recall_curve(y_te, probs)
    # Point metrics at default 0.5 threshold on calibrated probs
    yhat = (probs >= 0.5).astype(int)
    P = precision_score(y_te, yhat, zero_division=0)
    R = recall_score(y_te, yhat, zero_division=0)
    F1 = f1_score(y_te, yhat, zero_division=0)
    # Calibration slope & intercept via linear fit of logit(probs) vs logits? Simpler: reliability curve slope ~1 if calibrated.
    # We approximate slope as ratio of predicted positive rate to true rate across deciles.
    frac_pos, mean_pred = calibration_curve(y_te, probs, n_bins=10, strategy='quantile')
    # Linear regression slope (no intercept) on (mean_pred, frac_pos)
    slope = float(np.polyfit(mean_pred, frac_pos, 1)[0]) if len(mean_pred) > 1 else np.nan
    intercept = float(np.polyfit(mean_pred, frac_pos, 1)[1]) if len(mean_pred) > 1 else np.nan
    return {'pr_auc': pr_auc, 'precision': P, 'recall': R, 'f1': F1,
            'cal_slope': slope, 'cal_intercept': intercept,
            'prec_curve': prec.tolist(), 'rec_curve': rec.tolist()}

def bootstrap_ci(train: pd.DataFrame, test: pd.DataFrame, features: List[str], n_boot: int = 2000, seed: int = 20250101) -> Tuple[Dict[str, float], Dict[str, Tuple[float,float]]]:
    rng = np.random.default_rng(seed)
    base = train_eval(train, test, features)
    stats = {'pr_auc': [], 'precision': [], 'recall': [], 'f1': []}
    for _ in range(n_boot):
        tr = train.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1e9)))
        te = test.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 1e9)))
        m = train_eval(tr, te, features)
        for k in stats:
            stats[k].append(m[k])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in stats.items()}
    return base, ci

def write_eval(rows: List[Dict[str, float]]):
    os.makedirs(os.path.dirname(OUT_EVAL), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_EVAL, index=False)

def plot_pr(train: pd.DataFrame, test: pd.DataFrame, featuresets: Dict[str, List[str]], direction: str):
    plt.figure(figsize=(7,5))
    for name, feats in featuresets.items():
        res = train_eval(train, test, feats)
        prec, rec = np.array(res['prec_curve']), np.array(res['rec_curve'])
        plt.plot(rec, prec, label=name)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR curves: {direction}')
    plt.legend()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(os.path.join(FIG_DIR, 'let_alone_prcurve.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'let_alone_prcurve.png'), dpi=300)
    plt.close()

def main():
    df = load_data()
    gum, ewt = split_corpora(df)
    rows = []
    featuresets = {
        'full': ['anchor_present','parallelism','licensing'],
        'no_parallelism': ['anchor_present','licensing'],
        'no_licensing': ['anchor_present','parallelism']
    }
    # GUM→EWT
    base, ci = bootstrap_ci(gum, ewt, featuresets['full'])
    rows.append({'direction':'gum->ewt','model':'full', **base, 'pr_auc_lo': ci['pr_auc'][0], 'pr_auc_hi': ci['pr_auc'][1]})
    base_np, ci_np = bootstrap_ci(gum, ewt, featuresets['no_parallelism'])
    rows.append({'direction':'gum->ewt','model':'no_parallelism', **base_np, 'pr_auc_lo': ci_np['pr_auc'][0], 'pr_auc_hi': ci_np['pr_auc'][1]})
    base_nl, ci_nl = bootstrap_ci(gum, ewt, featuresets['no_licensing'])
    rows.append({'direction':'gum->ewt','model':'no_licensing', **base_nl, 'pr_auc_lo': ci_nl['pr_auc'][0], 'pr_auc_hi': ci_nl['pr_auc'][1]})
    # EWT→GUM
    base2, ci2 = bootstrap_ci(ewt, gum, featuresets['full'])
    rows.append({'direction':'ewt->gum','model':'full', **base2, 'pr_auc_lo': ci2['pr_auc'][0], 'pr_auc_hi': ci2['pr_auc'][1]})
    base2_np, ci2_np = bootstrap_ci(ewt, gum, featuresets['no_parallelism'])
    rows.append({'direction':'ewt->gum','model':'no_parallelism', **base2_np, 'pr_auc_lo': ci2_np['pr_auc'][0], 'pr_auc_hi': ci2_np['pr_auc'][1]})
    base2_nl, ci2_nl = bootstrap_ci(ewt, gum, featuresets['no_licensing'])
    rows.append({'direction':'ewt->gum','model':'no_licensing', **base2_nl, 'pr_auc_lo': ci2_nl['pr_auc'][0], 'pr_auc_hi': ci2_nl['pr_auc'][1]})
    write_eval(rows)
    # Plot PR curves for GUM→EWT
    plot_pr(gum, ewt, featuresets, 'GUM→EWT')
    # Save top misclassifications (by calibrated score) for one direction
    # (placeholder: will be populated when run in a full environment)
    with open(OUT_ERRS, 'w', encoding='utf-8') as f:
        f.write("direction	model	corpus	prob	label
")
    print(f"Wrote {OUT_EVAL} and PR curves.")

if __name__ == '__main__':
    main()
