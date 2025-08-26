#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
loocv.py — Leave-One-Animal-Out Cross-Validation (LOAO)

This script wraps around the model definition in `model.py` and performs
LOOCV across animals. It handles data loading (NumPy `.npz` arrays with
`features`, `labels`, and `assoc_ids`), training with balanced sample
weights, early stopping, model checkpointing, and evaluation with a
standard set of token-level metrics.

Author: Bekir Arda Yıldırım
"""
from __future__ import annotations

import argparse
import os
import random
import warnings
from typing import Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score
)

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import ModelConfig, build_model

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 43):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    """Slice (T, ...) → (Nseq, L, ...) dropping remainder to keep fixed L."""
    n_full = arr.shape[0] // seq_len
    if n_full == 0:
        raise ValueError("Array too short to form a single sequence.")
    arr = arr[: n_full * seq_len]
    return arr.reshape(n_full, seq_len, *arr.shape[1:])


def safe_metric(fn, *args, default=np.nan, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Core LOAO
# ---------------------------------------------------------------------------

def run_loao(npz_path: str,
             sequence_len: int,
             val_split: float,
             seed: int,
             outdir: str,
             epochs: int,
             batch_size: int) -> pd.DataFrame:
    # FS prep
    models_dir = os.path.join(outdir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load arrays
    data = np.load(npz_path, allow_pickle=True)
    features  = data['features']     # (T, 32, 32, 2)
    labels    = data['labels']       # (T,)
    assoc_ids = data['assoc_ids']    # (T,)

    # Group → fixed-length sequences
    unique_ids = np.unique(assoc_ids)
    seqs: Dict[object, np.ndarray] = {}
    labs: Dict[object, np.ndarray] = {}
    for aid in unique_ids:
        idx = np.where(assoc_ids == aid)[0]
        X_i = features[idx]
        y_i = labels[idx].reshape(-1, 1)
        seqs[aid] = make_sequences(X_i, sequence_len)
        labs[aid] = make_sequences(y_i, sequence_len)

    results = []

    for test_id in unique_ids:
        if isinstance(test_id, (np.integer, int, np.int64)):
            continue

        print(f"\n=== Test Animal: {int(test_id) if str(test_id).isdigit() else test_id} ===")

        # Train/Val pools
        train_val_ids = [aid for aid in unique_ids if aid != test_id]
        X_all = np.vstack([seqs[aid] for aid in train_val_ids])
        y_all = np.vstack([labs[aid] for aid in train_val_ids])
        X_test, y_test = seqs[test_id], labs[test_id]

        # Stratify by sequence-majority
        y_majority = (np.mean(y_all.reshape(y_all.shape[0], y_all.shape[1]), axis=1) > 0.5).astype(int)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
        train_idx, val_idx = next(sss.split(X_all, y_majority))
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val,   y_val   = X_all[val_idx],   y_all[val_idx]

        # Per-token balanced sample weights
        flat = y_train.ravel().astype(int)
        classes_present = np.unique(flat)
        if len(classes_present) == 1:
            cw = {int(classes_present[0]): 1.0}
        else:
            cw_vals = compute_class_weight('balanced', classes=classes_present, y=flat)
            cw = {int(k): v for k, v in zip(classes_present, cw_vals)}
        sw_train = np.array([[cw[int(l)] for l in seq.ravel()] for seq in y_train])

        # Build model
        K.clear_session()
        cfg = ModelConfig(sequence_len=sequence_len)
        model = build_model(cfg)

        # Train
        _ = model.fit(
            X_train, y_train,
            sample_weight=sw_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[
                EarlyStopping(monitor='val_f1_metric', mode='max', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_f1_metric', mode='max', factor=0.5, patience=3, min_lr=1e-6)
            ]
        )

        # Save fold model
        model_path = os.path.join(models_dir, f'model_test_animal_{int(test_id) if str(test_id).isdigit() else test_id}.keras')
        model.save(model_path)
        print(f"Model saved → {model_path}")

        # Evaluate (token-level)
        probs = model.predict(X_test, verbose=1)[:, :, 0].ravel()
        preds = (probs >= 0.5).astype(int)
        true  = y_test[:, :, 0].ravel().astype(int)

        tn, fp, fn, tp = confusion_matrix(true, preds, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        acc   = accuracy_score(true, preds)
        prec  = precision_score(true, preds, zero_division=0)
        rec   = recall_score(true, preds, zero_division=0)
        f1v   = f1_score(true, preds, zero_division=0)
        bacc  = balanced_accuracy_score(true, preds)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roc  = safe_metric(roc_auc_score, true, probs)
            ap   = safe_metric(average_precision_score, true, probs)

        fold_metrics = {
            'test_animal': int(test_id) if str(test_id).isdigit() else str(test_id),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'specificity': specificity,
            'f1_score': f1v,
            'balanced_accuracy': bacc,
            'roc_auc': roc,
            'avg_precision': ap
        }

        # Pretty print
        print("Metrics (3 dp):")
        for k, v in fold_metrics.items():
            if k in {'test_animal', 'tp', 'fp', 'tn', 'fn'}:
                print(f"  {k}: {v}")
            else:
                try:
                    print(f"  {k}: {v:.3f}")
                except Exception:
                    print(f"  {k}: {v}")

        results.append(fold_metrics)

    # Save per-fold and summary
    df = pd.DataFrame(results)
    per_fold_csv = os.path.join(outdir, 'loocv_per_fold_metrics.csv')
    df.to_csv(per_fold_csv, index=False)

    print("\n=== Per-fold Results (3 dp) ===")
    with pd.option_context('display.float_format', '{:0.3f}'.format):
        print(df)

    print("\n=== Overall Results (mean ± SEM) ===")
    summary = df.drop(columns=['test_animal', 'tp', 'fp', 'tn', 'fn']).agg(['mean', 'sem'])
    with pd.option_context('display.float_format', '{:0.1f}'.format):
        print(summary)

    summary_csv = os.path.join(outdir, 'loocv_summary_mean_sem.csv')
    summary.to_csv(summary_csv)

    print(f"\nSaved per‑fold → {per_fold_csv}")
    print(f"Saved summary → {summary_csv}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='LOAO training & eval for CWT‑Hybrid CNN')
    # Paths
    p.add_argument('--npz', default='/content/drive/MyDrive/eeg_arrays/Auto SWD/Arrays/Dataset.npz',
                   help='Path to .npz with features, labels, assoc_ids')
    p.add_argument('--outdir', default='/content/drive/MyDrive/saved_models',
                   help='Output directory for models & metrics')
    # Training setup
    p.add_argument('--sequence-len', type=int, default=100)
    p.add_argument('--val-split', type=float, default=0.20)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--seed', type=int, default=43)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_seeds(args.seed)

    _ = run_loao(
        npz_path=args.npz,
        sequence_len=args.sequence_len,
        val_split=args.val_split,
        seed=args.seed,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == '__main__':
    main()
