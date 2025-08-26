#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDF/Excel → NPZ Data Builder (Binary SWD vs. Background)
=======================================================

• Reads EDF files with MNE and annotation spreadsheets (Excel/CSV without header
  with 3 columns: [start, end, duration]).
• Robustly parses times like HH:MM:SS,ms (comma or dot decimal). 
• Assigns a unique assoc_id per association in a JSON manifest.
• Converts absolute annotation clock times to seconds relative to EDF start.
• Keeps only intervals whose start-time falls within a specified daily window
  (default 09:00–12:00).
• Splits signals into overlapping windows; computes Morlet CWT power per channel;
  rescales to a fixed (32×32) grid; stacks two channels → (32, 32, 2) feature.
• Median–IQR normalization per channel. Label = 1 if any overlap with an
  annotation; else 0.
• Parallel processing via ProcessPoolExecutor (configurable).
• Exports compressed NPZ with: features (N,32,32,2), labels (N,), assoc_ids (N,).

Usage:
    python data_processor.py \
      --manifest /path/to/Yeni_Data.json \
      --outdir ./Arrays \
      --interval-len 0.6 --overlap-len 0.3 \
      --freq-bins 32 --time-bins 32 \
      --window-start 09:00:00 --window-end 12:00:00 \
      --chunk-sec 600 \
      --jobs auto

Manifest JSON format:
{
  "associations": [
    {"edf_path": "/abs/path/rec1.edf", "excel_path": "/abs/path/rec1.xlsx", "selected_channel": 1},
    {"edf_path": "/abs/path/rec2.edf", "excel_path": "/abs/path/rec2.csv",  "selected_channel": 1}
  ]
}
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import json
import os
import re
from datetime import datetime, timedelta, time as dtime
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import mne
from tqdm import tqdm

# ---------------------------
# Helpers: logging & parsing
# ---------------------------

def log_failure(assoc: dict, reason: str):
    eid = assoc.get("assoc_id", "?")
    edf = assoc.get("edf_path", "<missing>")
    xls = assoc.get("excel_path", "<missing>")
    print(f"[PROCESS FAILURE] ID: {eid}, EDF: {edf}, ANNOT: {xls} — {reason}")


def robust_parse_hms_to_seconds(time_str: str) -> float | None:
    """Parse 'HH:MM:SS,ms' or similar into seconds (float). Returns None on fail."""
    cleaned = str(time_str).replace(',', '.').strip()
    cleaned = re.sub(r'[^0-9:\.]', '', cleaned)
    try:
        td = pd.to_timedelta(cleaned)
        return float(td.total_seconds())
    except Exception as e:
        print(f"[Parse Error] Could not parse {repr(time_str)} → {repr(cleaned)}: {e}")
        return None


def load_swd_table(path: str) -> List[Tuple[float, float]]:
    """Load (start,end) seconds from Excel or CSV; ignore duration column."""
    try:
        df = pd.read_excel(path, header=None, usecols=[0, 1, 2])
    except Exception:
        try:
            df = pd.read_csv(path, header=None, usecols=[0, 1, 2], engine='python')
        except Exception as e:
            print(f"[Load Error] Could not open annotations: {path}: {e}")
            return []
    out = []
    for _, row in df.iterrows():
        s = robust_parse_hms_to_seconds(row[0])
        e = robust_parse_hms_to_seconds(row[1])
        if s is not None and e is not None:
            out.append((s, e))
    return out


# ---------------------------
# Windowing & features
# ---------------------------

def divide_into_intervals(signal_2d: np.ndarray, sfreq: float, interval_len: float, overlap_len: float):
    step = int((interval_len - overlap_len) * sfreq)
    win = int(interval_len * sfreq)
    if win <= 0 or step <= 0:
        raise ValueError("interval_len and overlap_len must yield positive step and window size")
    segments = []
    total = signal_2d.shape[-1]
    for start in range(0, total - win + 1, step):
        segments.append(signal_2d[:, start:start + win])
    return segments


def compute_cwt_features(seg_1d: np.ndarray, sfreq: float = 1000.0, freq_bins: int = 32, time_bins: int = 32,
                         fmin: float = 6.0, fmax: float = 30.0, n_cycles: float = 2.0) -> np.ndarray:
    freqs = np.linspace(fmin, fmax, freq_bins)
    ncyc = np.full_like(freqs, n_cycles)
    arr = seg_1d[np.newaxis, np.newaxis, :]
    try:
        power = mne.time_frequency.tfr_array_morlet(
            arr, sfreq=sfreq, freqs=freqs, n_cycles=ncyc,
            output='power', decim=1, n_jobs=1, verbose=False
        )[0, 0]
    except Exception as e:
        print(f"[CWT Error] segment length {seg_1d.shape[-1]}: {e}")
        return np.zeros((freq_bins, time_bins), dtype=float)

    p_db = 10 * np.log10(power + np.finfo(float).eps)
    if p_db.shape[1] != time_bins:
        p_db = zoom(p_db, (1, time_bins / p_db.shape[1]), order=1)
    return p_db


def robust_median_iqr(x: np.ndarray) -> np.ndarray:
    m = np.median(x)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = max(q75 - q25, 1e-6)
    return (x - m) / iqr


# ---------------------------
# Single association processing
# ---------------------------

def process_association(assoc: dict, interval_len: float, overlap_len: float,
                        freq_bins: int, time_bins: int, chunk_sec: float,
                        window_start: dtime, window_end: dtime) -> Tuple[list, list, list]:
    eid = assoc.get("assoc_id", "?")
    edf_path = assoc.get("edf_path", "")
    xls_path = assoc.get("excel_path", "")

    if not (os.path.exists(edf_path) and os.path.exists(xls_path)):
        log_failure(assoc, "missing EDF or annotation file")
        return [], [], []

    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        log_failure(assoc, f"read_raw_edf failed: {e}")
        return [], [], []

    # EDF start time
    start_time = raw.info.get('meas_date')
    if start_time is None:
        log_failure(assoc, "EDF file has no start time information")
        raw.close()
        return [], [], []
    if isinstance(start_time, (int, float)):
        start_time = datetime.fromtimestamp(start_time)

    sfreq = float(raw.info['sfreq'])
    names = raw.info['ch_names']
    n_ch = len(names)

    ch_num = int(assoc.get("selected_channel", 1))
    idx1 = max(0, min(n_ch - 1, ch_num - 1))
    # Pair selection logic
    if "SINGLE" in names[idx1].upper():
        idx2 = idx1
    else:
        idx2 = idx1 + 1 if (idx1 % 2 == 0) else idx1 - 1
        if idx2 < 0 or idx2 >= n_ch:
            idx2 = idx1

    data = raw.get_data(picks=[idx1, idx2])  # shape (2, T)
    annots = load_swd_table(xls_path)

    # Absolute clock times → seconds relative to EDF start
    midnight = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    adjusted: List[Tuple[float, float]] = []
    for s, e in annots:
        try:
            srel = (midnight + timedelta(seconds=s) - start_time).total_seconds()
            erel = (midnight + timedelta(seconds=e) - start_time).total_seconds()
            adjusted.append((srel, erel))
        except Exception as exc:
            print(f"[Anno Conv Error] ID: {eid}, {edf_path}: {exc}")

    feats, labs, ids = [], [], []

    total = data.shape[1]
    chunk = int(chunk_sec * sfreq)
    try:
        for ci in range(int(np.ceil(total / chunk))):
            cs = ci * chunk
            ce = min((ci + 1) * chunk, total)
            seg = data[:, cs:ce]
            intervals = divide_into_intervals(seg, sfreq, interval_len, overlap_len)
            base_t = cs / sfreq

            for i, iv in enumerate(intervals):
                # Wall-clock of interval start
                t0 = base_t + i * (interval_len - overlap_len)
                wall = start_time + timedelta(seconds=t0)
                if not (window_start <= wall.time() < window_end):
                    continue

                c0 = compute_cwt_features(iv[0], sfreq, freq_bins, time_bins)
                c1 = compute_cwt_features(iv[1], sfreq, freq_bins, time_bins)
                stacked = np.stack([robust_median_iqr(c0), robust_median_iqr(c1)], axis=-1)

                # Overlap labeling (half-open intervals)
                lbl = 0
                for srel, erel in adjusted:
                    if (t0 < erel) and (t0 + interval_len > srel):
                        lbl = 1
                        break

                feats.append(stacked)
                labs.append(lbl)
                ids.append(eid)
    except Exception as e:
        log_failure(assoc, f"processing loop crashed: {e}")
        raw.close()
        gc.collect()
        return [], [], []

    raw.close()
    gc.collect()
    return feats, labs, ids


# ---------------------------
# Orchestrator
# ---------------------------

def process_manifest(manifest_path: str, out_dir: str, interval_len: float, overlap_len: float,
                     freq_bins: int, time_bins: int, chunk_sec: float,
                     window_start: dtime, window_end: dtime, jobs: int | str = 'auto',
                     log_csv: str | None = None):
    with open(manifest_path, 'r') as f:
        assocs = json.load(f).get('associations', [])

    # Assign assoc_ids deterministically in manifest order
    for idx, assoc in enumerate(assocs):
        assoc['assoc_id'] = idx

    # Parallelism
    max_workers = (os.cpu_count() or 1) if jobs == 'auto' else int(jobs)

    all_f, all_l, all_ids = [], [], []
    failures = []

    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_association, assoc, interval_len, overlap_len,
                             freq_bins, time_bins, chunk_sec, window_start, window_end)
                   for assoc in assocs]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Processing"):
            try:
                feats, labs, ids = fut.result()
            except Exception as e:
                # If a worker crashed entirely, record its error
                failures.append({"assoc_id": None, "error": str(e)})
                continue
            all_f.extend(feats)
            all_l.extend(labs)
            all_ids.extend(ids)

    if not all_f:
        print("⚠️ No data was processed. Check the manifest, EDF start time, or time window filter.")
        return None

    X = np.asarray(all_f, dtype=np.float32)
    y = np.asarray(all_l, dtype=np.int64)
    assoc_ids = np.asarray(all_ids, dtype=np.int64)

    print("▶️ Features before saving:", X.shape)
    if X.ndim != 4 or X.shape[1:] != (32, 32, 2):
        raise ValueError(f"Expected X to be (N,32,32,2), got {X.shape}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'Dataset.npz')
    np.savez_compressed(out_path, features=X, labels=y, assoc_ids=assoc_ids)
    print(f"✅ Saved → {out_path}")

    if log_csv and failures:
        pd.DataFrame(failures).to_csv(log_csv, index=False)
        print(f"⚠️ Logged failures → {log_csv}")

    # Free memory early
    del X, y, assoc_ids, all_f, all_l, all_ids
    gc.collect()
    return out_path


# ---------------------------
# CLI
# ---------------------------

def parse_time(s: str) -> dtime:
    # Accept HH:MM or HH:MM:SS
    parts = [int(p) for p in s.split(':')]
    if len(parts) == 2:
        h, m = parts
        sec = 0
    elif len(parts) == 3:
        h, m, sec = parts
    else:
        raise argparse.ArgumentTypeError('Time must be HH:MM or HH:MM:SS')
    return dtime(hour=h, minute=m, second=sec)


def main():
    ap = argparse.ArgumentParser(description='Build NPZ features from EDF + annotations')
    ap.add_argument('--manifest', required=True, help='Path to JSON manifest with associations')
    ap.add_argument('--outdir', required=True, help='Output directory for Dataset.npz')

    ap.add_argument('--interval-len', type=float, default=0.6)
    ap.add_argument('--overlap-len', type=float, default=0.3)
    ap.add_argument('--freq-bins', type=int, default=32)
    ap.add_argument('--time-bins', type=int, default=32)
    ap.add_argument('--chunk-sec', type=float, default=600.0)

    ap.add_argument('--window-start', type=parse_time, default='09:00:00')
    ap.add_argument('--window-end', type=parse_time, default='12:00:00')

    ap.add_argument('--jobs', default='auto', help="Number of parallel workers (int) or 'auto'")
    ap.add_argument('--log-csv', default=None, help='Optional CSV path to record association failures')

    args = ap.parse_args()

    out = process_manifest(
        manifest_path=args.manifest,
        out_dir=args.outdir,
        interval_len=args.interval_len,
        overlap_len=args.overlap_len,
        freq_bins=args.freq_bins,
        time_bins=args.time_bins,
        chunk_sec=args.chunk_sec,
        window_start=args.window_start,
        window_end=args.window_end,
        jobs=args.jobs,
        log_csv=args.log_csv,
    )
    if out is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
