# CWT-Hybrid CNN for Automated SWD Detection in GAERS Rats

This repository provides a full pipeline for **automated spike-and-wave discharge (SWD) detection** in **GAERS rats** using **Continuous Wavelet Transform (CWT)** features and a **hybrid CNN + multi-kernel Conv1D model with attention**.

It includes:

- **`data_processor.py`** → Builds training arrays (`.npz`) from EDF recordings + Excel annotations  
- **`model.py`** → Defines the hybrid CNN + Conv1D token encoder with SE attention  
- **`loocv.py`** → Runs Leave-One-Out Cross-Validation (LOOCV) across animals, saving fold models and metrics  

The workflow is optimized for **09:00–12:00 recordings**, following the experimental protocol.

---

## 📈 Workflow

```text
EDF + Excel Annotations
          │
          ▼
   data_processor.py
          │
          ▼
     Dataset.npz
          │
          ▼
      loocv.py
          │
 ┌────────┴─────────┐
 ▼                  ▼
Fold models (.keras)   Metrics (CSV)
```

## Installation

Clone this repo:

```bash
git clone https://github.com/omerdemir2k4/AutoSWD
cd swd-cwt-hybrid
```

Create a Python environment (Python 3.10–3.11 recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**GPU acceleration**: requires TensorFlow with CUDA/cuDNN installed.

## Data Preparation

### Expected inputs
- **EDF recordings** (8-channel EEG)
- **Excel/CSV annotations** with no header, 3 columns:

| start_time | end_time | duration |
|------------|----------|----------|
| Format: HH:MM:SS,ms or HH:MM:SS.ms |

### Manifest JSON
```json
{
  "associations": [
    {
      "edf_path": "/data/rat1.edf",
      "excel_path": "/data/rat1.xlsx",
      "selected_channel": 1
    },
    {
      "edf_path": "/data/rat2.edf",
      "excel_path": "/data/rat2.csv",
      "selected_channel": 3
    }
  ]
}

The model processes input from two EEG channels, where one channel is explicitly chosen in the manifest (selected_channel), and the second is automatically determined
```

### Generate NPZ dataset
```bash
python data_processor.py \
  --manifest manifest.json \
  --outdir ./Arrays \
  --interval-len 0.6 \
  --overlap-len 0.3 \
  --freq-bins 32 \
  --time-bins 32 \
  --window-start 09:00:00 \
  --window-end 12:00:00 \
  --chunk-sec 600 \
  --jobs auto
```

**Output**: `Arrays/Dataset.npz` with arrays:
- `features`: (N, 32, 32, 2) → CWT scalograms, 2 stacked channels
- `labels`: (N,) → binary SWD (1) vs background (0)
- `assoc_ids`: (N,) → recording/animal identifiers

## Model Training (LOOCV)

Run Leave-One-Out Cross-Validation:

```bash
python loocv.py \
  --npz ./Arrays/Dataset.npz \
  --outdir ./runs/cwt_hybrid \
  --sequence-len 100 \
  --val-split 0.20 \
  --epochs 50 \
  --batch-size 32 \
  --seed 43
```

### Outputs
- **Models per fold** → `runs/cwt_hybrid/models/model_<animal_id>.keras`
- **Per-fold metrics** → `loocv_per_fold_metrics.csv`
- **Summary (mean ± SEM)** → `loocv_summary_mean_sem.csv`

Metrics include Accuracy, Precision, Recall, Specificity, F1, Balanced Accuracy, ROC-AUC, PR-AUC, etc.

## Using Saved Models

Example inference on NPZ:

```python
import numpy as np
import tensorflow as tf

# Load dataset
data = np.load('./Arrays/Dataset.npz', allow_pickle=True)
X = data['features']  # (T, 32, 32, 2)

# Reshape into sequences
SEQUENCE_LEN = 100
n_full = X.shape[0] // SEQUENCE_LEN
X_seq = X[:n_full*SEQUENCE_LEN].reshape(n_full, SEQUENCE_LEN, 32, 32, 2)

# Load trained fold model
model = tf.keras.models.load_model(
    './runs/cwt_hybrid/models/model_test_animal_1.keras'
)

# Predict
probs = model.predict(X_seq)[:, :, 0]
preds = (probs >= 0.5).astype(int)
print(preds.shape)
```

## Reproducibility Notes

- Use `--seed` to fix random splits
- Expect small differences across GPUs/CPUs due to floating-point ops
- Ensure `sequence_len` matches between training and inference

## Troubleshooting

- **EDF load errors** → Check EDF header validity
- **Time parsing fails** → Ensure Excel times are in `HH:MM:SS,ms` or `HH:MM:SS.ms` format
- **Shape mismatch** → Model expects (32, 32, 2) patches and the same sequence length as training
- **Class imbalance** → Handled via class weights; defaults applied if a class is missing

## Requirements

```shell
numpy>=1.24,<2.3
pandas>=2.0
scipy>=1.10
tqdm>=4.66
mne>=1.6
pyEDFlib>=0.1.38
tensorflow==2.15.*
scikit-learn>=1.3
openpyxl>=3.1
protobuf<5
```

## Citation

If you use this repository, please cite:

- `data_processor.py` for dataset preparation
- `model.py` for hybrid CNN + Conv1D with SE attention  
- `loocv.py` for LOOCV training/evaluation pipeline
