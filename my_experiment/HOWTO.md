# Brain-JEPA: Getting Started with Your Own Resting-State fMRI

## Overview

Brain-JEPA is a self-supervised transformer trained on parcellated fMRI time series.
It expects data split into **450 ROIs** (50 subcortical + 400 cortical) and ~160 timepoints.

---

## Step 0 — Install dependencies

```bash
conda create -n brain-jepa python=3.10
conda activate brain-jepa

cd /home/amiklos/GitHub/Brain-JEPA
pip install -r requirement.txt
```

> **Flash-Attention** (required by default config) needs a matching CUDA toolkit:
> ```bash
> pip install flash-attn==2.5.5 --no-build-isolation
> ```
> If that fails, edit `config_single_gpu.yaml` and set `attn_mode: standard`.

Check everything works:
```bash
python my_experiment/check_env.py
```

---

## Step 1 — Parcellate your fMRI data

Your resting-state fMRI needs to be parcellated into the same atlas used in the paper:

| Atlas | ROIs | Toolbox |
|-------|------|---------|
| Schaefer 2018 (17 networks, 400 parcels) | 400 cortical | nilearn / fMRIPrep |
| Tian Subcortex Scale III | 50 subcortical | nilearn / fMRIPrep |

**Using nilearn** (if you have preprocessed NIfTI images):
```python
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker

# Schaefer 400 cortical
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=400)
masker_ctx = NiftiLabelsMasker(schaefer.maps, standardize=True)
ts_cortical = masker_ctx.fit_transform("sub-001_bold_preproc.nii.gz")  # shape [T, 400]

# Tian S3 subcortical  (download from https://github.com/yetianmed/subcortex)
masker_sub = NiftiLabelsMasker("Tian_Subcortex_S3_3T_1mm.nii.gz", standardize=True)
ts_subcortical = masker_sub.fit_transform("sub-001_bold_preproc.nii.gz")  # shape [T, 50]
```

Save as a single numpy array `[450, T]` (subcortical first):
```python
import numpy as np
ts = np.concatenate([ts_subcortical.T, ts_cortical.T], axis=0)  # [450, T]
np.save("data/sub-001.npy", ts)
```

---

## Step 2 — Prepare the data directory

Run `prepare_data.py` to convert your files and create the train/val/test split:

```bash
# If you have one .npy or .csv per subject (layout B):
python my_experiment/prepare_data.py \
    --input_root /path/to/your/npy_files \
    --output_root /path/to/brain_jepa_data \
    --input_layout B \
    --val_frac 0.1 \
    --test_frac 0.1

# If you already have separate Schaefer/Tian CSVs per subject (layout A):
python my_experiment/prepare_data.py \
    --input_root /path/to/your/subjects \
    --output_root /path/to/brain_jepa_data \
    --input_layout A
```

The output will look like:
```
/path/to/brain_jepa_data/
  time_series/
    sub-001/
      fMRI.Schaefer17n400p.csv.gz
      fMRI.Tian_Subcortex_S3_3T.csv.gz
    sub-002/
      ...
  subject_splits.pkl
  normalization_params_train.npz   ← created on first training run
```

---

## Step 3 — Configure the dataset

Edit `my_experiment/custom_dataset.py` and set the two paths at the top:

```python
ROOT_DIR = "/path/to/brain_jepa_data"
ID_FILE  = "/path/to/brain_jepa_data/subject_splits.pkl"
```

Also check `seq_length` in `MyFMRIDataset.__init__`:
- Default is `490` (UKB). Match this to your actual number of timepoints.

---

## Step 4 — Adjust the config

Review `my_experiment/config_single_gpu.yaml`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `batch_size` | 4 | Reduce to 2 if OOM |
| `accumulation_steps` | 32 | Effective batch = 4×32 = 128 |
| `use_bfloat16` | true | Set false on GPUs older than Ampere |
| `attn_mode` | flash_attn | Set `standard` if flash-attn unavailable |
| `epochs` | 300 | Use 50-100 for a quick experiment |
| `num_workers` | 4 | Reduce to 0 for debugging |

---

## Step 5 — Run pretraining

```bash
cd /home/amiklos/GitHub/Brain-JEPA

# Single GPU
python my_experiment/run_pretraining.py \
    --fname my_experiment/config_single_gpu.yaml \
    --devices cuda:0

# Two GPUs
python my_experiment/run_pretraining.py \
    --fname my_experiment/config_single_gpu.yaml \
    --devices cuda:0 cuda:1
```

Checkpoints are saved every 10 epochs to `my_experiment/logs/`.

Monitor training:
```bash
tensorboard --logdir my_experiment/logs
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `CUDA out of memory` | Reduce `batch_size` to 2, or `crop_size[1]` to 128 |
| `bfloat16` error | Set `use_bfloat16: false` in config |
| `flash_attn` import error | Set `attn_mode: standard` in config |
| `AssertionError: 450 ROIs` | Your parcellation has wrong number of ROIs; check atlas |
| `KeyError: train_ids` | Re-run `prepare_data.py`; check the pickle keys |
| Shape mismatch in ViT | `crop_size` in config must match your actual [ROIs, timepoints] |

---

## Memory guidelines (approximate)

| GPU VRAM | Max batch_size | Notes |
|----------|---------------|-------|
| 8 GB | 1–2 | May need `use_bfloat16: false` |
| 16 GB | 4 | Default config |
| 24 GB | 8 | Reduce `accumulation_steps` to 16 |
| 40–80 GB | 16 | Matches paper (4× A100) |
