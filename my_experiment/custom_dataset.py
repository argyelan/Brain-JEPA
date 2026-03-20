"""
Drop-in replacement dataset for Brain-JEPA pretraining.

Differences from the original ukbiobank_scale.py:
  - ROOT_DIR and ID_FILE are set here (no hardcoded paths in src/)
  - Works with either two-file layout or single-file layout
  - Patches into src/train.py via monkey-patching in run_pretraining.py

Set ROOT_DIR and ID_FILE below, then run:
  python run_pretraining.py --fname config_single_gpu.yaml --devices cuda:0
"""

import os
import sys

# ── USER CONFIG ──────────────────────────────────────────────────────────────
# Directory that contains the time_series/ subfolder (created by prepare_data.py)
ROOT_DIR = "/path/to/your/brain_jepa_data"

# Pickle file with train_ids, val_ids, test_ids (created by prepare_data.py)
ID_FILE = "/path/to/your/brain_jepa_data/subject_splits.pkl"
# ─────────────────────────────────────────────────────────────────────────────

# Add parent so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from logging import getLogger
from tqdm import tqdm

logger = getLogger()

CORTICAL_ROIS = 400
SUBCORTICAL_ROIS = 50
CORTICAL_FILE = f"fMRI.Schaefer17n{CORTICAL_ROIS}p.csv.gz"
SUBCORTICAL_FILE = "fMRI.Tian_Subcortex_S3_3T.csv.gz"


class MyFMRIDataset(Dataset):
    """
    Loads resting-state fMRI parcellated time series.

    Expects (after running prepare_data.py):
      <ROOT_DIR>/time_series/<subject_id>/
          fMRI.Schaefer17n400p.csv.gz       # rows=ROIs, col 0=label_name, rest=timepoints
          fMRI.Tian_Subcortex_S3_3T.csv.gz

    Returns:
      {'fmri': tensor of shape [1, 450, num_frames]}
    """

    def __init__(
        self,
        split="train",
        seq_length=490,          # original number of timepoints in your data
        downsample=True,
        sampling_rate=3,
        num_frames=160,
        use_standatdization=False,
        root_dir=ROOT_DIR,
        id_file=ID_FILE,
    ):
        self.root_dir = root_dir
        self.ts_dir = os.path.join(root_dir, "time_series")
        self.seq_length = seq_length
        self.downsample = downsample
        self.sampling_rate = sampling_rate
        self.num_frames = num_frames
        self.use_standatdization = use_standatdization

        # Load subject IDs
        with open(id_file, "rb") as f:
            splits = pickle.load(f)
        self.ids = splits[f"{split}_ids"]
        logger.info(f"[MyFMRIDataset] {split}: {len(self.ids)} subjects")

        # Compute or load robust scaling params
        params_path = os.path.join(root_dir, "normalization_params_train.npz")
        if os.path.exists(params_path):
            p = np.load(params_path)
            self.medians, self.iqrs = p["medians"], p["iqrs"]
            logger.info("Normalization params loaded.")
        else:
            logger.info("Computing normalization params from training set...")
            self.medians, self.iqrs = self._compute_norm_params(splits["train_ids"], params_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        subj = self.ids[idx]
        subj_dir = os.path.join(self.ts_dir, subj)

        ctx = self._read_csv(os.path.join(subj_dir, CORTICAL_FILE))     # [400, T]
        sub = self._read_csv(os.path.join(subj_dir, SUBCORTICAL_FILE))  # [50, T]
        ts = np.concatenate([sub, ctx], axis=0).astype(np.float32)      # [450, T]

        if not self.use_standatdization:
            ts = (ts - self.medians[:, None]) / (self.iqrs[:, None] + 1e-8)

        ts_t = torch.from_numpy(ts)  # [450, T]

        if self.downsample:
            clip = self.sampling_rate * self.num_frames
            delta = max(ts_t.shape[1] - clip, 0)
            start = int(random.uniform(0, delta))
            end = start + clip - 1
            idx_t = torch.linspace(start, end, self.num_frames).clamp(0, ts_t.shape[1] - 1).long()
            ts_t = ts_t[:, idx_t]

        if self.use_standatdization:
            ts_t = (ts_t - ts_t.mean()) / (ts_t.std() + 1e-8)

        return {"fmri": ts_t.unsqueeze(0).float()}  # [1, 450, num_frames]

    def _read_csv(self, path):
        """Read CSV.gz; skip the first column (label_name), return [ROIs, T]."""
        df = pd.read_csv(path)
        # first column = label names; rest = timepoints 0..T-1
        return df.iloc[:, 1:self.seq_length + 1].values

    def _compute_norm_params(self, train_ids, save_path):
        means = []
        for subj in tqdm(train_ids, desc="Norm params"):
            subj_dir = os.path.join(self.ts_dir, subj)
            ctx = self._read_csv(os.path.join(subj_dir, CORTICAL_FILE))
            sub = self._read_csv(os.path.join(subj_dir, SUBCORTICAL_FILE))
            data = np.concatenate([sub, ctx], axis=0)  # [450, T]
            means.append(data.mean(axis=1))             # [450]
        means = np.stack(means)                         # [N, 450]
        medians = np.median(means, axis=0)
        iqrs = np.percentile(means, 75, axis=0) - np.percentile(means, 25, axis=0)
        np.savez(save_path, medians=medians, iqrs=iqrs)
        return medians, iqrs


def make_my_dataset(
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=4,
    world_size=1,
    rank=0,
    drop_last=True,
    downsample=True,
    use_standatdization=False,
    portion=1,
):
    """Factory function compatible with Brain-JEPA's training loop."""
    dataset = MyFMRIDataset(
        split="train",
        downsample=downsample,
        use_standatdization=use_standatdization,
    )
    if portion < 1:
        n = int(len(dataset) * portion)
        dataset = torch.utils.data.Subset(dataset, list(range(n)))

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset, num_replicas=world_size, rank=rank
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
    )
    return dataset, loader, sampler
