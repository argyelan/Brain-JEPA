"""
Data preparation script for Brain-JEPA with your own resting-state fMRI data.

Expected input: per-subject time series already parcellated into 450 ROIs
  (400 cortical Schaefer + 50 subcortical Tian S3, concatenated as subcortical first)

Two supported input layouts
----------------------------
A) Two-file layout (matches original Brain-JEPA exactly):
   <input_root>/
     <subject_id>/
       fMRI.Schaefer17n400p.csv.gz     # 400 cortical ROIs x T timepoints
       fMRI.Tian_Subcortex_S3_3T.csv.gz # 50 subcortical ROIs x T timepoints

   CSV format: first column = 'label_name', remaining columns = timepoints

B) Single-file layout (simpler alternative):
   <input_root>/
     <subject_id>.csv   or  <subject_id>.npy
   Shape: [450, T] where row order = 50 subcortical + 400 cortical ROIs

This script:
  1. Validates your data
  2. Converts to the format Brain-JEPA expects (layout A)
  3. Creates the train/val/test split pickle file

Usage
-----
  python prepare_data.py \
      --input_root /path/to/your/fmri/data \
      --output_root /path/to/brain_jepa_data \
      --input_layout A   # or B \
      --val_frac 0.1 \
      --test_frac 0.1

Then set root_dir and id_file in custom_dataset.py to output_root.
"""

import os
import argparse
import pickle
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CORTICAL_ROIS = 400
SUBCORTICAL_ROIS = 50
TOTAL_ROIS = CORTICAL_ROIS + SUBCORTICAL_ROIS

CORTICAL_FILE = f"fMRI.Schaefer17n{CORTICAL_ROIS}p.csv.gz"
SUBCORTICAL_FILE = "fMRI.Tian_Subcortex_S3_3T.csv.gz"


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare fMRI data for Brain-JEPA")
    parser.add_argument("--input_root", required=True, help="Root dir of your fMRI data")
    parser.add_argument("--output_root", required=True, help="Where to write formatted data")
    parser.add_argument("--input_layout", choices=["A", "B"], default="B",
                        help="A=two-file per subject, B=single file per subject")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy_layout_a", action="store_true",
                        help="If layout A, just create the split pickle without copying files")
    return parser.parse_args()


def discover_subjects_layout_a(input_root):
    """Find subjects that have both cortical and subcortical files."""
    subjects = []
    for subj_dir in sorted(Path(input_root).iterdir()):
        if not subj_dir.is_dir():
            continue
        cortical = subj_dir / CORTICAL_FILE
        subcortical = subj_dir / SUBCORTICAL_FILE
        if cortical.exists() and subcortical.exists():
            subjects.append(subj_dir.name)
        else:
            print(f"  WARNING: {subj_dir.name} missing files, skipping")
    return subjects


def discover_subjects_layout_b(input_root):
    """Find .csv or .npy files, one per subject."""
    subjects = []
    for f in sorted(Path(input_root).iterdir()):
        if f.suffix in (".csv", ".npy") or f.name.endswith(".csv.gz"):
            subjects.append(f.stem.replace(".csv", "").replace(".npy", ""))
    return subjects


def load_single_file(path):
    """Load a [450, T] array from csv or npy."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(str(path))
    else:
        data = pd.read_csv(str(path), index_col=0).values if path.suffix in (".csv",) else \
               pd.read_csv(str(path)).values
    assert data.shape[0] == TOTAL_ROIS, \
        f"Expected {TOTAL_ROIS} ROIs (rows), got {data.shape[0]}. " \
        f"Make sure row order is 50 subcortical + 400 cortical."
    return data


def save_as_two_files(data, output_dir):
    """Save [450, T] array as Brain-JEPA's two CSV.gz files."""
    os.makedirs(output_dir, exist_ok=True)
    T = data.shape[1]

    # subcortical: first 50 rows
    subctx = data[:SUBCORTICAL_ROIS, :]
    df_sub = pd.DataFrame(subctx, columns=[f"t{i}" for i in range(T)])
    df_sub.insert(0, "label_name", [f"subcortical_{i}" for i in range(SUBCORTICAL_ROIS)])
    df_sub.to_csv(os.path.join(output_dir, SUBCORTICAL_FILE), index=False, compression="gzip")

    # cortical: next 400 rows
    ctx = data[SUBCORTICAL_ROIS:, :]
    df_ctx = pd.DataFrame(ctx, columns=[f"t{i}" for i in range(T)])
    df_ctx.insert(0, "label_name", [f"cortical_{i}" for i in range(CORTICAL_ROIS)])
    df_ctx.to_csv(os.path.join(output_dir, CORTICAL_FILE), index=False, compression="gzip")


def make_splits(subjects, val_frac, test_frac, seed):
    random.seed(seed)
    subjects = list(subjects)
    random.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_ids = subjects[:n_test]
    val_ids = subjects[n_test:n_test + n_val]
    train_ids = subjects[n_test + n_val:]
    return train_ids, val_ids, test_ids


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    ts_out = os.path.join(args.output_root, "time_series")
    os.makedirs(ts_out, exist_ok=True)

    print(f"\n=== Brain-JEPA Data Preparation ===")
    print(f"Input layout : {args.input_layout}")
    print(f"Input root   : {args.input_root}")
    print(f"Output root  : {args.output_root}")

    # 1. Discover subjects
    if args.input_layout == "A":
        subjects = discover_subjects_layout_a(args.input_root)
        print(f"Found {len(subjects)} subjects with both cortical and subcortical files.")
    else:
        subjects = discover_subjects_layout_b(args.input_root)
        print(f"Found {len(subjects)} subjects.")

    if len(subjects) == 0:
        print("ERROR: No subjects found. Check --input_root and --input_layout.")
        return

    # 2. Convert if needed (layout B → A)
    if args.input_layout == "B":
        print("\nConverting to Brain-JEPA two-file format...")
        for subj in tqdm(subjects):
            # try .csv, .csv.gz, .npy
            for ext in [".csv", ".csv.gz", ".npy"]:
                src = Path(args.input_root) / (subj + ext)
                if src.exists():
                    data = load_single_file(src)
                    save_as_two_files(data, os.path.join(ts_out, subj))
                    break
            else:
                print(f"  WARNING: could not find file for subject {subj}")

    elif args.input_layout == "A" and not args.copy_layout_a:
        # Symlink source to output (avoids copying large data)
        print("\nCreating symlinks to original data (no copying)...")
        for subj in tqdm(subjects):
            src = os.path.abspath(os.path.join(args.input_root, subj))
            dst = os.path.join(ts_out, subj)
            if not os.path.exists(dst):
                os.symlink(src, dst)

    # 3. Create train/val/test split pickle
    train_ids, val_ids, test_ids = make_splits(subjects, args.val_frac, args.test_frac, args.seed)
    splits = {"train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids}
    pkl_path = os.path.join(args.output_root, "subject_splits.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(splits, f)

    print(f"\n=== Split summary ===")
    print(f"  Train : {len(train_ids)}")
    print(f"  Val   : {len(val_ids)}")
    print(f"  Test  : {len(test_ids)}")
    print(f"  Saved : {pkl_path}")
    print(f"\nNext step: edit custom_dataset.py and set")
    print(f"  ROOT_DIR = '{args.output_root}'")
    print(f"  ID_FILE  = '{pkl_path}'")


if __name__ == "__main__":
    main()
