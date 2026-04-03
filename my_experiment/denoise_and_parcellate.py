"""
denoise_and_parcellate.py

Post-HCP denoising + parcellation pipeline (replaces ICA-FIX).

Strategy:
  1. Load volumetric BOLD (MNI, bold.nii.gz) for aCompCor computation only
  2. Load CIFTI BOLD (bold_Atlas.dtseries.nii) as primary signal source
  3. Compute FD from Movement_Regressors.txt → scrub high-motion volumes
  4. Build confound matrix: 24 motion params (Friston) + aCompCor (5 WM + 5 CSF)
  5. Regress confounds + bandpass filter (0.01–0.1 Hz) on CIFTI grayordinates
  6. Parcellate:
       Cortical    → Schaefer 400 (via dlabel CIFTI)
       Subcortical → Tian S3 50 ROIs (via CIFTI voxel coordinates)
  7. Save: denoised CIFTI + Brain-JEPA CSV files [450 ROIs x T]

If CIFTI or Schaefer dlabel are not available, falls back to volumetric NIfTI pipeline.

Usage
-----
  python my_experiment/denoise_and_parcellate.py \
      --subject_dir /analysis/HCP/BIDS/ses-24007/sub-13900 \
      --output_dir  /analysis/Argyelan/test/2026MAR20 \
      --tian_atlas  /path/to/Tian_Subcortex_S3_3T.nii.gz \
      --schaefer_dlabel /path/to/Schaefer2018_400Parcels_17Networks_order.dlabel.nii

  # Multiple subjects:
  python my_experiment/denoise_and_parcellate.py \
      --subject_dir /analysis/HCP/BIDS/ses-24007/sub-13900 \
                    /analysis/HCP/BIDS/ses-24007/sub-13901 \
      --output_dir  /analysis/Argyelan/test/2026MAR20 \
      --tian_atlas  /path/to/Tian_Subcortex_S3_3T.nii.gz \
      --schaefer_dlabel /path/to/Schaefer2018_400Parcels_17Networks_order.dlabel.nii

Atlas downloads
---------------
  Tian S3:
    git clone --depth 1 https://github.com/yetianmed/subcortex /tmp/subcortex
    File: Group-Parcellation/3T/Subcortex-Only/Tian_Subcortex_S3_3T.nii.gz

  Schaefer dlabel:
    git clone --depth 1 https://github.com/ThomasYeoLab/CBIG /tmp/cbig
    File: stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/
          HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii

Requirements
------------
  pip install nilearn nibabel numpy scipy pandas tqdm scikit-learn
"""

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ── USER-TUNABLE PARAMETERS ──────────────────────────────────────────────────
FD_THRESHOLD     = 0.2   # mm — volumes above this are scrubbed
MIN_VOLUMES      = 100   # minimum surviving volumes; run skipped if below this
N_COMPCOR        = 5     # aCompCor components from WM and CSF each
HEAD_RADIUS      = 50    # mm — for converting rotation (rad) → mm in FD
HIGH_PASS        = 0.01  # Hz
LOW_PASS         = 0.10  # Hz
SCHAEFER_N_ROIS  = 400
SCHAEFER_NETWORKS = 17
# ─────────────────────────────────────────────────────────────────────────────

# FreeSurfer aparc+aseg label sets (for aCompCor masks from volumetric BOLD)
WM_LABELS  = [2, 7, 41, 46, 251, 252, 253, 254, 255]  # cerebral + cerebellar WM
CSF_LABELS = [4, 5, 14, 15, 43, 44, 72]               # ventricles

CORTICAL_FILE    = f"fMRI.Schaefer17n{SCHAEFER_N_ROIS}p.csv.gz"
SUBCORTICAL_FILE = "fMRI.Tian_Subcortex_S3_3T.csv.gz"


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HCP denoising + Brain-JEPA parcellation")
    p.add_argument("--subject_dir", nargs="+", required=True,
                   help="Path(s) to subject dir (containing MNINonLinear/)")
    p.add_argument("--output_dir", required=True,
                   help="Output directory for parcellated CSVs")
    p.add_argument("--tian_atlas", required=True,
                   help="Path to Tian_Subcortex_S3_3T.nii.gz")
    p.add_argument("--schaefer_dlabel", default=None,
                   help="Path to Schaefer dlabel CIFTI (optional; enables CIFTI-based cortical parcellation)")
    p.add_argument("--fd_threshold", type=float, default=FD_THRESHOLD)
    p.add_argument("--n_compcor", type=int, default=N_COMPCOR)
    p.add_argument("--min_volumes", type=int, default=MIN_VOLUMES)
    p.add_argument("--n_motion_params", type=int, default=12, choices=[12, 24],
                   help="Motion regressors: 12 (6+derivatives) or 24 (adds squares)")
    p.add_argument("--csf_acompcor", action="store_true",
                   help="Include CSF in aCompCor (default: WM only)")
    p.add_argument("--hcpdata", action="store_true",
                   help="HCP non-BIDS dataset: only process REST1 LR runs (rfMRI_REST1_LR)")
    p.add_argument("--run", default=None,
                   help="Process only this run name (e.g. ses-23117_task-rest_acq-PA_run-01_bold)")
    p.add_argument("--debug", action="store_true",
                   help="Save WM/CSF masks as NIfTI for visual QC")
    return p.parse_args()


# ── FILE DISCOVERY ────────────────────────────────────────────────────────────

def find_bold_file(run_dir, run_name):
    """Pre-ICA-FIX volumetric BOLD: bold.nii.gz only."""
    bold_file = os.path.join(run_dir, f"{run_name}.nii.gz")
    return bold_file if os.path.exists(bold_file) else None


def find_cifti_file(run_dir, run_name):
    """Pre-ICA-FIX CIFTI BOLD: bold_Atlas.dtseries.nii."""
    cifti_file = os.path.join(run_dir, f"{run_name}_Atlas.dtseries.nii")
    return cifti_file if os.path.exists(cifti_file) else None


def find_hp2000_clean_cifti(run_dir, run_name):
    """ICA-FIX cleaned CIFTI: bold_Atlas_hp2000_clean.dtseries.nii."""
    f = os.path.join(run_dir, f"{run_name}_Atlas_hp2000_clean.dtseries.nii")
    return f if os.path.exists(f) else None


def find_hp2000_clean_vol(run_dir, run_name):
    """ICA-FIX cleaned volumetric: bold_hp2000_clean.nii.gz."""
    f = os.path.join(run_dir, f"{run_name}_hp2000_clean.nii.gz")
    return f if os.path.exists(f) else None


def find_runs(mni_results_dir, hcpdata=False):
    """Return list of (run_name, run_dir) for runs in Results/.
    If hcpdata=True, only return REST1 LR runs (rfMRI_REST1_LR)."""
    runs = []
    for entry in sorted(os.listdir(mni_results_dir)):
        run_dir = os.path.join(mni_results_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        if not os.path.exists(os.path.join(run_dir, "Movement_Regressors.txt")):
            continue
        if hcpdata and not ("REST1" in entry and "LR" in entry):
            continue
        runs.append((entry, run_dir))
    return runs


def find_segmentation(subject_dir):
    """Find aparc+aseg: MNI space (preferred) or T1w/FreeSurfer space."""
    mni_seg = os.path.join(subject_dir, "MNINonLinear", "aparc+aseg.nii.gz")
    if os.path.exists(mni_seg):
        return mni_seg, "MNI"
    subj_name = os.path.basename(subject_dir)
    fs_seg = os.path.join(subject_dir, "T1w", subj_name, "mri", "aparc+aseg.mgz")
    if os.path.exists(fs_seg):
        print("  WARNING: Using T1w-space aparc+aseg — slight misalignment possible.")
        return fs_seg, "T1w"
    return None, None


# ── MOTION ────────────────────────────────────────────────────────────────────

def load_motion_regressors(motion_file):
    """
    Load HCP Movement_Regressors.txt.
    Standard HCP: 12 columns (6 params + 6 derivatives).
    Translations in mm, rotations in radians.
    """
    data = np.loadtxt(motion_file)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    n_cols = data.shape[1]
    base_cols  = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    deriv_cols = [f"{c}_derivative1" for c in base_cols]
    if n_cols >= 12:
        df = pd.DataFrame(data[:, :12], columns=base_cols + deriv_cols)
    elif n_cols == 6:
        df = pd.DataFrame(data[:, :6], columns=base_cols)
        for col in deriv_cols:
            orig = col.replace("_derivative1", "")
            df[col] = df[orig].diff().fillna(0)
    else:
        raise ValueError(f"Unexpected columns in {motion_file}: {n_cols}")
    return df


def compute_fd(motion_df, head_radius=HEAD_RADIUS):
    """Framewise displacement. HCP rotations are in degrees → convert to radians → mm."""
    params = motion_df[["trans_x", "trans_y", "trans_z",
                         "rot_x", "rot_y", "rot_z"]].copy()
    params[["rot_x", "rot_y", "rot_z"]] *= (np.pi / 180) * head_radius  # degrees → mm
    diff = params.diff().abs()
    diff.iloc[0] = 0
    return diff.sum(axis=1).values


def motion_params(motion_df, n_params=12):
    """Motion regressors: 12 (6+derivatives) or 24 (adds squares of both)."""
    base_cols  = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    deriv_cols = [f"{c}_derivative1" for c in base_cols]
    df = motion_df[base_cols + deriv_cols].copy()
    if n_params == 24:
        for col in base_cols + deriv_cols:
            df[f"{col}_power2"] = df[col] ** 2
    elif n_params != 12:
        raise ValueError(f"n_motion_params must be 12 or 24, got {n_params}")
    return df.values  # [T, n_params]


# ── ACOMPCOR (from volumetric BOLD) ──────────────────────────────────────────

def make_tissue_mask(seg_img, labels, bold_img, erode=False):
    from nilearn.image import resample_to_img
    from scipy.ndimage import binary_erosion
    import nibabel as nib
    seg_data = seg_img.get_fdata()
    mask_data = np.zeros(seg_data.shape, dtype=np.int8)
    for lbl in labels:
        mask_data[seg_data == lbl] = 1
    if erode:
        mask_data = binary_erosion(mask_data.astype(bool)).astype(np.int8)
    mask_img = nib.Nifti1Image(mask_data, seg_img.affine, seg_img.header)
    return resample_to_img(mask_img, bold_img, interpolation="nearest")


def compute_acompcor(wm_signals, csf_signals=None, n_components=5, tr=0.72):
    """
    Run PCA on pre-scrubbed, pre-detrended tissue signals.

    Signals are high-pass filtered above the BOLD band (>LOW_PASS Hz) before
    PCA so that only high-frequency physiological noise (respiratory, cardiac)
    drives the components — not low-frequency neural-like fluctuations that
    are shared with cortex.

    wm_signals:  [T_kept, voxels]
    csf_signals: [T_kept, voxels] or None
    Returns:     [T_kept, n_components * n_tissues]
    """
    from sklearn.decomposition import PCA
    from scipy.signal import butter, filtfilt

    # High-pass at LOW_PASS (0.1 Hz) — keep only frequencies ABOVE the BOLD band
    nyq = 0.5 / tr
    b, a = butter(2, LOW_PASS / nyq, btype="high")

    components = []
    tissue_signals = [("WM", wm_signals)]
    if csf_signals is not None:
        tissue_signals.append(("CSF", csf_signals))
    for name, signals in tissue_signals:
        if signals.shape[1] == 0:
            print(f"    WARNING: aCompCor {name} mask is empty, skipping")
            continue
        signals_filt = filtfilt(b, a, signals, axis=0)
        n = min(n_components, signals_filt.shape[1])
        pca = PCA(n_components=n)
        comp = pca.fit_transform(signals_filt)
        print(f"    aCompCor {name}: {n} components, "
              f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance")
        components.append(comp)
    return np.hstack(components) if components else None


# ── SIGNAL CLEANING (works on any [T, features] array) ───────────────────────

def clean_signal(data, confounds, tr, high_pass, low_pass):
    """
    Detrend → regress confounds → bandpass filter.
    data:      [T_kept, features]  float  (already scrubbed)
    confounds: [T_kept, C]                (already scrubbed, aligned with data)
    Returns:   [T_kept, features]
    """
    from scipy.signal import butter, filtfilt, detrend as sp_detrend

    data = data.astype(np.float64)
    T = data.shape[0]

    # Detrend
    data = sp_detrend(data, axis=0)

    # Regress confounds (with intercept)
    X = np.column_stack([confounds, np.ones(T)])
    beta = np.linalg.lstsq(X, data, rcond=None)[0]
    data -= X @ beta

    # Bandpass
    nyq = 0.5 / tr
    b, a = butter(2, [high_pass / nyq, low_pass / nyq], btype="band")
    data = filtfilt(b, a, data, axis=0)

    return data.astype(np.float32)


# ── CIFTI PROCESSING ──────────────────────────────────────────────────────────

def load_cifti(cifti_file):
    """Load CIFTI dtseries. Returns (img, data [T, G], brain_model_axis)."""
    import nibabel as nib
    img  = nib.load(cifti_file)
    data = img.get_fdata(dtype=np.float32)   # [T, grayordinates]
    bm_ax = img.header.get_axis(1)
    return img, data, bm_ax


def save_cifti(clean_data, template_img, out_path):
    """Save cleaned [T, G] data as a new dtseries.nii using template header."""
    import nibabel as nib
    # Rebuild the time axis to match the (possibly scrubbed) T dimension
    old_time_ax = template_img.header.get_axis(0)
    new_time_ax = nib.cifti2.SeriesAxis(
        start=old_time_ax.start,
        step=old_time_ax.step,
        size=clean_data.shape[0],
        unit=old_time_ax.unit,
    )
    bm_ax = template_img.header.get_axis(1)
    new_header = nib.Cifti2Header.from_axes((new_time_ax, bm_ax))
    new_img = nib.Cifti2Image(clean_data, header=new_header,
                              nifti_header=template_img.nifti_header)
    nib.save(new_img, out_path)


def parcellate_cifti(clean_data, bm_ax, schaefer_dlabel_img, tian_img):
    """
    Parcellate cleaned CIFTI [T, G] into Schaefer 400 + Tian 50.

    Cortex:
      Uses Schaefer dlabel CIFTI — each cortical grayordinate is assigned
      a parcel label 1–400.

    Subcortex:
      CIFTI brain model axis contains MNI-space ijk voxel indices for each
      subcortical grayordinate. We map these to Tian S3 parcel labels.

    Returns: ts_cortical [400, T], ts_subcortical [50, T]
    """
    T = clean_data.shape[0]

    # ── Cortical: Schaefer dlabel ─────────────────────────────────────────────
    # dlabel covers all surface vertices (including medial wall, label=0),
    # while the dtseries only contains non-medial-wall cortical grayordinates.
    # We must look up each grayordinate's vertex index via the brain model axis.
    dlabel_data = schaefer_dlabel_img.get_fdata(dtype=np.float32)  # [1, G_dlabel]
    parcel_ids  = dlabel_data[0, :]   # [G_dlabel]; values 0–400, 0 = medial wall

    dlabel_bm_ax = schaefer_dlabel_img.header.get_axis(1)

    # Build vertex→parcel maps (left and right hemisphere separately)
    left_parcel  = np.zeros(64984, dtype=np.float32)   # generous upper bound
    right_parcel = np.zeros(64984, dtype=np.float32)
    for name, slc, model in dlabel_bm_ax.iter_structures():
        if 'LEFT'  in name:
            left_parcel[model.vertex]  = parcel_ids[slc]
        elif 'RIGHT' in name:
            right_parcel[model.vertex] = parcel_ids[slc]

    # Map each cortical grayordinate in the dtseries to its Schaefer parcel
    grayord_parcel = np.zeros(clean_data.shape[1], dtype=np.float32)
    for name, slc, model in bm_ax.iter_structures():
        if 'CORTEX_LEFT'  in name:
            grayord_parcel[slc] = left_parcel[model.vertex]
        elif 'CORTEX_RIGHT' in name:
            grayord_parcel[slc] = right_parcel[model.vertex]

    ts_cortical = np.zeros((SCHAEFER_N_ROIS, T), dtype=np.float32)
    for p in range(1, SCHAEFER_N_ROIS + 1):
        mask = grayord_parcel == p
        if mask.sum() > 0:
            ts_cortical[p - 1, :] = clean_data[:, mask].mean(axis=1)

    # ── Subcortical: map CIFTI voxels → Tian labels ───────────────────────────
    tian_data       = tian_img.get_fdata()
    tian_affine_inv = np.linalg.inv(tian_img.affine)
    n_tian          = int(tian_data.max())

    tian_signals = {p: [] for p in range(1, n_tian + 1)}

    for name, slc, model in bm_ax.iter_structures():
        if "CORTEX" in name:
            continue  # handled above

        vox_ijk  = model.voxel                                    # [V, 3] ijk in CIFTI volume space
        ones     = np.ones((len(vox_ijk), 1))
        mni_xyz  = (model.affine @ np.hstack([vox_ijk, ones]).T).T[:, :3]  # [V, 3] mm

        tian_ijk = (tian_affine_inv @ np.hstack([mni_xyz, ones]).T).T[:, :3]
        tian_ijk = np.round(tian_ijk).astype(int)

        struct_signal = clean_data[:, slc]   # [T, V]

        for v, (i, j, k) in enumerate(tian_ijk):
            if (0 <= i < tian_data.shape[0] and
                0 <= j < tian_data.shape[1] and
                0 <= k < tian_data.shape[2]):
                p = int(tian_data[i, j, k])
                if p > 0:
                    tian_signals[p].append(struct_signal[:, v])

    ts_subcortical = np.zeros((n_tian, T), dtype=np.float32)
    for p in range(1, n_tian + 1):
        if tian_signals[p]:
            ts_subcortical[p - 1, :] = np.mean(tian_signals[p], axis=0)

    return ts_cortical, ts_subcortical


# ── VOLUMETRIC PARCELLATION (fallback) ───────────────────────────────────────

def get_schaefer_atlas_volumetric():
    from nilearn import datasets
    print(f"  Fetching Schaefer {SCHAEFER_N_ROIS} volumetric atlas...")
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=SCHAEFER_N_ROIS, yeo_networks=SCHAEFER_NETWORKS, resolution_mm=2)
    return atlas.maps, list(atlas.labels)


def parcellate_volumetric(bold_img, atlas_img, atlas_labels, confounds, tr,
                          high_pass, low_pass, sample_mask):
    from nilearn.maskers import NiftiLabelsMasker
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, labels=atlas_labels,
        standardize=False, detrend=True,
        high_pass=high_pass, low_pass=low_pass, t_r=tr,
        memory_level=0, verbose=0)
    if sample_mask is not None and sample_mask.sum() < len(sample_mask):
        from nilearn.image import index_img
        keep_idx = np.where(sample_mask)[0]
        bold_img  = index_img(bold_img, keep_idx)
        confounds = confounds[keep_idx, :]
    return masker.fit_transform(bold_img, confounds=confounds).T  # [ROIs, T]


# ── OUTPUT ────────────────────────────────────────────────────────────────────

def save_brain_jepa_format(ts_cortical, ts_subcortical, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    T = ts_cortical.shape[1]

    df_ctx = pd.DataFrame(ts_cortical, columns=[f"t{i}" for i in range(T)])
    df_ctx.insert(0, "label_name", [f"cortical_{i}" for i in range(ts_cortical.shape[0])])
    df_ctx.to_csv(os.path.join(out_dir, CORTICAL_FILE), index=False, compression="gzip")

    df_sub = pd.DataFrame(ts_subcortical, columns=[f"t{i}" for i in range(T)])
    df_sub.insert(0, "label_name", [f"subcortical_{i}" for i in range(ts_subcortical.shape[0])])
    df_sub.to_csv(os.path.join(out_dir, SUBCORTICAL_FILE), index=False, compression="gzip")


def save_corrmat(ts_cortical, ts_subcortical, out_dir, tag):
    """Compute and save ROI x ROI correlation matrix (cortical + subcortical)."""
    ts = np.vstack([ts_cortical, ts_subcortical])   # [450, T]
    corrmat = np.corrcoef(ts).astype(np.float32)    # [450, 450]
    np.save(os.path.join(out_dir, f"corrmat_{tag}.npy"), corrmat)
    print(f"    [CORRMAT] Saved {tag} ({corrmat.shape[0]} ROIs) → corrmat_{tag}.npy")


# ── RUN PROCESSING ────────────────────────────────────────────────────────────

def process_run(run_name, run_dir, seg_img, schaefer_vol_atlas,
                tian_img, schaefer_dlabel_img, output_subj_dir, args):
    import nibabel as nib

    print(f"\n  Run: {run_name}")

    # ── Find files ────────────────────────────────────────────────────────────
    bold_file  = find_bold_file(run_dir, run_name)
    cifti_file = find_cifti_file(run_dir, run_name)
    motion_file = os.path.join(run_dir, "Movement_Regressors.txt")

    if bold_file is None:
        print(f"    SKIP: {run_name}.nii.gz not found")
        return None

    use_cifti = (cifti_file is not None and schaefer_dlabel_img is not None)
    print(f"    Volumetric BOLD : {os.path.basename(bold_file)}")
    if cifti_file:
        print(f"    CIFTI BOLD      : {os.path.basename(cifti_file)}"
              f"  {'(will parcellate)' if use_cifti else '(dlabel not provided, skipping)'}")

    # ── Load volumetric BOLD (needed for aCompCor + fallback parcellation) ────
    bold_img = nib.load(bold_file)
    tr       = bold_img.header.get_zooms()[3]
    n_vols   = bold_img.shape[3]
    print(f"    Shape: {bold_img.shape}   TR={tr:.3f}s")

    # ── Motion + FD + scrubbing ───────────────────────────────────────────────
    motion_df = load_motion_regressors(motion_file)
    if len(motion_df) != n_vols:
        min_len   = min(len(motion_df), n_vols)
        motion_df = motion_df.iloc[:min_len]

    fd          = compute_fd(motion_df)
    sample_mask = fd <= args.fd_threshold
    n_kept      = int(sample_mask.sum())
    pct_censored = (1 - n_kept / n_vols) * 100
    print(f"    FD   : mean={fd.mean():.3f}mm  max={fd.max():.3f}mm")
    print(f"    Scrub: {n_vols - n_kept}/{n_vols} volumes censored ({pct_censored:.1f}%)")

    if n_kept < args.min_volumes:
        print(f"    SKIP: only {n_kept} volumes survive scrubbing (min={args.min_volumes})")
        return None

    # ── Scrub motion params ───────────────────────────────────────────────────
    from nilearn.image import index_img
    from nilearn.maskers import NiftiMasker
    from scipy.signal import detrend as sp_detrend

    keep_idx       = np.where(sample_mask)[0]
    bold_scrubbed  = index_img(bold_img, keep_idx)
    motion_regs    = motion_params(motion_df, args.n_motion_params)
    motion_scrubbed = motion_regs[keep_idx, :]

    # ── Tissue masks ──────────────────────────────────────────────────────────
    wm_mask  = make_tissue_mask(seg_img, WM_LABELS,  bold_img, erode=True)
    csf_mask = make_tissue_mask(seg_img, CSF_LABELS, bold_img, erode=False)

    if args.debug:
        _dbg_dir = os.path.join(output_subj_dir, run_name)
        os.makedirs(_dbg_dir, exist_ok=True)
        nib.save(wm_mask,  os.path.join(_dbg_dir, f"{run_name}_wm_mask.nii.gz"))
        nib.save(csf_mask, os.path.join(_dbg_dir, f"{run_name}_csf_mask.nii.gz"))
        print(f"    [DEBUG] WM voxels : {int(wm_mask.get_fdata().sum())}")
        print(f"    [DEBUG] CSF voxels: {int(csf_mask.get_fdata().sum())}")
        print(f"    [DEBUG] Masks saved to {_dbg_dir}")

    # ── aCompCor: extract signals → scrub → detrend → PCA ────────────────────
    print(f"    Computing aCompCor...")
    wm_masker  = NiftiMasker(mask_img=wm_mask,  standardize=False)
    wm_signals = wm_masker.fit_transform(bold_img)        # [T, voxels]
    wm_signals = sp_detrend(wm_signals[keep_idx, :], axis=0)  # scrub + detrend

    csf_signals = None
    if args.csf_acompcor:
        csf_masker  = NiftiMasker(mask_img=csf_mask, standardize=False)
        csf_signals = csf_masker.fit_transform(bold_img)
        csf_signals = sp_detrend(csf_signals[keep_idx, :], axis=0)

    acompcor = compute_acompcor(wm_signals, csf_signals, args.n_compcor, tr)

    # ── Build confound matrix [T_kept, C] — fully aligned with scrubbed data ─
    if acompcor is not None:
        confounds_scrubbed = np.hstack([motion_scrubbed, acompcor])
    else:
        confounds_scrubbed = motion_scrubbed
    n_acompcor = confounds_scrubbed.shape[1] - args.n_motion_params
    print(f"    Confounds: {confounds_scrubbed.shape[1]} regressors "
          f"({args.n_motion_params} motion + {n_acompcor} aCompCor)")

    # ── Output directory ──────────────────────────────────────────────────────
    out_run_dir = os.path.join(output_subj_dir, run_name)
    os.makedirs(out_run_dir, exist_ok=True)

    if args.debug:
        # Save full confounds matrix (before scrubbing) as TSV
        motion_cols = [f"motion_{i}" for i in range(motion_regs.shape[1])]
        acompcor_cols = ([f"aCompCor_{i}" for i in range(acompcor.shape[1])]
                         if acompcor is not None else [])
        col_names = motion_cols + acompcor_cols
        confounds_df = pd.DataFrame(confounds_scrubbed, columns=col_names)
        confounds_df.insert(0, "volume", keep_idx)
        confounds_path = os.path.join(out_run_dir, f"{run_name}_confounds.tsv")
        confounds_df.to_csv(confounds_path, sep="\t", index=False)
        print(f"    [DEBUG] Confounds saved → {os.path.basename(confounds_path)}")

        # Split R² maps on scrubbed data
        brain_masker = NiftiMasker(mask_strategy="whole-brain-template", standardize=False)
        bold_2d = brain_masker.fit_transform(bold_scrubbed)  # [T_kept, voxels]
        bold_2d = sp_detrend(bold_2d, axis=0)
        n_motion = args.n_motion_params

        def _r2_map(regressors, y):
            X = np.column_stack([regressors, np.ones(len(regressors))])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            ss_res = ((y - X @ beta) ** 2).sum(axis=0)
            ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
            return np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0).astype(np.float32)

        r2_combined = _r2_map(confounds_scrubbed, bold_2d)
        r2_motion   = _r2_map(confounds_scrubbed[:, :n_motion], bold_2d)
        r2_acompcor = (_r2_map(confounds_scrubbed[:, n_motion:], bold_2d)
                       if acompcor is not None else np.zeros_like(r2_combined))

        for r2, tag in [(r2_combined, "r2_combined"),
                        (r2_motion,   "r2_motion"),
                        (r2_acompcor, "r2_acompcor")]:
            img  = brain_masker.inverse_transform(r2)
            path = os.path.join(out_run_dir, f"{run_name}_confound_{tag}.nii.gz")
            nib.save(img, path)
            print(f"    [DEBUG] {tag} map saved → {os.path.basename(path)}")

        # Save confound-regressed volume (no bandpass) for visual QC
        from nilearn.image import clean_img as _clean_img
        regressed_vol = _clean_img(
            bold_scrubbed,
            confounds=confounds_scrubbed,
            detrend=True,
            standardize=False,
            high_pass=None,
            low_pass=None,
            t_r=tr,
        )
        regressed_path = os.path.join(out_run_dir, f"{run_name}_confound_regressed.nii.gz")
        nib.save(regressed_vol, regressed_path)
        print(f"    [DEBUG] Confound-regressed volume saved → {os.path.basename(regressed_path)}")

    # ── Save temporal mean of scrubbed BOLD (before denoising) ───────────────
    import nibabel as nib2
    bold_scrubbed_data = bold_scrubbed.get_fdata(dtype=np.float32)
    vol_mean = bold_scrubbed_data.mean(axis=-1)
    vol_mean_img = nib2.Nifti1Image(vol_mean, bold_scrubbed.affine, bold_scrubbed.header)
    vol_mean_path = os.path.join(out_run_dir, f"{run_name}_mean.nii.gz")
    nib.save(vol_mean_img, vol_mean_path)
    print(f"    [VOL] Saved temporal mean → {os.path.basename(vol_mean_path)}")

    # ── Save denoised volumetric BOLD ─────────────────────────────────────────
    from nilearn.image import clean_img
    clean_vol = clean_img(
        bold_scrubbed,
        confounds=confounds_scrubbed,
        detrend=True,
        standardize=False,
        high_pass=HIGH_PASS,
        low_pass=LOW_PASS,
        t_r=tr,
    )
    clean_vol_path = os.path.join(out_run_dir, f"{run_name}_denoised.nii.gz")
    nib.save(clean_vol, clean_vol_path)
    print(f"    [VOL] Saved denoised volumetric → {os.path.basename(clean_vol_path)}")

    # Save denoised + mean restored
    clean_data_vol = clean_vol.get_fdata(dtype=np.float32) + vol_mean[..., np.newaxis]
    clean_mean_vol_img = nib.Nifti1Image(clean_data_vol, clean_vol.affine, clean_vol.header)
    clean_mean_vol_path = os.path.join(out_run_dir, f"{run_name}_denoised_meanadded.nii.gz")
    nib.save(clean_mean_vol_img, clean_mean_vol_path)
    print(f"    [VOL] Saved denoised+mean volumetric → {os.path.basename(clean_mean_vol_path)}")

    # ── CIFTI track (primary if available + dlabel provided) ──────────────────
    if use_cifti:
        print(f"    [CIFTI] Loading {os.path.basename(cifti_file)}...")
        cifti_img, cifti_data, bm_ax = load_cifti(cifti_file)

        # Scrub CIFTI and trim to match confounds length
        min_len    = min(cifti_data.shape[0], len(sample_mask))
        cifti_data = cifti_data[:min_len, :]
        cifti_data = cifti_data[keep_idx[keep_idx < min_len], :]

        # Save temporal mean of scrubbed CIFTI (before denoising)
        cifti_mean = cifti_data.mean(axis=0)   # [G]
        cifti_mean_path = os.path.join(out_run_dir, f"{run_name}_Atlas_mean.npy")
        np.save(cifti_mean_path, cifti_mean)
        print(f"    [CIFTI] Saved temporal mean → {os.path.basename(cifti_mean_path)}")

        print(f"    [CIFTI] Cleaning signal ({cifti_data.shape[1]} grayordinates)...")
        conf_cifti = confounds_scrubbed[:cifti_data.shape[0], :]
        clean_data = clean_signal(cifti_data, conf_cifti,
                                  tr, HIGH_PASS, LOW_PASS)   # [T_kept, G]

        # Save denoised CIFTI
        clean_cifti_path = os.path.join(out_run_dir, f"{run_name}_Atlas_denoised.dtseries.nii")
        save_cifti(clean_data, cifti_img, clean_cifti_path)
        print(f"    [CIFTI] Saved denoised CIFTI → {os.path.basename(clean_cifti_path)}")

        # Save denoised + mean restored
        clean_data_meanadded = clean_data + cifti_mean[np.newaxis, :]
        clean_cifti_mean_path = os.path.join(out_run_dir, f"{run_name}_Atlas_denoised_meanadded.dtseries.nii")
        save_cifti(clean_data_meanadded, cifti_img, clean_cifti_mean_path)
        print(f"    [CIFTI] Saved denoised+mean CIFTI → {os.path.basename(clean_cifti_mean_path)}")

        # Parcellate → Brain-JEPA format
        print(f"    [CIFTI] Parcellating (Schaefer {SCHAEFER_N_ROIS} + Tian S3)...")
        ts_cortical, ts_subcortical = parcellate_cifti(
            clean_data, bm_ax, schaefer_dlabel_img, tian_img)
        save_brain_jepa_format(ts_cortical, ts_subcortical, out_run_dir)
        print(f"    [CIFTI] Brain-JEPA CSVs saved → {out_run_dir}")

        # ── Correlation matrices ───────────────────────────────────────────────
        print(f"    [CORRMAT] Computing correlation matrices...")

        # 1. Raw (original, no denoising) — scrubbed only
        ts_ctx_raw, ts_sub_raw = parcellate_cifti(
            cifti_data, bm_ax, schaefer_dlabel_img, tian_img)
        save_corrmat(ts_ctx_raw, ts_sub_raw, out_run_dir, "raw")

        # 2. Denoised + mean added
        ts_ctx_mean, ts_sub_mean = parcellate_cifti(
            clean_data_meanadded, bm_ax, schaefer_dlabel_img, tian_img)
        save_corrmat(ts_ctx_mean, ts_sub_mean, out_run_dir, "denoised_meanadded")

        # 3. ICA-FIX hp2000_clean CIFTI (if available)
        hp_cifti_file = find_hp2000_clean_cifti(run_dir, run_name)
        hp_vol_file   = find_hp2000_clean_vol(run_dir, run_name)
        if hp_cifti_file:
            print(f"    [CORRMAT] Loading hp2000_clean CIFTI...")
            _, hp_cifti_data, hp_bm_ax = load_cifti(hp_cifti_file)
            hp_cifti_data = hp_cifti_data[keep_idx[keep_idx < hp_cifti_data.shape[0]], :]
            ts_ctx_hp, ts_sub_hp = parcellate_cifti(
                hp_cifti_data, hp_bm_ax, schaefer_dlabel_img, tian_img)
            save_corrmat(ts_ctx_hp, ts_sub_hp, out_run_dir, "hp2000_clean")
        else:
            print(f"    [CORRMAT] hp2000_clean CIFTI not found, skipping")

        if hp_vol_file:
            print(f"    [CORRMAT] hp2000_clean volumetric found → {os.path.basename(hp_vol_file)}")
        else:
            print(f"    [CORRMAT] hp2000_clean volumetric not found")

    # ── Volumetric track (fallback or when CIFTI not available) ───────────────
    else:
        print(f"    [VOL] Parcellating (Schaefer {SCHAEFER_N_ROIS})...")
        schaefer_img, schaefer_labels = schaefer_vol_atlas
        ts_cortical = parcellate_volumetric(
            bold_scrubbed, schaefer_img, schaefer_labels, confounds_scrubbed, tr,
            HIGH_PASS, LOW_PASS, sample_mask=None)

        print(f"    [VOL] Parcellating (Tian S3)...")
        tian_n = int(tian_img.get_fdata().max())
        tian_labels = [f"subcortical_{i}" for i in range(1, tian_n + 1)]
        ts_subcortical = parcellate_volumetric(
            bold_scrubbed, tian_img, tian_labels, confounds_scrubbed, tr,
            HIGH_PASS, LOW_PASS, sample_mask=None)

        save_brain_jepa_format(ts_cortical, ts_subcortical, out_run_dir)
        print(f"    [VOL] Brain-JEPA CSVs saved → {out_run_dir}")

    print(f"    cortical={ts_cortical.shape}  subcortical={ts_subcortical.shape}")

    return {"run": run_name, "n_vols": n_vols, "n_kept": n_kept,
            "pct_censored": pct_censored, "mean_fd": float(fd.mean()),
            "mode": "cifti" if use_cifti else "volumetric"}


# ── SUBJECT PROCESSING ────────────────────────────────────────────────────────

def process_subject(subject_dir, schaefer_vol_atlas, tian_img,
                    schaefer_dlabel_img, args):
    import nibabel as nib

    subj_name = os.path.basename(subject_dir)
    print(f"\n{'='*60}")
    print(f"Subject: {subj_name}")
    print(f"{'='*60}")

    results_dir = os.path.join(subject_dir, "MNINonLinear", "Results")
    if not os.path.isdir(results_dir):
        print(f"  SKIP: {results_dir} not found")
        return []

    seg_file, seg_space = find_segmentation(subject_dir)
    if seg_file is None:
        print(f"  SKIP: no aparc+aseg segmentation found")
        return []
    print(f"  Segmentation: {os.path.basename(seg_file)} ({seg_space} space)")
    seg_img = nib.load(seg_file)

    output_subj_dir = os.path.join(args.output_dir, "time_series", subj_name)
    os.makedirs(output_subj_dir, exist_ok=True)

    runs = find_runs(results_dir, hcpdata=args.hcpdata)
    if args.run:
        runs = [(n, d) for n, d in runs if n == args.run]
    print(f"  Found {len(runs)} runs: {[r[0] for r in runs]}")

    qc_records = []
    for run_name, run_dir in runs:
        result = process_run(run_name, run_dir, seg_img, schaefer_vol_atlas,
                             tian_img, schaefer_dlabel_img, output_subj_dir, args)
        if result:
            result["subject"] = subj_name
            qc_records.append(result)

    return qc_records


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    import nibabel as nib

    os.makedirs(args.output_dir, exist_ok=True)

    # Tian atlas
    if not os.path.exists(args.tian_atlas):
        print(f"ERROR: Tian atlas not found: {args.tian_atlas}")
        sys.exit(1)
    tian_img = nib.load(args.tian_atlas)
    print(f"  Tian S3 atlas: {int(tian_img.get_fdata().max())} ROIs")

    # Schaefer dlabel (optional — enables CIFTI parcellation)
    schaefer_dlabel_img = None
    if args.schaefer_dlabel:
        if not os.path.exists(args.schaefer_dlabel):
            print(f"ERROR: Schaefer dlabel not found: {args.schaefer_dlabel}")
            sys.exit(1)
        schaefer_dlabel_img = nib.load(args.schaefer_dlabel)
        print(f"  Schaefer dlabel: loaded ({args.schaefer_dlabel})")
        print(f"  Mode: CIFTI-based parcellation (cortex=surface, subcortex=volume)")
    else:
        print(f"  Schaefer dlabel: not provided → volumetric fallback")
        print(f"  Mode: volumetric parcellation (NIfTI-based)")

    # Volumetric Schaefer (used in fallback mode)
    schaefer_vol_atlas = get_schaefer_atlas_volumetric()

    # Process subjects
    all_qc = []
    for subject_dir in args.subject_dir:
        qc = process_subject(subject_dir, schaefer_vol_atlas, tian_img,
                             schaefer_dlabel_img, args)
        all_qc.extend(qc)

    # QC summary
    if all_qc:
        qc_df = pd.DataFrame(all_qc)[["subject", "run", "mode", "n_vols",
                                       "n_kept", "pct_censored", "mean_fd"]]
        qc_path = os.path.join(args.output_dir, "denoising_qc.csv")
        qc_df.to_csv(qc_path, index=False)
        print(f"\n{'='*60}")
        print(f"QC summary → {qc_path}")
        print(qc_df.to_string(index=False))

    print(f"\nDone. Output: {args.output_dir}/time_series/")
    print("Next: run prepare_data.py to create the subject split pickle.")


if __name__ == "__main__":
    main()
