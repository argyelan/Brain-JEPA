# fMRI Denoising and Parcellation Pipeline

`denoise_and_parcellate.py` takes raw HCP-preprocessed fMRI data and produces
denoised, parcellated timeseries ready for Brain-JEPA pretraining. It replaces
ICA-FIX as the denoising step, giving full control over the confound strategy.

---

## What it does

1. **FD-based scrubbing** — computes framewise displacement from `Movement_Regressors.txt` and censors high-motion volumes (default threshold: 0.2 mm)
2. **aCompCor** — extracts WM (and optionally CSF) signals from the volumetric BOLD, high-pass filters them above 0.1 Hz, and runs PCA to get noise components
3. **Motion regressors** — 12 (6 params + derivatives) or 24 (adds squares, Friston-24)
4. **Signal cleaning** — detrend → regress confounds → bandpass filter (0.01–0.1 Hz), applied to CIFTI grayordinates
5. **Parcellation** — Schaefer 400 (cortical, surface-based via dlabel) + Tian S3 (50 subcortical, volume-based)
6. **Correlation matrices** — 450×450 ROI–ROI Pearson correlation saved for: raw, denoised+mean, and ICA-FIX hp2000_clean (for QC comparison)
7. **Outputs** — denoised volumetric NIfTI, denoised CIFTI, Brain-JEPA CSV files, and QC summary

---

## Pipeline order (important)

```
Load BOLD → compute FD → scrub volumes →
scrub motion params → extract WM (± CSF) signals →
scrub WM → detrend WM → high-pass filter WM (>0.1 Hz) → PCA →
build confound matrix [T_kept × C] →
save temporal mean → clean_signal (detrend + regress + bandpass) →
add mean back → save denoised+mean outputs →
parcellate → save Brain-JEPA CSVs → compute correlation matrices
```

---

## Outputs per run

| File | Description |
|------|-------------|
| `*_mean.nii.gz` | Temporal mean (volumetric) |
| `*_denoised.nii.gz` | Denoised volumetric (mean removed) |
| `*_denoised_meanadded.nii.gz` | Denoised volumetric with mean restored |
| `*_Atlas_mean.npy` | Temporal mean (CIFTI grayordinates) |
| `*_Atlas_denoised.dtseries.nii` | Denoised CIFTI |
| `*_Atlas_denoised_meanadded.dtseries.nii` | Denoised CIFTI with mean restored |
| `fMRI.Schaefer17n400p.csv.gz` | Parcellated cortical timeseries [T × 400] |
| `fMRI.Tian_Subcortex_S3_3T.csv.gz` | Parcellated subcortical timeseries [T × 50] |
| `corrmat_raw.npy` | 450×450 correlation matrix from raw BOLD |
| `corrmat_denoised_meanadded.npy` | 450×450 correlation matrix from denoised+mean |
| `corrmat_hp2000_clean.npy` | 450×450 correlation matrix from ICA-FIX (QC reference) |

Debug outputs (with `--debug`): `*_wm_mask.nii.gz`, `*_csf_mask.nii.gz`, `*_confounds.tsv`, `*_confound_r2_*.nii.gz`, `*_confound_regressed.nii.gz`

---

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--subject_dir` | required | Path(s) to subject dir(s) containing `MNINonLinear/` |
| `--output_dir` | required | Output root directory |
| `--tian_atlas` | required | Path to `Tian_Subcortex_S3_3T.nii.gz` |
| `--schaefer_dlabel` | None | Path to Schaefer 400 dlabel CIFTI (enables surface-based parcellation) |
| `--fd_threshold` | 0.2 | FD scrubbing threshold in mm |
| `--min_volumes` | 100 | Skip run if fewer than this many volumes survive scrubbing |
| `--n_compcor` | 5 | Number of aCompCor PCA components per tissue |
| `--n_motion_params` | 12 | Motion regressors: `12` (6+derivatives) or `24` (Friston-24) |
| `--csf_acompcor` | off | Include CSF aCompCor in addition to WM |
| `--run` | None | Process only this specific run name |
| `--hcpdata` | off | Non-BIDS HCP data: only process `rfMRI_REST1_LR` runs |
| `--debug` | off | Save intermediate QC volumes |

---

## Atlas paths (on this system)

```
Tian S3:
  /analysis/Argyelan/atlases/Subcortex-Only/Tian_Subcortex_S3_3T.nii

Schaefer 400 dlabel:
  /analysis/Argyelan/atlases/brain_parcellation/Schaefer2018_LocalGlobal/
  Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii
```

---

## Example calls

### Activate environment

```bash
source ~/.venvs/brain-jepa/bin/activate
```

### Single subject, all runs (default: 12 motion params, WM aCompCor only)

```bash
python my_experiment/denoise_and_parcellate.py \
    --subject_dir /analysis/HCP/BIDS/ses-24007/sub-13900 \
    --output_dir  /analysis/Argyelan/test/2026MAR20 \
    --tian_atlas  /analysis/Argyelan/atlases/Subcortex-Only/Tian_Subcortex_S3_3T.nii \
    --schaefer_dlabel /analysis/Argyelan/atlases/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii
```

### Single run, 24 motion params + CSF aCompCor (for comparison with ICA-FIX)

```bash
python my_experiment/denoise_and_parcellate.py \
    --subject_dir /analysis/HCP/BIDS/ses-23117/sub-12485 \
    --output_dir  /analysis/Argyelan/test/2026APR03 \
    --tian_atlas  /analysis/Argyelan/atlases/Subcortex-Only/Tian_Subcortex_S3_3T.nii \
    --schaefer_dlabel /analysis/Argyelan/atlases/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii \
    --run ses-23117_task-rest_acq-PA_run-01_bold \
    --n_motion_params 24 \
    --csf_acompcor
```

### Multiple subjects

```bash
python my_experiment/denoise_and_parcellate.py \
    --subject_dir /analysis/HCP/BIDS/ses-23117/sub-12485 \
                  /analysis/HCP/BIDS/ses-24007/sub-13900 \
    --output_dir  /analysis/Argyelan/test/2026APR03 \
    --tian_atlas  /analysis/Argyelan/atlases/Subcortex-Only/Tian_Subcortex_S3_3T.nii \
    --schaefer_dlabel /analysis/Argyelan/atlases/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii \
    --n_motion_params 24 \
    --csf_acompcor
```

### HCP non-BIDS data (rfMRI_REST1_LR only)

```bash
python my_experiment/denoise_and_parcellate.py \
    --subject_dir /path/to/hcp/subject \
    --output_dir  /analysis/Argyelan/test/2026APR03 \
    --tian_atlas  /analysis/Argyelan/atlases/Subcortex-Only/Tian_Subcortex_S3_3T.nii \
    --schaefer_dlabel /analysis/Argyelan/atlases/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_400Parcels_17Networks_order.dlabel.nii \
    --hcpdata
```

---

## Next step

After running, feed the output into `prepare_data.py` to create the Brain-JEPA subject split:

```bash
python my_experiment/prepare_data.py \
    --input_root /analysis/Argyelan/test/2026APR03/time_series \
    --output_root /analysis/Argyelan/test/2026APR03 \
    --input_layout A
```

Then see `HOWTO.md` for training.
