"""
Entry point for Brain-JEPA pretraining with your own fMRI data.

This script patches the data loader used by src/train.py to use MyFMRIDataset
(defined in custom_dataset.py) instead of the hardcoded UKBiobank loader.

Usage:
  # Single GPU
  python my_experiment/run_pretraining.py \
      --fname my_experiment/config_single_gpu.yaml \
      --devices cuda:0

  # Two GPUs
  python my_experiment/run_pretraining.py \
      --fname my_experiment/config_single_gpu.yaml \
      --devices cuda:0 cuda:1

Before running:
  1. Run prepare_data.py to convert your fMRI data to the expected format.
  2. Edit custom_dataset.py and set ROOT_DIR and ID_FILE.
  3. Check config_single_gpu.yaml (batch size, epochs, use_bfloat16, etc.).
"""

import sys
import os

# Make sure imports resolve from the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# ── Patch the data loader BEFORE importing train ──────────────────────────
import importlib
import src.datasets.ukbiobank_scale as _orig_ds
import my_experiment.custom_dataset as _custom_ds

# Replace the factory function that src/train.py calls
_orig_ds.make_ukbiobank1k = _custom_ds.make_my_dataset

# Also patch it in the train module's namespace (it was already imported there)
import src.train as _train_mod
_train_mod.make_ukbiobank1k = _custom_ds.make_my_dataset
# ─────────────────────────────────────────────────────────────────────────

import argparse
import pprint
import yaml
import multiprocessing as mp
from src.utils.distributed import init_distributed
from src.train import main as app_main


parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, default="my_experiment/config_single_gpu.yaml")
parser.add_argument("--devices", type=str, nargs="+", default=["cuda:0"])


def process_main(rank, fname, world_size, devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging
    logging.basicConfig()
    log = logging.getLogger()
    log.setLevel(logging.INFO if rank == 0 else logging.ERROR)

    log.info(f"called-params {fname}")
    with open(fname, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    if rank == 0:
        pprint.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    log.info(f"Running... (rank: {rank}/{world_size})")
    if rank != 0:
        log.setLevel(logging.ERROR)

    app_main(args=params)


if __name__ == "__main__":
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method("spawn")
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices),
        ).start()
