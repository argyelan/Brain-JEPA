"""
Quick sanity-check script. Run this first to verify your environment is set up correctly.

  python my_experiment/check_env.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=== Python / CUDA environment ===")
import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}  |  {mem_gb:.1f} GB  |  Compute {props.major}.{props.minor}")
    print(f"  bfloat16 supported: {torch.cuda.is_bf16_supported()}")

print()
print("=== Package checks ===")
pkgs = ["timm", "einops", "yaml", "pandas", "numpy", "scipy", "sklearn", "tqdm"]
for pkg in pkgs:
    try:
        m = __import__(pkg)
        print(f"  {pkg:12s} OK  ({getattr(m, '__version__', '?')})")
    except ImportError:
        print(f"  {pkg:12s} MISSING")

print()
print("=== Flash-Attention check ===")
try:
    import flash_attn
    print(f"  flash_attn   OK  ({flash_attn.__version__})")
except ImportError:
    print("  flash_attn   MISSING")
    print("  → Install: pip install flash-attn==2.5.5 --no-build-isolation")
    print("  → Or set attn_mode: standard in your config (slower but works)")

print()
print("=== Gradient CSV check ===")
grad_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "gradient_mapping_450.csv")
if os.path.exists(grad_path):
    import pandas as pd
    df = pd.read_csv(grad_path)
    print(f"  gradient_mapping_450.csv  OK  shape={df.shape}")
else:
    print(f"  MISSING: {grad_path}")

print()
print("=== custom_dataset.py config check ===")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "custom_dataset",
        os.path.join(os.path.dirname(__file__), "custom_dataset.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = mod.ROOT_DIR
    id_f = mod.ID_FILE
    root_ok = os.path.isdir(root) and root != "/path/to/your/brain_jepa_data"
    id_ok   = os.path.isfile(id_f) and id_f != "/path/to/your/brain_jepa_data/subject_splits.pkl"
    print(f"  ROOT_DIR  : {root}  {'OK' if root_ok else '** PLEASE UPDATE **'}")
    print(f"  ID_FILE   : {id_f}  {'OK' if id_ok else '** PLEASE UPDATE **'}")

    if root_ok and id_ok:
        import pickle
        with open(id_f, "rb") as f:
            splits = pickle.load(f)
        for split in ("train", "val", "test"):
            print(f"    {split:5s}: {len(splits.get(split + '_ids', []))} subjects")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone.")
