#!/usr/bin/env python
"""Quick GPU / hardware check.

Run this first to verify PyTorch sees your GPU before starting experiments.

Usage
-----
  python scripts/check_gpu.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK  = "[OK]"
ERR = "[!!]"
WARN = "[--]"


def main() -> None:
    print("=" * 60)
    print("DE-INTERACT  --  Hardware Check")
    print("=" * 60)

    # ---- Python
    import platform
    print(f"\nPython      : {sys.version.split()[0]}")
    print(f"Platform    : {platform.platform()}")

    # ---- PyTorch
    try:
        import torch
        print(f"\nPyTorch     : {torch.__version__}  {OK}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA avail  : {cuda_ok}")
        if cuda_ok:
            n = torch.cuda.device_count()
            print(f"GPU count   : {n}")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024**3
                print(
                    f"  GPU {i}     : {props.name}"
                    f"  |  {mem_gb:.1f} GB VRAM"
                    f"  |  SM {props.major}.{props.minor}"
                )
            cur = torch.cuda.current_device()
            print(f"Default GPU : cuda:{cur} ({torch.cuda.get_device_name(cur)})")

            # Quick smoke test
            x = torch.randn(128, 128, device="cuda")
            y = x @ x.T
            shape = tuple(y.shape)
            del x, y
            print(f"\n{OK} Matmul smoke test on GPU passed (shape: {shape})")
        else:
            print(f"\n{WARN} No CUDA GPU found -- training will run on CPU (very slow).")
            print(f"    Re-install PyTorch with your CUDA version from https://pytorch.org")
    except OSError as exc:
        print(f"\n{ERR} PyTorch DLL load error: {exc}")
        print("    Your PyTorch wheel does not match your CUDA driver.")
        print("    Re-install from https://pytorch.org/get-started/locally/")
        sys.exit(1)
    except ImportError:
        print(f"\n{ERR} PyTorch not installed. See README Step 2.")
        sys.exit(1)

    # ---- RDKit
    try:
        from rdkit import Chem, rdBase
        mol = Chem.MolFromSmiles("CCO")
        assert mol is not None
        print(f"RDKit       : {rdBase.rdkitVersion}  {OK}")
    except ImportError:
        print(f"{ERR} RDKit not installed. Run: pip install rdkit")

    # ---- scikit-fingerprints
    try:
        import skfp  # noqa: F401
        from importlib.metadata import version as _pkg_version
        skfp_ver = _pkg_version("scikit-fingerprints")
        print(f"skfp        : {skfp_ver}  {OK}")
    except ImportError:
        print(f"{ERR} scikit-fingerprints not installed. Run: pip install scikit-fingerprints")

    # ---- scikit-learn
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}  {OK}")
    except ImportError:
        print(f"{ERR} scikit-learn not installed.")

    # ---- tqdm
    try:
        import tqdm
        print(f"tqdm        : {tqdm.__version__}  {OK}")
    except ImportError:
        print(f"{ERR} tqdm not installed. Run: pip install tqdm")

    # ---- config sanity
    try:
        from src.utils.config import load_config
        cfg = load_config()
        families = list(cfg["features"]["families"].keys())
        device_pref = cfg.get("hardware", {}).get("device", "auto")
        print(f"\nConfig      : loaded  {OK}")
        print(f"  Feature families : {families}")
        print(f"  Device setting   : {device_pref}")
    except Exception as exc:
        print(f"\n{ERR} Config load failed: {exc}")

    print("\n" + "=" * 60)
    print("All checks done. If CUDA avail = True, you are good to go.")
    print("=" * 60)


if __name__ == "__main__":
    main()
