"""check_gpu.py -- interrogate the GPU we're running on.

The first M0 artifact. It answers "what is this GPU made of?" and records the
two ceilings every later benchmark is measured against:

  - memory bandwidth (GB/s)    -> the limit for MEMORY-bound kernels (vector add,
                                  norms, decode -- anything that mostly moves bytes)
  - compute throughput (FLOPS) -> the limit for COMPUTE-bound kernels (big matmuls)

Those two numbers are the axes of the roofline; this script is the graph paper.

Run it ON the GPU and commit the output as the device's spec card, so no later
benchmark ever has to guess whether the T4 was 320 or 300 GB/s.

Usage:
    python setup/check_gpu.py            # print to stdout
    python setup/check_gpu.py --save     # also write reports/00_gpu.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

# Spec-sheet ceilings, LOOKED UP from the vendor datasheet -- not measured here.
# Keyed by the exact torch device name. Verify every number against NVIDIA's
# Tesla T4 datasheet and drop the real URL in; that verification is the rep.
SPEC_SHEET = {
    "Tesla T4": {
        "memory_bandwidth_GBs": 320,      # 16 GB GDDR6, ~320 GB/s
        "fp32_TFLOPs": 8.1,
        "fp16_tensor_TFLOPs": 65,         # FP16 via Tensor Cores
        "datasheet": "NVIDIA Tesla T4 datasheet -- verify these and add the URL",
    },
}


def driver_version() -> str | None:
    """The NVIDIA driver version (torch doesn't expose it; nvidia-smi does)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect() -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible -- is the runtime set to GPU?")

    p = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,   # CUDA torch was built against
        "driver_version": driver_version(),
        "device_name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",  # feature level of the silicon
        "sm_count": p.multi_processor_count,            # parallel-engine width
        "total_vram_GiB": round(p.total_memory / 1024**3, 2),
        "spec_sheet": SPEC_SHEET.get(p.name, "UNKNOWN -- add this GPU to SPEC_SHEET"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write reports/00_gpu.json")
    args = ap.parse_args()

    info = collect()
    print(json.dumps(info, indent=2))

    if args.save:
        out = Path("reports/00_gpu.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(info, indent=2) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
