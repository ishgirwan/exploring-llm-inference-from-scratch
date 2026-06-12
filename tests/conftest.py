"""Auto-skip for GPU-marked tests on machines without CUDA.

Tests that need a GPU carry ``@pytest.mark.gpu``. CI and the local Windows
machine run the GPU-free set; the full set runs on Colab / rented GPUs
(LEARNING_PATH.md, "The build loop").
"""

import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(reason="needs a CUDA GPU (torch missing or no device)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
