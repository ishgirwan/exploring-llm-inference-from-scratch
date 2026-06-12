"""Stage 0 smoke tests: the package imports, and the GPU marker skips correctly."""

import pytest

import common


def test_common_is_importable():
    assert common.__doc__ is not None


@pytest.mark.gpu
def test_gpu_marker_runs_only_with_cuda():
    import torch

    assert torch.cuda.is_available()
