import os
import tempfile

import pytest
import torch


def pytest_configure(config: pytest.Config):
    temp_root = os.path.abspath(".pytest_cache/tmp")
    os.makedirs(temp_root, exist_ok=True)
    tempfile.tempdir = temp_root


def pytest_runtest_setup(item: pytest.Item):
    if "gpu" in item.keywords:
        os.environ["DIFFONNX_PATCHED"] = "1" if torch.cuda.is_available() else "0"
