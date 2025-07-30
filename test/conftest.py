import os
import tempfile

import pytest


def pytest_configure(config: pytest.Config):
    temp_root = os.path.abspath(".pytest_cache/tmp")
    os.makedirs(temp_root, exist_ok=True)
    tempfile.tempdir = temp_root
