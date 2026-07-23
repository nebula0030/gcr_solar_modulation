"""Guards that the venv has everything later tasks import."""
from __future__ import annotations

import sys


def test_python_is_at_least_3_9():
    assert sys.version_info[:2] >= (3, 9)


def test_all_required_packages_import():
    import netCDF4  # noqa: F401
    import numpy  # noqa: F401
    import plotly  # noqa: F401
    import requests  # noqa: F401
