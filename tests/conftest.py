import sys
from pathlib import Path

# Prioritise local repo modules over any site-packages editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture(scope="session")
def device(request):
    name_value = request.config.option.device
    if name_value is None:
        pytest.skip()
    return name_value
