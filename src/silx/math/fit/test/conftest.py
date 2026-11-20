import copy

import pytest

from silx.math.fit import bgtheories

_DEFAULT_CONFIG = copy.deepcopy(bgtheories.CONFIG)


@pytest.fixture(autouse=True)
def _default_bgtheories_config(monkeypatch):
    """Run each test with the default background configuration"""
    monkeypatch.setattr(bgtheories, "CONFIG", copy.deepcopy(_DEFAULT_CONFIG))
