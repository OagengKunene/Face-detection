import numpy as np
import pytest

from bioguard.storage import Vault


@pytest.fixture
def vector():
    return np.eye(128, dtype=np.float32)[0]


@pytest.fixture
def vault(tmp_path):
    with Vault(tmp_path / "vault.sqlite3", "a private testing passphrase", create=True) as item:
        yield item
