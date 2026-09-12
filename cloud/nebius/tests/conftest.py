import copy
import json
from pathlib import Path

import pytest

from cloud.nebius.journal import Journal


@pytest.fixture
def config():
    return copy.deepcopy(json.loads((Path(__file__).parents[1] / "cartpole.json").read_text()))


@pytest.fixture
def journal(tmp_path, config):
    receipt = Journal(tmp_path / "receipt.sqlite")
    receipt.plan(config, [1, 2, 3], 1)
    yield receipt
    receipt.close()
