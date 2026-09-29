import json
from pathlib import Path

import numpy as np
import pytest

from protest_simulation.synthetic_population import build_synthetic_india

TESTS_FOLDER = Path(__file__).resolve().parent
SMALL_AGENT_COUNT = 20_000


@pytest.fixture(scope="session")
def small_population():
    return build_synthetic_india(SMALL_AGENT_COUNT, np.random.default_rng(7))


@pytest.fixture(scope="session")
def tiny_population():
    return build_synthetic_india(2_000, np.random.default_rng(11))


@pytest.fixture(scope="session")
def reference_outputs():
    return json.loads((TESTS_FOLDER / "reference_specification_outputs.json").read_text())
