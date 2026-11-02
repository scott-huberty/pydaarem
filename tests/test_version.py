from pathlib import Path

import pydaarem
import tomllib

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_version_matches_pyproject():
    expected = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    assert pydaarem.__version__ == expected
