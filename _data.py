"""Locations and hashes of the example data files.

The files come from the stephenslab/daarem repository (MIT license), at a
fixed commit, so that the downloaded content does not change.
"""

UPSTREAM_REPO = "https://github.com/stephenslab/daarem"
UPSTREAM_COMMIT = "7b09e4e61cae376483a36a24a05ffcd31144ee7b"
BASE_URL = (
    f"https://raw.githubusercontent.com/stephenslab/daarem/{UPSTREAM_COMMIT}/data/"
)
REGISTRY = {
    "mixdata.RData": (
        "sha256:6a2d6332ad48b89ee4d258aa000840ecfecc3133ffd2290bf00a5d0237870f4a"
    ),
}
# Set this environment variable to store the downloaded files in another folder.
DATA_DIR_ENV = "PYDAAREM_DATA_DIR"
