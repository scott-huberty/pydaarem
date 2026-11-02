"""Sphinx configuration for the pydaarem documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from branding import DARK, LIGHT  # noqa: E402

project = "pydaarem"
author = "pydaarem contributors"
copyright = "pydaarem contributors"

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
]

exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "pydaarem"
html_theme_options = {
    "light_css_variables": LIGHT,
    "dark_css_variables": DARK,
    "source_repository": "https://github.com/scott-huberty/pydaarem",
    "source_branch": "main",
    "source_directory": "docs/",
}

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "callable": ":term:`python:callable`",
    "ndarray": "numpy.ndarray",
    "dict": ":class:`python:dict`",
}
numpydoc_xref_ignore = {"optional", "default", "shape", "of", "or"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "matplotlib_animations": False,
}
