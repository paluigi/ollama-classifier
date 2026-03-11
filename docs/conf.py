"""Sphinx configuration for ollama-classifier documentation."""

import os
import sys

# Add the source directory to the path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "ollama-classifier"
copyright = "2024, Luigi Palumbo"
author = "Luigi Palumbo"

# Read version from package
with open("../pyproject.toml") as f:
    for line in f:
        if line.startswith("version ="):
            version = line.split('"')[1]
            break

release = version

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "show-inheritance": True,
    "undoc-members": True,
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# MyST parser settings
myst_enable_extensions = ["colon_fence"]

# Templates path
templates_path = ["_templates"]

# Source file extensions
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# Theme configuration
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# Static files
html_static_path = ["_static"]