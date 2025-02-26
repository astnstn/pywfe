# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------

project = 'pywfe'
copyright = '2023, Austen Stone'
author = 'Austen Stone'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and directories to ignore when looking for source files.
exclude_patterns = ['pywfe/__init__.py']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Generate autosummary pages
autosummary_generate = True

# Control the order of members in autodoc
autodoc_member_order = 'bysource'

# Control whether numpydoc should include a class members toctree
numpydoc_class_members_toctree = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Autodoc options
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'show-inheritance': True,
    'inherited-members': True,
    'special-members': '__init__',
    'exclude-members': '__weakref__',
    'noindex': True,  # This prevents individual functions from appearing in the index
}

# --- OPTIONS FOR LATEX

fh = open('latex_preamble.tex', 'r+')
PREAMBLE = fh.read()
fh.close()
latex_elements = {
    'preamble': PREAMBLE,
}
