# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'smt-optim'
copyright = '2026, O. Cordelier'
author = 'O. Cordelier'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_collections',
    'myst_parser']


autosummary_generate = True

napoleon_numpy_docstring = True


collections = {
    "tutorial": {
        "driver": "copy_folder",
        "source": "../examples",
    }
}


html_theme_options = {
    "show_nav_level": 2,

    # header
    "logo": {
        "text": "smt-optim",
        "image_light": "https://avatars.githubusercontent.com/u/26074483?s=200&v=4",
        "image_dark": "https://avatars.githubusercontent.com/u/26074483?s=200&v=4",
    },
}


source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# html_logo = "https://avatars.githubusercontent.com/u/26074483?s=200&v=4"
html_favicon = "https://avatars.githubusercontent.com/u/26074483?s=200&v=4"

html_sidebars = {
    "get-started": [],
    "concepts": [],
}
