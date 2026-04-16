# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

from sphinx.highlighting import lexers
from pygments.lexers import PythonLexer

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
    # 'nbsphinx',
    'sphinx_collections',
    # 'myst_parser',
    'myst_nb',
]


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
    # '.txt': 'markdown',
    '.md': 'myst-nb',
    # '.ipynb': 'myst-nb',
}


templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']

# html_logo = "https://avatars.githubusercontent.com/u/26074483?s=200&v=4"
html_favicon = "https://avatars.githubusercontent.com/u/26074483?s=200&v=4"

html_sidebars = {
    "get-started": [],
    "concepts": [],
}

nb_render_plugin = "default"

nb_render_markdown_format = "myst"
myst_enable_extensions = ["colon_fence"]

nb_execution_mode = "off"
highlight_language = "python"
pygments_style = "sphinx"
nb_merge_streams = True
nb_render_text_lexer = "python"

lexers["ipython2"] = PythonLexer()
lexers["ipython"] = PythonLexer()
lexers["ipython3"] = PythonLexer()