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
copyright = '2026, SMT Optim contributors'
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


autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
add_module_names = False

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
        "image_light": "smt_logo.png",
        "image_dark": "smt_logo.png",
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

html_favicon = "_static/smt_logo.png"

nb_render_plugin = "default"

nb_render_markdown_format = "myst"
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",   # enable the use of $ and $$
    "amsmath",      # enable some latex commands (e.g., align)
]

nb_execution_mode = "off"
highlight_language = "python"
pygments_style = "sphinx"
nb_merge_streams = True
nb_render_text_lexer = "python"

lexers["ipython2"] = PythonLexer()
lexers["ipython"] = PythonLexer()
lexers["ipython3"] = PythonLexer()

myst_dmath_double_inline = True

# sets mathjax v3. v4 makes vertical scrollbar appear
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
