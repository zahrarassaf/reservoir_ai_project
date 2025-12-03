"""
Sphinx configuration for documentation.
"""

import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'Reservoir AI'
copyright = f'{datetime.now().year}, Reservoir AI Team'
author = 'Reservoir AI Team'

# Version
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx_rtd_theme',
    'myst_parser',
    'sphinxcontrib.mermaid',
    'nbsphinx',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = [('Returns', 'params_style')]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_class_signature = 'separated'
autodoc_typehints = 'signature'
autodoc_typehints_format = 'short'
autodoc_type_aliases = {
    'ArrayLike': 'numpy.typing.ArrayLike',
    'PathLike': 'os.PathLike',
}

# Template paths
templates_path = ['_templates']

# File extensions
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Master document
master_doc = 'index'

# Exclude patterns
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# HTML theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': True,
    'prev_next_buttons_navigation': True,
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Static files
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# Logo
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# Sidebar
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
        'localtoc.html',
    ]
}

# Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
}

# Graphviz
graphviz_output_format = 'svg'
graphviz_dot_args = [
    '-Grankdir=TB',
    '-Gnodesep=0.5',
    '-Granksep=1.0',
    '-Gfontname=Helvetica',
    '-Nfontname=Helvetica',
    '-Efontname=Helvetica',
]

# Todo
todo_include_todos = True
todo_link_only = True

# MyST
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'fieldlist',
    'html_admonition',
    'html_image',
    'colon_fence',
    'smartquotes',
    'replacements',
    'linkify',
    'substitution',
]
myst_heading_anchors = 3
myst_footnote_transition = True

# Nbsphinx
nbsphinx_execute = 'never'
nbsphinx_prolog = """
.. note:: This notebook can be downloaded from the repository: https://github.com/yourusername/reservoir-ai
"""
