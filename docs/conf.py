# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IARA'
copyright = '2026, Isabela Santiago de Matos, Joao Ferri, Raul Abramo, Tessa Baker, Charles Dalang, Miguel Quartin, Riccardo Sturani, Elena Colangeli, Ian Tashiro'
author = 'Isabela Santiago de Matos, Joao Ferri, Raul Abramo, Tessa Baker, Charles Dalang, Miguel Quartin, Riccardo Sturani, Elena Colangeli, Ian Tashiro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints', '**/.ipynb_checkpoints']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo' # same layout as the GLASS docs
html_title = 'IARA' # title of the browser tabs
html_logo = 'logo.png'
html_static_path = ['_static']
html_css_files = ['custom.css']
pygments_style = 'github-dark'
pygments_dark_style = 'github-dark'

# Always dark: navy sidebar (the background of the logo) separated by a line from a dark gray
# content area with light gray text and blue links. The same colors are used whether the
# browser asks for a light or a dark page, and the light/dark toggle is hidden (custom.css).
_dark = {
    'color-foreground-primary': '#cfd0d0',
    'color-foreground-secondary': '#a0a6ae',
    'color-foreground-muted': '#81868d',
    'color-foreground-border': '#666666',
    'color-background-primary': '#131416',
    'color-background-secondary': '#1a1c1e',
    'color-background-hover': '#1e2124',
    'color-background-hover--transparent': '#1e212400',
    'color-background-border': '#303335',
    'color-background-item': '#444',
    'color-brand-primary': '#4da3ff',
    'color-brand-content': '#4da3ff',
    'color-brand-visited': '#4da3ff',
    'color-link': '#4da3ff',
    'color-link--hover': '#8cc4ff',
    'color-link--visited': '#4da3ff',
    'color-link--visited--hover': '#8cc4ff',
    'color-link-underline': '#4da3ff55',
    'color-link-underline--hover': '#8cc4ff',
    'color-sidebar-background': '#000919',
    'color-sidebar-background-border': '#1f3355',
    'color-sidebar-brand-text': '#e6edf7',
    'color-sidebar-caption-text': '#a0a6ae',
    'color-sidebar-link-text': '#cfd0d0',
    'color-sidebar-link-text--top-level': '#4da3ff',
    'color-sidebar-item-background--current': '#0d1a33',
    'color-sidebar-item-background--hover': '#0d1a33',
    'color-sidebar-item-expander-background--hover': '#0d1a33',
    'color-sidebar-search-background': '#000919',
    'color-sidebar-search-background--focus': '#0d1626',
    'color-sidebar-search-border': '#1f3355',
    'color-sidebar-search-icon': '#a0a6ae',
    'color-sidebar-search-text': '#cfd0d0',
    'color-header-background': '#000919',
    'color-header-border': '#1f3355',
    'color-header-text': '#cfd0d0',
    'color-code-background': '#0d1117',
    'color-code-foreground': '#e6edf7',
    'color-inline-code-background': '#1a1c1e',
    'color-api-background': '#1a1c1e',
    'color-api-background-hover': '#1e2124',
    'color-api-name': '#e6edf7',
    'color-api-pre-name': '#a0a6ae',
    'color-highlighted-background': '#1c2a42',
    'color-highlighted-text': '#e6edf7',
    'color-guilabel-background': '#1a1c1e',
    'color-table-header-background': '#1a1c1e',
    'color-table-border': '#303335',
    'color-admonition-background': '#1a1c1e',
    'color-card-background': '#1a1c1e',
    'color-card-border': '#303335',
    'color-toc-title-text': '#a0a6ae',
    'color-toc-item-text': '#cfd0d0',
    'color-toc-item-text--hover': '#8cc4ff',
    'color-toc-item-text--active': '#4da3ff',
}
html_theme_options = {
    'light_css_variables': _dark,
    'dark_css_variables': _dark,
    'sidebar_hide_name': True, # the logo already says IARA
}

# -- API documentation -------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.napoleon']
autosummary_generate = True
autodoc_member_order = 'bysource'
autodoc_default_options = {'members': True, 'undoc-members': True}

# Packages that are hard to install on Read the Docs are mocked
autodoc_mock_imports = [
    'mpi4py', 'camb', 'healpy', 'pymaster', 'glass', 'transformcl',
    'emcee', 'arviz', 'getdist', 'cosmology',
]