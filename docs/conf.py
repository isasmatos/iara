# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'iara'
copyright = '2026, Isabela Santiago de Matos, Joao Ferri, Raul Abramo, Tessa Baker, Charles Dalang, Miguel Quartin, Riccardo Sturani, Elena Colangeli, Ian Tashiro'
author = 'Isabela Santiago de Matos, Joao Ferri, Raul Abramo, Tessa Baker, Charles Dalang, Miguel Quartin, Riccardo Sturani, Elena Colangeli, Ian Tashiro'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

#extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

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