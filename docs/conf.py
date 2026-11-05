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

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
