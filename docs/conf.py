import os
import sys
for x in os.walk('../src'):
  sys.path.insert(0, x[0])


project = 'Models'
author = 'Alejandro Barrachina Argudo (FratosVR)'

extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension'
]

autoapi_dirs = ['../src/']
autoapi_python_use_implicit_namespaces = True

autoapi_template_dir = '_templates'

templates_path = ['_templates']
exclude_patterns = []

autodoc_mock_imports = ['ydf', 'yggdrasil-decision-forests']

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
}
autoapi_add_toctree_entry = False
html_static_path = ['_static']
html_favicon = '_static/favicon.jpg'
