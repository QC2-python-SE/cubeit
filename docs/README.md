# Building the Documentation

To build the HTML documentation, you need to have Sphinx installed:

```bash
pip install sphinx sphinx-rtd-theme
```

Then, from the `docs` directory, run:

```bash
# On Linux/Mac
make html

# On Windows (if you have make)
make html

# Or directly with sphinx-build
sphinx-build -b html . _build/html
```

The HTML files will be generated in `docs/_build/html/`. Open `index.html` in your browser to view the documentation.

