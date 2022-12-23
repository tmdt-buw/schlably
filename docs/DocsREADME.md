# Documentation Tutorial

This brief Readme is intended to help setup and build automatic documentation using Sphinx.

## Installation
Make sure to install sphinx, e.g. via :
``
pip install sphinx
``

Also install the used style (called theme) of ReadTheDocs via: ``pip install sphinx-rtd-theme``



## Building a Documentation
In the terminal, navigate to the [docs](../docs) folder using ``cd docs``.

### Running for the first time or including a new file
If you have included a new file, the docs need to be updated. 

From the [docs](../docs) folder, run ``sphinx-apidoc -o source/ ../src/ -f``.
This will go through all folders in this project and check for those with an __init__.py file, 
marking them as a package. Accordingly, you should add an empty __init__.py file, if you want to include a
new folder in the documentation.

### Updating the Documentation
If you have made changes to a file, you need to rebuild the documentation.

To do so, run ```./make html``` in the terminal from the [docs](../docs) folder. 



## Checking out your documentation
Open the  [index.html](build/html/index.html) located in the docs/build/html direcotry in your browser.