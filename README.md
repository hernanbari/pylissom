
PyLissom [![Build Status](https://travis-ci.com/hernanbari/pylissom.svg?branch=master)](https://travis-ci.com/hernanbari/pylissom) [![Documentation Status](https://readthedocs.org/projects/pylissom/badge/?version=latest)](https://pylissom.readthedocs.io/en/latest/?badge=latest)  [![Maintainability](https://api.codeclimate.com/v1/badges/05d5a41d500fcdd8e90d/maintainability)](https://codeclimate.com/github/hernanbari/pylissom/maintainability) [![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/hernanbari/pylissom/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/hernanbari/pylissom/?branch=master) <!-- ![Coverage Status](https://coveralls.io/repos/github/hernanbari/pylissom/badge.svg?branch=master) --> [![codecov](https://codecov.io/gh/hernanbari/pylissom/branch/master/graph/badge.svg)](https://codecov.io/gh/hernanbari/pylissom)
========

The [LISSOM](http://homepages.inf.ed.ac.uk/jbednar/rflissom_small.html) family of self-organizing computational models aims to replicate the detailed development of the visual cortex of humans.

PyLissom is a Pytorch extension implementing the LISSOM networks. It's split in two parts: the core nn and optim packages, which implement the LISSOM network itself, and the datasets, models, and utils packages. 

Some of the datasets, models and utils of PyLissom were inspired by [Topographica](http://ioam.github.io/topographica/index.html), a former implementation of the LISSOM networks oriented in its design to the neuroscience community. Instead, PyLissom was designed for a hybrid use case of the machine learning and the neuroscience communities.


Getting Started
---------------

The library and API documentation are at: https://pylissom.readthedocs.io/, you should check it out for a high level overview. There is an UML class diagram for reference. For hands-on examples there are jupyter notebooks at `notebooks/`. If Github is not rendering them, we leave these links at your disposal:

[Lissom modules](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Lissom_modules.ipynb)

[Linear modules](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Linear_modules.ipynb)

[Optimizers](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Optimizers.ipynb)

[Orientation Maps and pylissom tools](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Orientation_maps_and_tools.ipynb)


The main features provided by **pylissom** are:

- LISSOM's activation

- LISSOM's hebbian learning mechanism and others

- Configuration and model building tools

- Common Guassian stimuli for LISSOM experiments

- Plotting helpers

- Training pipeline objects


Installation
------------

You should first install PyTorch with conda as explained at: https://pytorch.org/

Then you can install PyLissom by running:

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pylissom    
    
The code is hosted in pypi: https://test.pypi.org/project/pylissom/

Contributing
------------

The tests are in the `tests/` folder, and can be run with [`pytest`](https://docs.pytest.org/en/latest/). Also, the repository has [Travis CI](https://docs.travis-ci.com/) enabled, meaning every commit and Pull Request runs the tests in a virtualenv, showing as green checkmarks and red crosses in the PR page. These are all the integrations links of the repo:

Travis - Continuous Integration: [repo_page](https://travis-ci.com/hernanbari/pylissom)

Codecov - Code coverage: [repo_page](https://codecov.io/gh/hernanbari/pylissom)

Scrutinizer - Code health: [repo_page](https://scrutinizer-ci.com/g/hernanbari/pylissom/)

CodeClimate - Maintainability: [repo_page](https://codeclimate.com/github/hernanbari/pylissom)

ReadTheDocs - Documentation: [repo_page](https://readthedocs.org/projects/pylissom/)

For any questions please contact the repo collaborators.

License
-------

The project is licensed under the GPLv3 license.
