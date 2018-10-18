
Pylissom [![Build Status](https://travis-ci.com/hernanbari/pylissom.svg?branch=master)](https://travis-ci.com/hernanbari/pylissom) [![Documentation Status](https://readthedocs.org/projects/pylissom/badge/?version=latest)](https://pylissom.readthedocs.io/en/latest/?badge=latest)  [![Maintainability](https://api.codeclimate.com/v1/badges/05d5a41d500fcdd8e90d/maintainability)](https://codeclimate.com/github/hernanbari/pylissom/maintainability) [![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/hernanbari/pylissom/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/hernanbari/pylissom/?branch=master) <!-- ![Coverage Status](https://coveralls.io/repos/github/hernanbari/pylissom/badge.svg?branch=master) --> [![codecov](https://codecov.io/gh/hernanbari/pylissom/branch/master/graph/badge.svg)](https://codecov.io/gh/hernanbari/pylissom)
========

![Lissom](http://homepages.inf.ed.ac.uk/jbednar/images/000506_or_map_128MB.RF-LISSOM.anim.gif)

Pylissom is a Pytorch extension implementing the LISSOM network and other tools, based in the Topographica framework.

It's split in two parts, the core nn and optim packages, which implement the network itself,
 and the datasets, models and utils packages, that consist of a researcher's toolkit akin to Topographica features.

[LISSOM](http://homepages.inf.ed.ac.uk/jbednar/rflissom_small.html) is a model of human neocortex (mainly modeled on visual cortex) at a neural column level. The model was developed by Bednar, Choe, Miikkulainen, and Sirosh, at the University of Texas

[Topographica](http://ioam.github.io/topographica/index.html) is an old software package for computational modeling of neural maps, developed by the the same team of LISSOM. The goal is to help researchers understand brain function at the level of the topographic maps that make up sensory and motor systems, but became outdated over the years.


Getting Started
---------------

The library and API documentation is at: https://pylissom.readthedocs.io/, you should check it out for a high level overview. There is an UML class diagram for reference. And for hands-on examples there are jupyter notebooks at `notebooks/`. If Github is not rendering them, we leave these links at your disposal:

[Lissom modules](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Tests_lissom_modules.ipynb)

[Linear modules](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Tests_simple_modules.ipynb)

[Optimizers](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Test_optimizers.ipynb)

[Orientation Maps](https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Orientation_preferences.ipynb)


Installation
------------

You should first install pytorch with conda as explained at: https://pytorch.org/

Then you can install pylissom by running:

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
