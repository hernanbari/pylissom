
Pylissom [![Build Status](https://travis-ci.com/hernanbari/pylissom.svg?branch=master)](https://travis-ci.com/hernanbari/pylissom)
========

Pylissom is a Pytorch extension implementing the Lissom network and other tools, based in the topographica framework.

It's split in two parts, the core nn and optim packages, which implement the network itself,
 and the datasets, models and utils packages, that consist of a researcher's toolkit akin to topographica features.


The API documentation is at: https://pylissom.readthedocs.io/

Getting Started
---------------

For examples on how to use it you should check the jupyter notebooks at notebooks/.

Installation
------------

You should first install pytorch with conda as explained at: https://pytorch.org/

Then you can install pylissom by running:

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pylissom    
    
The code is hosted in pypi: https://test.pypi.org/project/pylissom/

Contributing
------------

The tests are in the `tests/` folder, and can be run with [`pytest`](https://docs.pytest.org/en/latest/). Also, the repository has [Travis CI](https://docs.travis-ci.com/) enabled, meaning every commit and Pull Request runs the tests in a virtualenv, showing as green checkmarks and red crosses in the PR page.

For any questions please contact the repo collaborators.

License
-------

The project is licensed under the GPLv3 license.
