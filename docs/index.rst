.. pylissom documentation master file, created by
   sphinx-quickstart on Tue Jun  5 22:22:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyLissom documentation
======================

The `LISSOM <http://homepages.inf.ed.ac.uk/jbednar/rflissom_small.html>`_ family of self-organizing computational models aims to replicate the detailed development of the visual cortex of humans.

PyLissom is a PyTorch extension implementing the LISSOM networks. It's split in two parts: the core nn and optim packages, which implement the LISSOM network itself, and the datasets, models, and utils packages. Some of the datasets, models and utils of PyLissom were inspired by `Topographica <http://ioam.github.io/topographica/index.html>`_, a former implementation of the LISSOM networks oriented in its design to the neuroscience community. Instead, PyLissom was designed for a hybrid use case of the machine learning and the neuroscience communities.



.. toctree::
   :caption: Usage
   :maxdepth: 1

   usage/installation
   usage/tutorials
   usage/design

.. toctree::   
   :caption: Package Reference
   :maxdepth: 1

   _modules/pylissom
   _modules/pylissom.datasets
   _modules/pylissom.models
   _modules/pylissom.nn.functional
   _modules/pylissom.nn.modules
   _modules/pylissom.optim
   _modules/pylissom.utils
   _modules/pylissom.utils.config
   _modules/pylissom.utils.plotting
   _modules/pylissom.utils.training


Contribute
----------

- Source Code: https://github.com/hernanbari/pylissom

License
-------

The project is licensed under the GPLv3 license.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
