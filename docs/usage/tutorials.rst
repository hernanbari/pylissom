
Getting Started
---------------

There are Jupyter notebooks with tutorials at the `github's page <https://github.com/hernanbari/pylissom>`_ of the
project. If Github is not rendering them, we leave these links at your disposal:

`Linear modules <https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Linear_modules.ipynb>`_

`Lissom modules <https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Lissom_modules.ipynb>`_

`Optimizers <https://nbviewer.jupyter.org/github/hernanbari/pylissom/blob/master/notebooks/Optimizers.ipynb>`_

`Orientation Maps and pylissom tools <https://nbviewer.jupyter
.org/github/hernanbari/pylissom/blob/master/notebooks/Orientation_maps_and_tools.ipynb>`_

The main features provided are:

LISSOM's activation
    Consisting of several layers following the :py:class:`torch.nn.Module` interface. Found in :ref:`nn_module`.

LISSOM's hebbian learning mechanism and others
    Implemented following the :py:class:`torch.optim.Optimizer` interface. Found in :ref:`optim`.

Configuration and model building tools
    Make it easy to track and change layer's hyperparameters. Based in the :py:mod:`configobj` and :py:mod:`yaml`
    config libraries. Examples of config files and code can be found in :ref:`models` and :ref:`config`.

Common Guassian stimuli for LISSOM experiments
    Following the :py:class:`torch.utils.data.Dataset` interface. Uses the popular :py:mod:`scikit-image` and
    :py:mod:`cv2` libs. Found in :ref:`datasets` and :ref:`stimuli`.

Plotting helpers
    For displaying LISSOM layers weights and activations mainly in Jupyter Notebooks. Uses :py:mod:`matplotlib`. Found
    in :ref:`plotting`.

Training objects for simplifying :py:mod:`torch` boilerplate.
    Found in :ref:`pipeline`.
