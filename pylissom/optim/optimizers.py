"""
Extends the py:class:`torch.optim.Optimizer` class with Lissom optimizers, mainly Hebbian Learning
"""

import torch

from pylissom.nn.modules import register_recursive_input_output_hook
from pylissom.math import normalize
from pylissom.nn.functional.functions import kill_neurons, linear_decay
from pylissom.nn.functional.weights import apply_circular_mask_to_weights
from pylissom.nn.modules.lissom import Cortex

# This is necessary for docs inter-sphinx to work
torch.optim.Optimizer.__module__ = 'torch.optim'


class CortexOptimizer(torch.optim.Optimizer):
    """Abstract py:class:`Optimizer` that can only be used with py:class:`Cortex`"""
    def __init__(self, cortex):
        assert isinstance(cortex, Cortex)
        self.cortex = cortex
        super(CortexOptimizer, self).__init__(cortex.parameters(), {})


class SequentialOptimizer(object):
    r"""Similar to :py:class:`torch.nn.Sequential` but for optimizers, used to contain
    :py:class:`pylissom.optim.optimizers.CortexHebbian` for ReducedLissom modules"""
    # TODO: inherit from torch.optim.Optimizer
    def __init__(self, *optimizers):
        self.optimizers = optimizers

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()


class CortexHebbian(CortexOptimizer):
    r"""Implements hebbian learning over a py:class:`pylissom.nn.modules.Cortex` weights

    The formula is as follows:

    ..math::

        \begin{equation*}
        \text{w\'}_pq,ij = \text{w\'}_pq,ij + \alpha X_pq n_ij
        \end{equation*}

    Parameters:
        - **cortex** - :py:class:`pylissom.nn.modules.Cortex` map to apply formula
        - **learning_rate** -
    """
    # TODO:  Should use batch hebbian learning:
    # TODO: https://pdfs.semanticscholar.org/presentation/0fdc/eaea9ce40ea82711051d770714d1b0c7c17f.pdf
    # TODO: http://www.gatsby.ucl.ac.uk/~remo/TN1/ch8.pdf
    # TODO: http://elderlab.yorku.ca/~elder/teaching/psych6256/lectures/09%20Hebbian%20Learning.pdf
    def __init__(self, cortex, learning_rate):
        super(CortexHebbian, self).__init__(cortex)
        self.learning_rate = learning_rate
        # This adds a hook so the cortex saves the input and output activation in memory
        self.handles = register_recursive_input_output_hook(cortex)

    def step(self, **kwargs):
        self._hebbian_learning(self.cortex.weight, self.cortex.input, self.cortex.output,
                               self.learning_rate, self.cortex.radius)

    @staticmethod
    def _hebbian_learning(weights, input, output, learning_rate, radius):
        # Calculates the hebbian delta, applies the connective radius mask and updates the weights, normalizing them

        # Weight adaptation of a single neuron
        # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)

        delta = learning_rate * torch.mm(input.data.t(), output.data)
        apply_circular_mask_to_weights(delta.t_(), radius)
        weights.data.add_(delta.t_())
        weights.data = normalize(weights.data, norm=1, axis=0)
        return


class CortexPruner(CortexOptimizer):
    r"""Abstract class that prunes the weights in each step, subclasses must implement
    :py:func:`pylissom.optim.optimizers.CortexPruner._prune`

    Parameters:
        - **cortex** - :py:class:`pylissom.nn.modules.Cortex` map to apply formula
        - **pruning_step** -
    """
    def __init__(self, cortex, pruning_step=2000):
        super(CortexPruner, self).__init__(cortex)
        self.pruning_step = pruning_step
        self.step_counter = 1

    def step(self, **kwargs):
        if self.step_counter % self.pruning_step == 0:
            self._prune()
        self.step_counter += 1

    def _prune(self):
        raise NotImplementedError


class ConnectionDeath(CortexPruner):
    r"""Prunes the weights that are less than a threshold

    Parameters:
        - **cortex** - :py:class:`pylissom.nn.modules.Cortex` map to apply formula
        - **pruning_step** -
        - **connection_death_threshold** -
    """
    def __init__(self, cortex, pruning_step=2000, connection_death_threshold=1.0 / 400):
        super(ConnectionDeath, self).__init__(cortex, pruning_step)
        self.connection_death_threshold = connection_death_threshold

    def _prune(self):
        map(lambda w: kill_neurons(w, self.connection_death_threshold),
            [self.cortex.excitatory_weights, self.cortex.inhibitory_weights])


class NeighborsDecay(CortexPruner):
    r"""Reduces the connective radius of each neuron

    Parameters:
        - **cortex** - :py:class:`pylissom.nn.modules.Cortex` map to apply formula
        - **pruning_step** -
        - **decay_fn** - Default = linear_decay
        - **final_epoch** - necessary for the linear function
    """
    def __init__(self, cortex, pruning_step=2000, decay_fn=linear_decay, final_epoch=8.0):
        super(NeighborsDecay, self).__init__(cortex, pruning_step)
        self.decay_fn = decay_fn
        self.final_epoch = final_epoch

    def _prune(self):
        self.decay_fn(self.cortex.excitatory_weights, self.cortex.excitatory_radius,
                      epoch=self.cortex.epoch, final_epoch=self.final_epoch)
