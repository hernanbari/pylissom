import torch
from src.supervised_gcal.modules.lissom import register_recursive_input_output_hook, Cortex
from src.supervised_gcal.utils.functions import kill_neurons, linear_decay
from src.supervised_gcal.utils.weights import apply_circular_mask_to_weights
from src.supervised_gcal.utils.math import normalize


class CortexOptimizer(torch.optim.Optimizer):
    def __init__(self, cortex):
        assert isinstance(cortex, Cortex)
        self.cortex = cortex
        super(CortexOptimizer, self).__init__(cortex.parameters(), {})


class SequentialOptimizer(object):
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
    def __init__(self, cortex, learning_rate):
        super(CortexHebbian, self).__init__(cortex)
        self.learning_rate = learning_rate
        self.handles = register_recursive_input_output_hook(cortex)

    def step(self, **kwargs):
        self.hebbian_learning(self.cortex.weight, self.cortex.input, self.cortex.output,
                              self.learning_rate, self.cortex.radius)

    @staticmethod
    def hebbian_learning(weights, input, output, learning_rate, radius):
        # Weight adaptation of a single neuron
        # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)

        delta = learning_rate * torch.mm(input.data.t(), output.data)
        apply_circular_mask_to_weights(delta.t_(), radius)
        weights.data.add_(delta.t_())
        weights.data = normalize(weights.data, norm=1, axis=0)
        return


class CortexPruner(CortexOptimizer):
    def __init__(self, cortex, pruning_step=2000):
        super(CortexPruner, self).__init__(cortex)
        self.pruning_step = pruning_step
        self.step_counter = 1

    def step(self, **kwargs):
        if self.step_counter % self.pruning_step == 0:
            self.prune()
        self.step_counter += 1

    def prune(self):
        raise NotImplementedError


class ConnectionDeath(CortexPruner):
    def __init__(self, cortex, pruning_step=2000, connection_death_threshold=1.0 / 400):
        super(ConnectionDeath, self).__init__(cortex, pruning_step)
        self.connection_death_threshold = connection_death_threshold

    def prune(self):
        map(lambda w: kill_neurons(w, self.connection_death_threshold),
            [self.cortex.excitatory_weights, self.cortex.inhibitory_weights])


class NeighborsDecay(CortexPruner):
    def __init__(self, cortex, pruning_step=2000, decay_fn=linear_decay, final_epoch=8.0):
        super(NeighborsDecay, self).__init__(cortex, pruning_step)
        self.decay_fn = decay_fn
        self.final_epoch = final_epoch

    def prune(self):
        self.decay_fn(self.cortex.excitatory_weights, self.cortex.excitatory_radius,
                      epoch=self.cortex.epoch, final_epoch=self.final_epoch)
