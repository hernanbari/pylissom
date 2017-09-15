import torch
from src.supervised_gcal.cortex_layer import CortexLayer
from src.supervised_gcal.utils.functions import kill_neurons, linear_decay


class CortexOptimizer(torch.optim.Optimizer):
    def step(self, closure):
        raise NotImplementedError


class CombinedCortexOptimizer(CortexOptimizer):
    optimizers = []

    def step(self, **kwargs):
        for opt in self.optimizers:
            opt.step()


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
    def __init__(self, params, cortex_layer, learning_rate=0.005):
        self.cortex_layer = cortex_layer
        assert isinstance(self.cortex_layer, CortexLayer)
        self.learning_rate = learning_rate
        super().__init__(params, {'learning_rate': learning_rate})

    def step(self, **kwargs):
        self._hebbian_learning(self.cortex_layer.afferent_weights, self.cortex_layer.cortex_input,
                               self.cortex_layer.activation, self.learning_rate)

        self._hebbian_learning(self.cortex_layer.excitatory_weights, self.cortex_layer.activation,
                               self.cortex_layer.activation, self.learning_rate)

        self._hebbian_learning(self.cortex_layer.inhibitory_weights, self.cortex_layer.activation,
                               self.cortex_layer.activation, self.learning_rate)

    @staticmethod
    def _hebbian_learning(weights, input, output, learning_rate, sum=True):
        # Weight adaptation of a single neuron
        # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
        zero_mask = torch.gt(weights.data, 0).float()

        delta = learning_rate * torch.matmul(torch.t(input.data), output.data)
        weights.data.add_(delta)
        weights.data.mul_(zero_mask)
        if sum:
            den = torch.norm(weights.data, p=1, dim=0)
        else:
            # L2 without input image normalization is garbage
            # In spite of being used correctly, the activations are too low
            den = torch.norm(weights.data, p=2, dim=0)
        weights.data.div_(den)
        return


class CortexPruner(CortexOptimizer):
    def __init__(self, cortex_layer, pruning_step=2000):
        super().__init__(cortex_layer)
        self.pruning_step = pruning_step
        self.step_counter = 0

    def step(self, **kwargs):
        if self.pruning_step % self.step_counter == 0:
            self.prune()
        self.step_counter += 1

    def prune(self):
        raise NotImplementedError


class ConnectionDeath(CortexPruner):
    def __init__(self, pruning_step, connection_death_threshold=1.0 / 400):
        super().__init__(pruning_step)
        self.connection_death_threshold = connection_death_threshold

    def prune(self):
        map(lambda w: kill_neurons(w, self.connection_death_threshold),
            [self.cortex_layer.excitatory_weights, self.cortex_layer.inhibitory_weights])


class NeighborsDecay(CortexPruner):
    def __init__(self, pruning_step, decay_fn=linear_decay, final_epoch=8.0):
        super().__init__(pruning_step)
        self.decay_fn = decay_fn
        self.epoch = 1
        self.final_epoch = final_epoch

    def prune(self):
        self.decay_fn(self.cortex_layer.excitatory_weights, self.cortex_layer.excitatory_radius,
                      epoch=self.epoch, final_epoch=self.final_epoch)
        self.epoch += 1
