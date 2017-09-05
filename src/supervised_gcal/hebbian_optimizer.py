import torch
from src.supervised_gcal.cortex_layer import LissomCortexLayer
from src.supervised_gcal.utils import linear_neighbors_decay, kill_neurons


class LissomHebbianOptimizer(object):
    def update_weights(self, lissom_layer, step):
        assert isinstance(lissom_layer, LissomCortexLayer)
        hebbian_learning(lissom_layer.retina_weights, lissom_layer.retina,
                         lissom_layer.previous_activations,
                         self.learning_rate)

        hebbian_learning(lissom_layer.excitatory_weights, lissom_layer.previous_activations,
                         lissom_layer.previous_activations, self.learning_rate, sum=True)

        hebbian_learning(lissom_layer.inhibitory_weights, lissom_layer.previous_activations,
                         lissom_layer.previous_activations, self.learning_rate, sum=True)

        if self.pruning_step == step:
            if self.connection_death_threshold is not None:
                map(lambda w: kill_neurons(w, self.connection_death_threshold),
                    [lissom_layer.excitatory_weights, lissom_layer.inhibitory_weights])

            if self.neighbors_decay_fn is not None:
                self.neighbors_decay_fn(lissom_layer.excitatory_weights, lissom_layer.excitatory_radius)
        return

    def __init__(self, learning_rate=0.005, pruning_step=2000,
                 connection_death_threshold=1 / 400, neighbors_decay_fn=linear_neighbors_decay):
        self.neighbors_decay_fn = neighbors_decay_fn
        assert pruning_step is None or connection_death_threshold is not None or neighbors_decay_fn is not None
        self.connection_death_threshold = connection_death_threshold
        self.pruning_step = pruning_step
        self.learning_rate = learning_rate


def hebbian_learning(weights, input, output, learning_rate, sum=False):
    # Weight adaptation of a single neuron
    # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
    zero_mask = torch.gt(weights.data, 0).float()

    delta = learning_rate * torch.matmul(torch.t(input), output)
    hebbian = weights.data + delta
    zero_update = torch.mul(hebbian, zero_mask)
    if sum:
        den = torch.norm(zero_update, p=1, dim=0)
    else:
        # L2 without input image normalization is garbage
        # In spite of being used correctly, the activations are too low
        den = torch.norm(zero_update, p=2, dim=0)
    normalization = torch.div(zero_update, den)
    weights.data = normalization
    return



