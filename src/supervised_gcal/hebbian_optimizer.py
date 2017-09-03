import torch
from src.supervised_gcal.cortex_layer import LissomCortexLayer


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
        # L2 o L1 para aferente??
        den = torch.norm(zero_update, p=1, dim=0)
    normalization = torch.div(zero_update, den)
    weights.data = normalization
    return


class LissomHebbianOptimizer(object):
    def update_weights(self, lissom_layer, simple_lissom):
        assert isinstance(lissom_layer, LissomCortexLayer)
        if simple_lissom:
            hebbian_learning(lissom_layer.retina_weights, lissom_layer.retina,
                             lissom_layer.previous_activations,
                             self.learning_rate)
        else:
            pass
            # NOT IMPLEMENTED
            # hebbian_learning(lissom_layer.on_weights, lissom_layer.on, lissom_layer.previous_activations,
            #                  self.learning_rate)
            #
            # hebbian_learning(lissom_layer.off_weights, lissom_layer.off, lissom_layer.previous_activations,
            #                  self.learning_rate)

        hebbian_learning(lissom_layer.excitatory_weights, lissom_layer.previous_activations,
                         lissom_layer.previous_activations, self.learning_rate, sum=True)

        hebbian_learning(lissom_layer.inhibitory_weights, lissom_layer.previous_activations,
                         lissom_layer.previous_activations, self.learning_rate, sum=True)

        return

    def __init__(self, learning_rate=0.005):
        self.learning_rate = learning_rate
