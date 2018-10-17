import numpy as np
import pytest
import torch

from pylissom.nn.functional.weights import apply_circular_mask_to_weights
from pylissom.optim import CortexHebbian


@pytest.fixture(params=[0.1, 0.2])
def learning_rate(request):
    return request.param


class TestOptimizers(object):
    """
    Test class for the optim module
    """
    def test_cortex_hebbian(self, cortex, learning_rate, gaussian_variable):
        previous_weights = cortex.weight.data.numpy().copy()
        inp = gaussian_variable.data.numpy().copy()
        assert inp.T.shape[1] == 1
        cortex_hebbian = CortexHebbian(cortex, learning_rate)
        out = cortex(gaussian_variable)
        out = out.data.numpy().copy()
        assert np.allclose(previous_weights, cortex.weight.data.numpy())
        assert np.allclose(inp, cortex.input.data.numpy())
        cortex_hebbian.step()
        assert np.allclose(inp, cortex.input.data.numpy())
        hebbian_weights = self.hebbian_calculation(cortex, inp, learning_rate, out, previous_weights)
        assert np.allclose(cortex.weight.data.numpy(), hebbian_weights)

    def hebbian_calculation(self, cortex, inp, learning_rate, out, previous_weights):
        # w'_pq,ij = (w_pq,ij + alpha * input_pq * output_ij) / sum_uv (w_uv,ij + alpha * input_uv * output_ij)
        corr_matrix = np.matmul(inp.T, out)
        corr_matrix = apply_circular_mask_to_weights(torch.from_numpy(corr_matrix), cortex.radius).numpy()
        corr_matrix_norm = np.sum(corr_matrix, axis=0)
        previous_weights_norm = np.sum(previous_weights, axis=0)
        hebbian_weights = np.zeros(previous_weights.shape)
        for idx, x in np.ndenumerate(cortex.weight.data.numpy()):
            col_sum = previous_weights_norm[idx[1]]
            den_term = (col_sum + learning_rate * corr_matrix_norm[idx[1]])
            nom_term = (previous_weights[idx] + learning_rate * inp[0][idx[0]] * out[0][idx[1]])
            new_weight = nom_term / den_term
            hebbian_weights[idx] = new_weight
        hebbian_weights = apply_circular_mask_to_weights(torch.from_numpy(hebbian_weights), cortex.radius).numpy()
        return hebbian_weights

    def test_reduced_lissom_hebbian(self):
        pass

    def test_connection_death(self):
        pass

    def test_neighbors_decay(self):
        pass
