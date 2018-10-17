import pytest

from pylissom.nn.functional.functions import linear_decay, kill_neurons, piecewise_sigmoid


class TestFunctions(object):
    """
    Test class for the functions module
    """

    def test_linear_decay(self):
        pass

    def test_kill_neurons(self):
        pass

    @pytest.mark.parametrize("min_theta, max_theta", [
        (0, 1),
        (0, 0),
        (1, 1),
        (0, 0.5),
        (0.5, 1),
        (1.5, 10),
    ])
    def test_piecewise_sigmoid(self, gaussian_tensor, min_theta, max_theta):
        out = piecewise_sigmoid(min_theta, max_theta, gaussian_tensor).numpy()
        for idx, x in enumerate(gaussian_tensor.numpy()):
            if x <= min_theta:
                assert out[idx] == 0
            elif x >= max_theta:
                assert out[idx] == 1
            else:
                assert out[idx] == (x-min_theta)/(max_theta-min_theta)
