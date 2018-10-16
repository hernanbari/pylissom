import pytest
import numpy as np

from pylissom.math import gaussian, euclidian_distances
from pylissom.nn.functional.weights import apply_fn_to_weights_between_maps, circular_mask


class TestWeights(object):
    """
    Test class for the weights module
    """

    @pytest.mark.parametrize("in_features, out_features, sigma", [
        (9, 9, 1),
        (4, 4, 1),
        (9, 9, 5),
        (4, 4, 5),
    ])
    def test_apply_fn_to_weights_between_maps_gaussian(self, in_features, out_features, sigma):
        self.apply_fn(in_features, out_features,
                      gaussian, lambda x, y, mu_x, mu_y: gaussian(x, y, mu_x, mu_y, sigma, sigma), sigma=sigma)

    @pytest.mark.parametrize("in_features, out_features", [
        (9, 9),
        (4, 4),
        (16, 16),
    ])
    def test_apply_fn_to_weights_between_maps_euclidian(self, in_features, out_features):
        self.apply_fn(in_features, out_features, euclidian_distances, euclidian_distances)

    def apply_fn(self, in_features, out_features, fn, partial_fn, **kwargs):
        ans = apply_fn_to_weights_between_maps(in_features, out_features, fn, **kwargs)
        sqrt = int(in_features ** 0.5)
        for idx, r in enumerate(ans):
            reshaped = r.reshape(sqrt, sqrt)
            mu_x = idx // sqrt
            mu_y = idx % sqrt
            assert np.allclose(reshaped, partial_fn(*np.indices([sqrt, sqrt]), mu_x, mu_y))

    @pytest.mark.parametrize("in_features, out_features, radius", [
        (9, 9, 1),
        (4, 4, 1),
        (9, 9, 5),
        (4, 4, 5),
        (9, 9, 0),
        (4, 4, 0),
    ])
    def test_circular_mask(self, in_features, out_features, radius):
        distances = apply_fn_to_weights_between_maps(in_features=in_features, out_features=out_features,
                                                     fn=euclidian_distances)
        mask = distances > radius
        assert np.allclose(circular_mask(in_features, out_features, radius).numpy(), mask)
