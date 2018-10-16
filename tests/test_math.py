import pytest
from sklearn.preprocessing import normalize as sklearn_normalize
import torch

from pylissom.math import euclidian_distances, euclidean_distance_general, gaussian, normalize

@pytest.fixture(params=[(3, 3)])
def two_dim_grid_indices(request):
    return np.indices(request.param)


class TestMath(object):
    """
    Test class for the math module
    """

    @pytest.mark.parametrize("x, y, mu_x, mu_y, expected", [
        (1, 1, 1, 1, 0),
        (1, 0, 0, 0, 1),
        (1, 2, 1, -2, 4),
        (0, 0, 1, 2, np.sqrt(5)),
    ])
    def test_euclidean(self, x, y, mu_x, mu_y, expected):
        assert euclidian_distances(x, y, mu_x, mu_y) == expected

    @pytest.mark.parametrize("x_vector, y_vector, expected", [
        (np.array([1, 1]), np.array([1, 1]), 0),
        (np.array([1, 0]), np.array([0, 0]), 1),
        (np.array([1, 2]), np.array([1, -2]), 4),
        (np.array([0, 0]), np.array([1, 2]), np.sqrt(5)),
    ])
    def test_euclidean_general(self, x_vector, y_vector, expected):
        assert euclidean_distance_general(x_vector, y_vector) == expected

    @pytest.mark.parametrize("mu_x, mu_y, sigma, sigma_y, expected", [
        (0, 0, 1, 1,
         np.array([[1., 0.60653067, 0.13533528],
                   [0.60653067, 0.36787945, 0.082085],
                   [0.13533528, 0.082085, 0.01831564]])
         ),
        (1, 1, 1, 1,
         np.array([[0.36787945, 0.60653067, 0.36787945],
                   [0.60653067, 1., 0.60653067],
                   [0.36787945, 0.60653067, 0.36787945]])
         ),
        (1, 1, 2, 2,
         np.array([[0.77880079, 0.88249689, 0.77880079],
                   [0.88249689, 1., 0.88249689],
                   [0.77880079, 0.88249689, 0.77880079]])
         ),
        (1, 1, 5, 5,
         np.array([[0.96078944, 0.98019868, 0.96078944],
                   [0.98019868, 1., 0.98019868],
                   [0.96078944, 0.98019868, 0.96078944]])
         ),
        (1, 1, 1, 5,
         np.array([[0.59452057, 0.60653067, 0.59452057],
                   [0.98019868, 1., 0.98019868],
                   [0.59452057, 0.60653067, 0.59452057]])
         ),
        (1, 1, 5, 1,
         np.array([[0.59452057, 0.98019868, 0.59452057],
                   [0.60653067, 1., 0.60653067],
                   [0.59452057, 0.98019868, 0.59452057]])
         ),
    ])
    def test_gaussian(self, two_dim_grid_indices, mu_x, mu_y, sigma, sigma_y, expected):
        x, y = two_dim_grid_indices
        assert np.allclose(gaussian(x, y, mu_x, mu_y, sigma, sigma_y), expected, atol=1e-05)

    @pytest.mark.parametrize("matrix, norm, axis", [
        (torch.Tensor([[1, 1, 1], [1, 1, 1]]), 1, 1),
        (torch.Tensor([[1, 1, 1], [1, 1, 1]]), 1, 0),
        (torch.Tensor([[1, 1, 1], [3, 3, 3]]), 1, 0),
        (torch.Tensor([[1, 2, 3], [1, 2, 3]]), 1, 1),
        (torch.Tensor([[1, 1, 1], [1, 1, 1]]), 2, 1),
        (torch.Tensor([[1, 1, 1], [1, 1, 1]]), 2, 0),
        (torch.Tensor([[1, 1, 1], [3, 3, 3]]), 2, 0),
        (torch.Tensor([[1, 2, 3], [1, 2, 3]]), 2, 1),
    ])
    def test_normalize(self, matrix, norm, axis):
        if norm > 2:
            raise ValueError
        str_norm = 'l1' if norm == 1 else 'l2' if norm == 2 else 'max'
        np.allclose(normalize(matrix, norm, axis).numpy(), sklearn_normalize(matrix.numpy(), str_norm, axis))
