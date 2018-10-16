import pytest

from pylissom.math import *


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
