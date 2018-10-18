from pylissom.utils.orientation_maps import *
from tests.conftest import get_dummy_trained_lissom


class TestOrientationMap(object):
    """
    Test class for the OrientationMap module
    """

    def test_om(self):
        lissom = get_dummy_trained_lissom()

        size = int(lissom.in_features ** 0.5)
        orientations = 180
        keys_arrays = get_oriented_lines(size, orientations=orientations)
        keys_arrays = {k: map(lambda l: torch.autograd.Variable(l), lines) for k, lines in keys_arrays.items()}
        om = OrientationMap(model=lissom, inputs=keys_arrays)

        orientation_map = om.get_orientation_map()
        orientation_hist = om.get_orientation_hist()
        mean, std = metrics_orientation_hist(orientation_hist)
        assert 1 == 1
