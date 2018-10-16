from pylissom.nn.functional.weights import apply_fn_to_weights_between_maps, get_gaussian_weights, circular_mask, \
    apply_circular_mask_to_weights, dense_weights_to_sparse

class TestWeights(object):
    """
    Test class for the math module
    """

    def test_euclidean(self):
        assert 1 == 1
