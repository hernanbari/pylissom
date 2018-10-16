import unittest

from pylissom.nn.modules.linear import GaussianLinear


class TestMath(unittest.TestCase):
    """
    Test class for the math module
    """

    def test_euclidean(self):
        assert 1 == 1

suite = unittest.TestLoader().loadTestsFromTestCase(TestMath)
unittest.TextTestRunner(verbosity=2).run(suite)
