import unittest
import numpy as np

from src.supervised_gcal.utils.math import *

class TestMath(unittest.TestCase):
    """
    Test class for the math module
    """

    def test_euclidean(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        self.assertEqual(euclidian_distances(1,1,1,1), 0)
        self.assertEqual(euclidian_distances(1,0,0,0), 1)
        self.assertEqual(euclidian_distances(1,2,1,-2), 4)
        self.assertEqual(euclidian_distances(0,0,1,2), np.sqrt(5))

    def test_euclidean_general(self):
        """
        The actual test.
        Any method which starts with ``test_`` will considered as a test case.
        """
        self.assertEqual(euclidean_distance_general(np.array([1,1]),np.array([1,1])), 0)
        self.assertEqual(euclidean_distance_general(np.array([1,0]),np.array([0,0])), 1)
        self.assertEqual(euclidean_distance_general(np.array([1,2]),np.array([1,-2])), 4)
        self.assertEqual(euclidean_distance_general(np.array([0,0]),np.array([1,2])), np.sqrt(5))

suite = unittest.TestLoader().loadTestsFromTestCase(TestMath)
unittest.TextTestRunner(verbosity=2).run(suite)