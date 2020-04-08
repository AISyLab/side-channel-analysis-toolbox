import unittest
import numpy as np
from sca.analysis.pearson import Pearson


class TestPearsonUnit(unittest.TestCase):
    """Unit tests the Pearson class."""

    def test_pearson_coefficients(self):
        """Test that the calculation of pearson coefficients works."""

        traces = np.array([[20, 68], [217, 241], [58, 153], [192, 21], [101, 141]])
        hamming_weights = np.array([4, 3, 2, 2, 5])

        pc = Pearson.pearson_coefficients(hamming_weights, 2, traces)

        self.assertTrue(np.array_equal(pc, [1, 0]))
