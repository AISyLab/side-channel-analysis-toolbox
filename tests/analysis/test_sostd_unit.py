import unittest
import numpy as np
from sca.analysis.sostd import SOSTD


class TestSOSTUnit(unittest.TestCase):

    def test_list_of_indices_per_value(self):
        """ Tests list_of_indices_per_value for a simple case"""

        values = np.array([1, 2, 1, 1, 1, 1])
        traces = values
        correct_list = [[], [0, 2, 3, 4, 5], [1], [], [], [], [], [], []]
        result_list = SOSTD.list_of_indices_per_value(values, traces, True)
        self.assertTrue(np.array_equal(result_list, correct_list))
