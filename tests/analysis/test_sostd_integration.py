import unittest
import numpy as np
from sca.analysis.sostd import SOSTD


class TestSOSTIntegration(unittest.TestCase):

    def test_single_run(self):
        """Tests whether the selcted traces are returned properly"""

        traces = np.load('data/traces.npy')
        keys = np.load('data/key.npy')
        plain = np.load('data/plain.npy')
        correct_indices = [4254, 4257, 1400, 4429, 4919, 1399, 4918, 1398, 4917, 1397]

        indices = SOSTD.list_of_best_indices(traces, keys, plain, 10, 0)
        self.assertTrue(np.array_equal(indices, correct_indices))

    def test_single_run_sosd(self):
        """Tests whether the selcted traces are returned properly"""

        traces = np.load('data/traces.npy')[:1000]
        keys = np.load('data/key.npy')[:1000]
        plain = np.load('data/plain.npy')[:1000]
        correct_indices = [4371, 2401, 2402, 4878, 4370, 4877, 4882, 4369, 4881, 2723]

        indices = SOSTD.list_of_best_indices(traces, keys, plain, 10, 0, False)
        self.assertTrue(np.array_equal(indices, correct_indices))
