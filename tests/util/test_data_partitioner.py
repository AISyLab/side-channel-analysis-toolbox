import unittest
import numpy as np
from sca.util.data_partitioner import DataPartitioner


class TestDataPartitioner(unittest.TestCase):

    def test_basic_run(self):
        """ Tests a run with normal amount of traces to select."""

        traces = np.array([[1, 2, 3], [4, 5, 6],  [7, 0.4, 9], [2, 3, 12]])
        plain = np.array([[1], [2], [1], [2]])
        keys = plain

        profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
            DataPartitioner.get_traces(traces, keys, plain, 2, 1, 0, 1, 2, 1, 0, True)

        # Test whether the dimensions of the returned traces match the expectations.
        self.assertEqual(len(profiling_traces), 2)
        self.assertEqual(len(profiling_traces[0]), 3)

    def test_run_too_much_traces(self):
        """ Tests a run with more traces asked than provided."""

        traces = np.array([[1, 2, 3], [4, 5, 6], [7, 0.4, 9], [2, 3, 12]])
        plain = np.array([[1], [2], [1], [2]])
        keys = plain

        with self.assertRaises(SystemExit) as cm:
            DataPartitioner.get_traces(traces, keys, plain, 5, 1, 0, 1, 2, 1, 0, True)

        # Test whether the right exit code was used.
        self.assertEqual(cm.exception.code, 1)
