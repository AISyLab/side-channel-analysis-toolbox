import unittest
import numpy as np
from sca.attack.mia import Mia
from sca.util.file_loader import FileLoader
from sca.__main__ import CONST_DEFAULT_PLAIN_FILE
from sca.__main__ import CONST_DEFAULT_TRACES_FILE
from sca.__main__ import CONST_DEFAULT_KEYS_FILE
from sca.util.data_partitioner import DataPartitioner


class TestMiaIntegration(unittest.TestCase):

    def test_mia_run(self):
        """"This tests the run method of MIA with a reduced data set"""
        traces, keys, plain = FileLoader.main(CONST_DEFAULT_TRACES_FILE,
                                              CONST_DEFAULT_KEYS_FILE,
                                              CONST_DEFAULT_PLAIN_FILE)

        profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext = \
            DataPartitioner.get_traces(traces, keys, plain, 1000, 0,
                                       0, 1000, 10, 1, 0, False)

        expected = np.array([43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.array_equal(Mia.run(profiling_traces, profiling_keys, profiling_plaintext, attack_traces,
                        attack_keys, attack_plaintext, 1, 0, 0, 1, 10, True), expected))
