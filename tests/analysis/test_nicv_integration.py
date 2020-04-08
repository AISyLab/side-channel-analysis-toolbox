import unittest
import numpy as np
from sca.analysis.nicv import NICV
from sca.util.file_loader import FileLoader
from sca.__main__ import CONST_DEFAULT_PLAIN_FILE
from sca.__main__ import CONST_DEFAULT_TRACES_FILE
from sca.__main__ import CONST_DEFAULT_KEYS_FILE


class TestNicvIntegration(unittest.TestCase):

    def test_solve_nicv(self):
        """Tests whether the nicv is calculated correctly"""

        traces = np.array([[1, 2, 3], [4, 5, 6],  [7, 0.4, 9], [2, 3, 12]])
        plain = np.array([[1], [2], [1], [2]])
        keys = plain
        resulting_nicvs = np.array([0.23045267, 0.24016342, 0.49382716])
        calculated_nicvs = NICV.calculate_nicv_array(plain, traces, 0, keys)
        self.assertTrue(np.allclose(calculated_nicvs, resulting_nicvs))

    def test_run_nicv(self):
        """"This tests if the most probable point is contained in the result set"""

        traces, keys, plain = FileLoader.main(CONST_DEFAULT_TRACES_FILE,
                                              CONST_DEFAULT_KEYS_FILE,
                                              CONST_DEFAULT_PLAIN_FILE)

        result = NICV.run(traces, keys, plain)

        self.assertTrue(1398 in result)

    def test_run_nicv_without_hamming_weight(self):
        """"This tests if a probable point is contained in the result set not using hamming weights"""

        traces, keys, plain = FileLoader.main(CONST_DEFAULT_TRACES_FILE,
                                              CONST_DEFAULT_KEYS_FILE,
                                              CONST_DEFAULT_PLAIN_FILE)

        result = NICV.run(traces, keys, plain, hamming_weight=False)
        self.assertTrue(1350 in result)
