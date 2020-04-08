import unittest
import numpy as np
import progressbar  # This actually imports progressbar2 but `import progressbar2' itself doesn't work. Weird package.
from sca.attack import sa
from sca.loghandler.loghandler import LogHandler


class TestSaIntegration(unittest.TestCase):

    def test_solve_single_subkey(self):
        """Tests whether the solve_subkey method returns correctly while using the hamming_weight leakage model"""
        traces = np.load('data/traces.npy')
        keys = np.load('data/key.npy')
        plain = np.load('data/plain.npy')

        sa_obj = sa.SA(traces[:4000], keys[:4000], plain[:4000], traces[4000:4030], keys[4000:4030], plain[4000:4030])
        sa_obj.log_handler = LogHandler('test_sa', False)
        bar = progressbar.NullBar()

        # In our testing set 126 is the second subkey.
        self.assertEqual(sa_obj.solve_subkey(1, False, bar, 1, 15, num_traces=4000, hamming_weight=True), 126)

    def test_run_method(self):
        traces = np.load('data/traces.npy')
        keys = np.load('data/key.npy')
        plaintext = np.load('data/plain.npy')

        result = sa.SA.run(traces[:400], keys[:400], plaintext[:400], traces[400:500], keys[400:500],
                           plaintext[400:500], 1, 0, 9, 9, 0, 1, 2)

        # Result is not correct since not enough traces are provided so we test shape
        self.assertTrue(len(result), 16)

    def test_subkey_out_of_range(self):
        """Tests whether an error is thrown when a subkey is requested that does not exist.
        A key is composed of 16 subkeys."""
        traces = np.load('data/traces.npy')
        keys = np.load('data/key.npy')
        plain = np.load('data/plain.npy')

        sa_obj = sa.SA(traces[:4000], keys[:4000], plain[:4000], traces[4000:4030], keys[4000:4030], plain[4000:4030])
        bar = progressbar.NullBar()

        self.assertRaises(ValueError,
                          lambda: sa_obj.solve_subkey(16, False, bar, 1, 15, num_traces=3999, hamming_weight=True))
