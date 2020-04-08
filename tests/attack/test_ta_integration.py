import unittest
import numpy as np

from sca.attack import ta


class TestTaIntegration(unittest.TestCase):
    TRACES = np.load('data/traces.npy')
    PLAIN = np.load('data/plain.npy')
    KEY = np.load('data/key.npy')

    @unittest.skip
    def test_template_attack(self):
        """Test if template attack works correctly for the whole key"""

        expected = np.array([43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60])
        result = ta.TA.run(TestTaIntegration.TRACES[:9500], TestTaIntegration.KEY[:9500],
                           TestTaIntegration.PLAIN[:9500], TestTaIntegration.TRACES[9500:10000],
                           TestTaIntegration.KEY[9500:10000], TestTaIntegration.PLAIN[9500:10000],
                           False, 5, 1, 16, debug_mode_enabled=False)

        self.assertTrue(np.array_equal(expected, result))

    @unittest.skip
    def test_pooled_template_attack(self):
        """Test if pooled template attack works correctly for the whole key"""

        expected = np.array([43, 126, 21, 22, 40, 174, 210, 166, 171, 247, 21, 136, 9, 207, 79, 60])
        result = ta.TA.run(TestTaIntegration.TRACES, TestTaIntegration.KEY,
                           TestTaIntegration.PLAIN, TestTaIntegration.TRACES[8000:10000],
                           TestTaIntegration.KEY[8000:10000], TestTaIntegration.PLAIN[8000:10000],
                           True, 5, 1, 16, debug_mode_enabled=False)

        self.assertTrue(np.array_equal(expected, result))
