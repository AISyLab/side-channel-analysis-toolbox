import unittest
import os.path
import numpy as np
from sca.__main__ import CONST_DEFAULT_PLAIN_FILE
from sca.__main__ import CONST_DEFAULT_TRACES_FILE
from sca.__main__ import CONST_DEFAULT_KEYS_FILE

from sca.analysis.pearson import Pearson


class TestPearsonIntegration(unittest.TestCase):
    """Tests the whole Pearson class."""

    def test_pearson(self):
        """"Test if pearson creates the correct file when ran"""
        Pearson.run(np.load(CONST_DEFAULT_TRACES_FILE)[:10000],
                    np.load(CONST_DEFAULT_KEYS_FILE),
                    np.load(CONST_DEFAULT_PLAIN_FILE), 5, False, 0, 1, 0)

        self.assertTrue(os.path.isfile('out/pearson_correlation_selected_indices.npy'))

    def test_pearson_save_trace(self):
        """"Test if pearson creates the correct file when ran"""
        Pearson.run(np.load(CONST_DEFAULT_TRACES_FILE)[:10000],
                    np.load(CONST_DEFAULT_KEYS_FILE),
                    np.load(CONST_DEFAULT_PLAIN_FILE), 5, True, 0, 1, 0)

        self.assertTrue(os.path.isfile('out/pearson_correlation_selected_traces.npy'))
