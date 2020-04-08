import unittest
import numpy as np
from sca.attack.mia import Mia


class TestMiaUnit(unittest.TestCase):

    def test_extract_key(self):
        """"This tests the correct functionality the key retrieval function.
        This function should give the key that corresponds to the highest mutual information"""

        fake_data = [(0.04, 32), (0.032, 12), (0.78, 42)]

        mia = Mia(np.load("data/traces.npy"),
                  np.load("data/key.npy"),
                  np.load("data/plain.npy"))

        self.assertEqual(mia.extract_key(fake_data), 42)

    def test_visualize_data(self):
        """"This tests if the data visualization method returns a string that contains the correct key occurrences."""

        fake_data = [(0.04, 32), (0.032, 12), (0.78, 42), (0.66, 42), (0.44, 12), (0.77, 42)]

        mia = Mia(np.load("data/traces.npy"),
                  np.load("data/key.npy"),
                  np.load("data/plain.npy"))

        tabulate = mia.tabulate_top_n(fake_data, 2)
        self.assertTrue("42" in tabulate)
        self.assertTrue("12" in tabulate)
        self.assertFalse("32" in tabulate)

    def test_get_value_at_time(self):
        """"This tests if the observation selection runs correctly on an existing time index"""

        traces = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])

        mia = Mia(traces, np.array([[12], [13]]), np.array([[14], [15]]))

        self.assertTrue(np.array_equal(mia.get_observations_at_time(1), np.array([3, 4])))

    def test_get_value_at_time_out_of_range(self):
        """"This tests if the observation selection runs correctly on a non-existing time index"""

        traces = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])

        mia = Mia(traces, np.array([[12], [13]]), np.array([[14], [15]]))

        self.assertTrue(np.array_equal(mia.get_observations_at_time(-1), np.array([])))
        self.assertTrue(np.array_equal(mia.get_observations_at_time(mia.NUMBER_OF_TRACES), np.array([])))
