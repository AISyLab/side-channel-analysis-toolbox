import unittest
import numpy as np
from sca.analysis import nicv


class TestNicvUnit(unittest.TestCase):

    def test_calculate_mean_x_given_y_matrix(self):
        """ Tests whether the calculations of means work properly"""

        traces = np.array([[1, 2, 3], [4, 5, 6], [7, 0.4, 9], [2, 3, 12]])
        plain = np.array([[1], [2], [1], [2]])
        keys = plain
        resulting_matrix = np.zeros((9, 3))
        resulting_matrix[4] = [3.5, 2.6, 7.5]

        calculated_matrix = nicv.NICV.calculate_mean_x_given_y_matrix(plain, traces, 0, keys)

        print(calculated_matrix)

        self.assertTrue(np.allclose(calculated_matrix, resulting_matrix))

    def test_calculate_single_nicv(self):
        """ Tests whether the calculation of a single nicv value works properly"""

        mean_x_given_y = np.array([[-0.01, 0.01, 0, 0.014]])
        y = np.array([[0.1, -0.01, 0.03, 0.1]])

        resulting_nicv = 0.03898876404494381
        calculated_nicv = nicv.NICV.calculate_single_nicv(mean_x_given_y, y)

        self.assertAlmostEqual(calculated_nicv, resulting_nicv)

    def test_get_points_of_interest_indices(self):
        """ Tests if the point of interest selection works properly"""

        traces = np.array([[1, 2, 3], [4, 5, 6], [7, 0.4, 9], [2, 3, 12]])
        plain = np.array([[1], [2], [1], [2]])
        keys = plain

        resulting_points_of_interest_indices = [1, 2]
        calculated_points_of_interest_indices = nicv.NICV.get_points_of_interest_indices(plain, traces, 2, 0, keys)
        print(calculated_points_of_interest_indices)

        self.assertTrue(np.allclose(resulting_points_of_interest_indices, calculated_points_of_interest_indices))

    def test_get_points_of_interest(self):
        """ Tests if the point of interest selection works properly"""

        traces = np.array([[1, 2, 3], [4, 5, 6], [7, 0.4, 9], [2, 3, 12]])
        plain = np.array([[1], [2], [1], [2]])
        keys = plain

        resulting_points_of_interest = [[2, 3], [5, 6], [0.4, 9], [3, 12]]
        calculated_points_of_interest = nicv.NICV.get_points_of_interest(plain, traces, 2, 0, keys)

        self.assertTrue(np.allclose(resulting_points_of_interest, calculated_points_of_interest))
