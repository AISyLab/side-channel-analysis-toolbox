import unittest
import numpy as np
import progressbar

from sca.attack import sa


class TestSaUnit(unittest.TestCase):
    TRACES = np.array([[34, 123, 137, 185, 176, 241, 60, 235],
                       [5, 203, 149, 118, 67, 107, 181, 88],
                       [243, 98, 136, 17, 128, 170, 12, 18],
                       [152, 77, 141, 53, 182, 175, 145, 242],
                       [0, 236, 171, 201, 204, 216, 94, 50],
                       [243, 98, 136, 17, 128, 170, 12, 18],
                       [152, 77, 141, 53, 182, 175, 145, 242],
                       [34, 123, 137, 185, 176, 241, 60, 235],
                       [5, 203, 149, 118, 67, 107, 181, 88], ])
    KEYS = np.array([[34, 123, 137, 185, 176, 241, 60, 235, 195, 90, 64, 204, 173, 62, 49, 13],
                     [5, 203, 149, 118, 67, 107, 181, 88, 186, 63, 9, 246, 52, 7, 144, 59],
                     [243, 98, 136, 17, 128, 170, 12, 18, 143, 62, 83, 49, 244, 204, 247, 192],
                     [152, 77, 141, 53, 182, 175, 145, 242, 232, 203, 104, 91, 5, 174, 206, 126],
                     [0, 236, 171, 201, 204, 216, 94, 50, 233, 124, 230, 23, 13, 100, 65, 150], ])
    PLAIN = np.array([[253, 175, 146, 139, 39, 24, 40, 196, 38, 225, 26, 31, 238, 15, 0, 30],
                      [16, 181, 1, 53, 76, 95, 58, 44, 17, 219, 19, 230, 141, 133, 68, 136],
                      [14, 44, 55, 24, 226, 225, 128, 116, 29, 129, 98, 180, 185, 188, 241, 254],
                      [162, 7, 17, 31, 86, 200, 98, 60, 166, 235, 184, 253, 211, 48, 91, 149],
                      [48, 0, 86, 35, 156, 15, 97, 135, 201, 46, 209, 49, 238, 12, 197, 127], ])

    def test_matrix_calculations(self):
        """Tests whether the matrix calculations work properly"""

        # Identity matrix of 16x16. 16 is arbitrarily chosen
        bit_matrix = np.identity(16)

        # Vector or random numbers to simulate our measurements
        measurements = np.random.rand(16, 1) * 10

        # Due to identity matrix properties this calculation should return the measurements matrix
        self.assertTrue(np.alltrue(sa.SA.matrix_calculations(bit_matrix, measurements, False) == measurements))

    def test_poi_selection(self):
        """Tests whether our point of interest selection selects proper poi's"""

        # Simulate estimates matrix with a single obvious outlier.
        estimates = np.full((5000, 9), 0)
        estimates[25][1] = 25

        # Calculate points of interest from simulated estimates
        points_of_interest = sa.SA.poi_selection(estimates)

        # We should only get a single outlier at the chosen location
        self.assertEqual(points_of_interest[0], 25)

    def test_parameter_estimation(self):
        estimation = sa.SA.parameter_estimation(TestSaUnit.TRACES, np.identity(9), False, num_traces=9)

        self.assertTrue(np.array_equal(estimation, [[34, 5, 243, 152, 0, 243, 152, 34, 5, ],
                                                    [123, 203, 98, 77, 236, 98, 77, 123, 203, ],
                                                    [137, 149, 136, 141, 171, 136, 141, 137, 149, ],
                                                    [185, 118, 17, 53, 201, 17, 53, 185, 118, ],
                                                    [176, 67, 128, 182, 204, 128, 182, 176, 67, ],
                                                    [241, 107, 170, 175, 216, 170, 175, 241, 107, ],
                                                    [60, 181, 12, 145, 94, 12, 145, 60, 181, ],
                                                    [235, 88, 18, 242, 50, 18, 242, 235, 88, ], ]
                                       ))

    def test_poa_output(self):
        empty = np.empty(0)

        sa_obj = sa.SA(empty, TestSaUnit.KEYS, TestSaUnit.PLAIN, empty, empty, empty)
        result = sa_obj.poa_output(0, num_traces=5)
        expected = np.array([[1, 0, 1, 1, 1, 0, 1, 1, 1],
                             [1, 0, 1, 0, 1, 1, 0, 0, 1],
                             [1, 0, 1, 0, 1, 0, 1, 0, 0],
                             [1, 1, 0, 0, 0, 0, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 1, 0, 0], ])

        self.assertTrue(np.alltrue(result == expected))

    def test_key_extraction(self):
        empty = np.empty(0)
        estimates = np.array([[253, 175, 146, 139, 39, 24, 40, 196, 38],
                              [162, 7, 17, 31, 86, 200, 98, 60, 166],
                              [16, 181, 1, 53, 76, 95, 58, 44, 17],
                              [48, 0, 86, 35, 156, 15, 97, 135, 201],
                              [14, 44, 55, 24, 226, 225, 128, 116, 29],
                              [16, 181, 1, 53, 76, 95, 58, 44, 17],
                              [253, 175, 146, 139, 39, 24, 40, 196, 38],
                              [162, 7, 17, 31, 86, 200, 98, 60, 166], ])

        sa1 = sa.SA(empty, empty, empty, empty, TestSaUnit.KEYS, TestSaUnit.PLAIN)

        # bit_matrix is unused
        single_result = sa1.key_extraction(estimates, TestSaUnit.TRACES, 0, progressbar.NullBar(), num_attack_traces=5)
        three_results = sa1.key_extraction(estimates, TestSaUnit.TRACES, 0, progressbar.NullBar(), num_attack_traces=5,
                                           top_n_guesses=3)

        self.assertEqual(single_result, 66)
        self.assertTrue(np.array_equal(three_results, [66, 76, 124]))
