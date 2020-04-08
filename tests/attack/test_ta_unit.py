import unittest
from sca.attack import ta
import numpy as np


class TestTaUnit(unittest.TestCase):
    TRACES = np.load('data/traces.npy')
    PLAIN = np.load('data/plain.npy')
    KEY = np.load('data/key.npy')

    def test_subkey_template_attack(self):
        """Test if template attack works correctly for a single subkey"""

        subkey_result = np.array([43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.assertTrue(np.alltrue(
            subkey_result == ta.TA.run(TestTaUnit.TRACES[:9000], TestTaUnit.KEY[:9000], TestTaUnit.PLAIN[:9000],
                                       TestTaUnit.TRACES[9000:10000], TestTaUnit.KEY[9000:10000],
                                       TestTaUnit.PLAIN[9000:10000], False, 5, 5, 0, debug_mode_enabled=False)))

    def test_subkey_pooled_template_attack(self):
        """Test if pooled template attack works correctly for a single subkey"""

        expected_result = np.array([43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        calculated_subkey = ta.TA.run(TestTaUnit.TRACES[:9000], TestTaUnit.KEY[:9000], TestTaUnit.PLAIN[:9000],
                                      TestTaUnit.TRACES[9000:10000], TestTaUnit.KEY[9000:10000],
                                      TestTaUnit.PLAIN[9000:10000], True, 5, 5, 0, debug_mode_enabled=False)

        self.assertTrue(np.alltrue(expected_result == calculated_subkey))

    def test_find_points_of_interest(self):
        """Test if the find points of interest sub method finds the correct points of interest"""

        result = ta.TA.find_points_of_interest(2, 2, np.array([0.1, 2000, 0.3]))

        self.assertTrue(np.alltrue(np.array([1, 0]) == result))

    def test_init_makes_object(self):
        empty = np.empty(0)
        obj = ta.TA(empty, empty, empty, empty, empty, empty)

        self.assertIsNone(obj.log_handler)
        self.assertEqual(obj.hw, [bin(x).count("1") for x in range(256)])

    def test_calc_hw_and_average(self):
        empty = np.empty(0)
        plaintext = np.array([[253, 175, 146, 139, 39, 24, 40, 196, 38, 225, 26, 31, 238, 15, 0, 30],
                              [162, 7, 17, 31, 86, 200, 98, 60, 166, 235, 184, 253, 211, 48, 91, 149],
                              [16, 181, 1, 53, 76, 95, 58, 44, 17, 219, 19, 230, 141, 133, 68, 136],
                              [48, 0, 86, 35, 156, 15, 97, 135, 201, 46, 209, 49, 238, 12, 197, 127],
                              [14, 44, 55, 24, 226, 225, 128, 116, 29, 129, 98, 180, 185, 188, 241, 254], ])
        key = np.array([[34, 123, 137, 185, 176, 241, 60, 235, 195, 90, 64, 204, 173, 62, 49, 13],
                        [5, 203, 149, 118, 67, 107, 181, 88, 186, 63, 9, 246, 52, 7, 144, 59],
                        [243, 98, 136, 17, 128, 170, 12, 18, 143, 62, 83, 49, 244, 204, 247, 192],
                        [152, 77, 141, 53, 182, 175, 145, 242, 232, 203, 104, 91, 5, 174, 206, 126],
                        [0, 236, 171, 201, 204, 216, 94, 50, 233, 124, 230, 23, 13, 100, 65, 150], ])
        traces = np.array([[253, 175, 146, 139, 39, 24, 40, 196, 38, 225, 26, 31, 238, 15, 0, 30],
                           [16, 181, 1, 53, 76, 95, 58, 44, 17, 219, 19, 230, 141, 133, 68, 136],
                           [14, 44, 55, 24, 226, 225, 128, 116, 29, 129, 98, 180, 185, 188, 241, 254],
                           [162, 7, 17, 31, 86, 200, 98, 60, 166, 235, 184, 253, 211, 48, 91, 149],
                           [48, 0, 86, 35, 156, 15, 97, 135, 201, 46, 209, 49, 238, 12, 197, 127], ])

        ta_obj = ta.TA(traces, key, plaintext, empty, empty, empty)

        result = ta_obj.calc_traces_hamming_weight(5, True, 0)
        expected2 = [[16, 181, 1, 53, 76, 95, 58, 44, 17, 219, 19, 230, 141, 133, 68, 136, ],
                     [48, 0, 86, 35, 156, 15, 97, 135, 201, 46, 209, 49, 238, 12, 197, 127, ]]
        expected3 = [[162, 7, 17, 31, 86, 200, 98, 60, 166, 235, 184, 253, 211, 48, 91, 149, ]]
        expected4 = [[253, 175, 146, 139, 39, 24, 40, 196, 38, 225, 26, 31, 238, 15, 0, 30, ]]
        expected5 = [[14, 44, 55, 24, 226, 225, 128, 116, 29, 129, 98, 180, 185, 188, 241, 254, ]]

        self.assertTrue(result[0].size == 0)
        self.assertTrue(result[1].size == 0)
        self.assertTrue(result[6].size == 0)
        self.assertTrue(result[7].size == 0)
        self.assertTrue(result[8].size == 0)

        self.assertTrue(np.array_equal(result[2], expected2))
        self.assertTrue(np.array_equal(result[3], expected3))
        self.assertTrue(np.array_equal(result[4], expected4))
        self.assertTrue(np.array_equal(result[5], expected5))

    def test_covariance_matrix(self):
        empty = np.empty(0)
        ta_obj = ta.TA(TestTaUnit.TRACES, TestTaUnit.KEY, TestTaUnit.PLAIN, empty, empty, empty)
        hamming_weights = ta_obj.calc_traces_hamming_weight(9, True)
        result = ta.TA.calc_covariance_matrix(2, hamming_weights, [4, 1])

        expected = [[[8.40362484e-06, 1.17626553e-05],
                     [1.17626553e-05, 2.42975247e-05]],
                    [[6.10700304e-06, 6.81138455e-06],
                     [6.81138455e-06, 1.67846296e-05]],
                    [[6.39532532e-06, 7.76576382e-06],
                     [7.76576382e-06, 1.93442021e-05]],
                    [[6.93599381e-06, 8.39899445e-06],
                     [8.39899445e-06, 2.02131000e-05]],
                    [[6.84149764e-06, 8.04314808e-06],
                     [8.04314808e-06, 1.94904910e-05]],
                    [[6.69222434e-06, 8.00577549e-06],
                     [8.00577549e-06, 1.98666777e-05]],
                    [[6.88548021e-06, 8.20945147e-06],
                     [8.20945147e-06, 2.00889529e-05]],
                    [[7.12873851e-06, 8.00295529e-06],
                     [8.00295529e-06, 2.04074348e-05]],
                    [[8.91241583e-06, 1.32275773e-05],
                     [1.32275773e-05, 2.99677119e-05]]]

        self.assertTrue(np.allclose(result, expected))
