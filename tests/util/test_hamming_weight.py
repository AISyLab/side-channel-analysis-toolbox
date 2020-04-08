import numpy as np
import unittest

from sca.util.hamming_weight import HammingWeight


class TestHammingWeight(unittest.TestCase):
    """
    Tests the HammingWeight class. Some functions do not have to be tested
    because they're tested via the general hamming_weights function.
    """

    def test_calculate_hamming_weights(self):
        """Test that the calculated hamming weights are correct."""

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

        hamming_weights = HammingWeight.hamming_weights(plaintext, key, 0)

        self.assertTrue(np.array_equal(hamming_weights, [5, 4, 2, 3, 5]))

    def test_nonstandard_aes(self):
        """Tests that the hamming weights for another round or operation are correct."""
        plaintext = np.array([[253, 175, 146, 139, 39, 24, 40, 196, 38, 225, 26, 31, 238, 15, 0, 30],
                              [16, 181, 1, 53, 76, 95, 58, 44, 17, 219, 19, 230, 141, 133, 68, 136],
                              [14, 44, 55, 24, 226, 225, 128, 116, 29, 129, 98, 180, 185, 188, 241, 254],
                              [162, 7, 17, 31, 86, 200, 98, 60, 166, 235, 184, 253, 211, 48, 91, 149],
                              [48, 0, 86, 35, 156, 15, 97, 135, 201, 46, 209, 49, 238, 12, 197, 127], ])
        key = np.array([[34, 123, 137, 185, 176, 241, 60, 235, 195, 90, 64, 204, 173, 62, 49, 13],
                        [243, 98, 136, 17, 128, 170, 12, 18, 143, 62, 83, 49, 244, 204, 247, 192],
                        [152, 77, 141, 53, 182, 175, 145, 242, 232, 203, 104, 91, 5, 174, 206, 126],
                        [5, 203, 149, 118, 67, 107, 181, 88, 186, 63, 9, 246, 52, 7, 144, 59],
                        [0, 236, 171, 201, 204, 216, 94, 50, 233, 124, 230, 23, 13, 100, 65, 150]])

        hamming_weights = HammingWeight.hamming_weights(plaintext, key, 0, 1, 1)

        self.assertTrue(np.array_equal(hamming_weights, [5, 2, 2, 4, 1]))

    def test_leakage_func(self):
        result = HammingWeight._hamming_leakage(np.array([0xFF, 0x3F, 0x2C, 0xD9]))

        print(result)
        self.assertTrue(np.array_equal(result, [8, 6, 3, 5]))
