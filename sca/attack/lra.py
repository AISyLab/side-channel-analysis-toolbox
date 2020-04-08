import numpy as np
from sca.util.hamming_weight import HammingWeight
import progressbar
from sca.util import progress_bar_util
import warnings
from sca.util.data_partitioner import DataPartitioner


class LRA:
    """ This class contains methods for perform Linear Regression Analysis.
        for details on the algorithm see: https://eprint.iacr.org/2013/794.pdf """

    traces: np.ndarray
    plain: np.ndarray
    bar: progressbar
    number_of_traces: int
    dimension_leakage_points: int
    KEY_SIZE = 256

    def __init__(self, traces, plain):
        self.traces = traces
        self.plains = plain
        self.number_of_traces = len(self.traces)
        self.dimension_leakage_points = len(self.traces[0])

    @staticmethod
    def run(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, subkey: int, num_features: int,
            feature_select: int):

        """
        Runs Linear Regression Analysis

        :param traces: the traces to use
        :param keys: the keys to use
        :param plain: the plaintexts corresponding to the traces
        :param subkey: the specific subkey index to calculate.
                       16 is used to indicate that the whole key should be returned
        :param num_features: the number of features to select with feature selection
        :param feature_select: the type of feature selection to use.
        :return: the (sub)key most likely to be the real (sub)key
        """

        print('This performs Linear Regression Analysis')

        lra = LRA(traces, plain)

        lra.bar = progressbar.ProgressBar(max_value=2 * lra.KEY_SIZE, widgets=progress_bar_util.get_widgets(False))

        result = [0 for _ in range(16)]
        subkey_indices = [subkey]
        if subkey == 16:
            subkey_indices = list(range(16))
            lra.bar.max_value *= 16

        for i in subkey_indices:
            if feature_select != 0:
                feature_indices = DataPartitioner.select_features(traces, keys, plain, i, feature_select,
                                                                  num_features, 1, 0, True)
                lra.traces = traces[:, feature_indices]
                lra.dimension_leakage_points = len(lra.traces[0])

            result[i] = lra.solve_subkey(i)

        print('The final key is: ', result)

        return result

    def solve_subkey(self, subkey: int) -> int:

        """ Method that calculates the given subkey

        :param subkey: the specified subkey index to calculate
        :return the subkey calculated by the lra algorithm
        """

        warnings.simplefilter("ignore", category=RuntimeWarning)

        functions_per_byte_value_matrix = self.calculate_function_matrix(subkey)
        predictions_per_byte_value_matrix = self.calculate_prediction_matrix(functions_per_byte_value_matrix)
        coefficients_per_byte_value_matrix = \
            self.calculate_coefficients_of_determination(functions_per_byte_value_matrix,
                                                         predictions_per_byte_value_matrix)

        # Get the most likely subkey out of the coefficients matrix
        result = np.argmax(np.max(coefficients_per_byte_value_matrix, 1), 0)

        return int(result)

    def calculate_function_matrix(self, subkey: int):
        """
        Applies model functions to the traces and plains per possible subkey hypothesis byte
        current 10 model functions:
            functions m0 till m7: returns the i'th bith of the input
            hamming_weight m8: returns the hamming weights for every subkey hypothesis
            basis function m9: just returns 1 for every leakage

        :param: the subkey index to calculate
        :return: matrix containing model functions applied to each possible subkey and all traces
        """

        functions_per_byte_value_matrix = np.zeros((self.KEY_SIZE, self.number_of_traces, 10))

        for subkey_hypothesis in range(self.KEY_SIZE):
            plain = self.plains

            keys = np.array([[subkey_hypothesis] * 16] * self.number_of_traces)
            hamming_weight_func = HammingWeight.hamming_weights(plain, keys, subkey)

            matrix_m = np.ones((10, self.number_of_traces))
            matrix_m[9] = hamming_weight_func

            for i in range(8):
                for j in range(self.number_of_traces):
                    xor = plain[j][subkey] ^ subkey_hypothesis
                    matrix_m[i][j] = xor >> i & 1

            matrix_m = np.swapaxes(matrix_m, 0, 1)

            functions_per_byte_value_matrix[subkey_hypothesis] = matrix_m

            self.bar.update(self.bar.value + 1)

        return functions_per_byte_value_matrix

    def calculate_prediction_matrix(self, function_matrix: np.ndarray) -> np.ndarray:

        """ Method for calculating our prediction matrix given prediction functions

            :param function_matrix: matrix containing model functions applied to each possible subkey and al traces
            :return matrix containing prediction outputs for every subkey and every leakage point
        """
        predictions_per_byte_value_matrix = np.zeros((self.KEY_SIZE, 10, self.number_of_traces))

        for i in range(len(function_matrix)):
            subkey_hypothesis_matrix = function_matrix[i]
            bit_matrix_transposed = subkey_hypothesis_matrix.T
            pred = np.dot(np.linalg.inv(np.dot(bit_matrix_transposed, subkey_hypothesis_matrix)), bit_matrix_transposed)
            predictions_per_byte_value_matrix[i] = pred

        return predictions_per_byte_value_matrix

    def calculate_coefficients_of_determination(self, function_matrix: np.ndarray,
                                                prediction_matrix: np.ndarray) -> np.ndarray:
        """
        Method for testing the hypothesised subkey for every leakage coordinate and calculating a coefficient
        specifying a probability

        :param function_matrix: matrix containing function outputs for every subkey and every leakage point
        :param prediction_matrix: matrix containing prediction outputs for every subkey and every leakage point
        :return: matrix with coefficients per possible subkey per leakage point
        """

        coefficients_per_byte_value_matrix = np.zeros((self.KEY_SIZE, self.dimension_leakage_points))
        sst = LRA.sum_of_total_squares(self.traces)

        for subkey_hypothesis in range(self.KEY_SIZE):
            for feature_index in range(self.dimension_leakage_points):
                column = self.traces[:, feature_index]
                matrix_b = np.dot(prediction_matrix[subkey_hypothesis], column)
                estimator = np.dot(function_matrix[subkey_hypothesis], matrix_b)

                ssr = 0
                for i in range(self.number_of_traces):
                    ssr += estimator[feature_index] - self.traces[i][feature_index]

                coefficients_per_byte_value_matrix[subkey_hypothesis][feature_index] = 1.0 - (ssr / sst[feature_index])

            if self.bar.value < self.bar.max_value:
                self.bar.update(self.bar.value + 1)

        return coefficients_per_byte_value_matrix

    @staticmethod
    def sum_of_total_squares(traces: np.ndarray) -> np.ndarray:
        """
        Method that calculates the sum of total squares for a given set of traces per feature

        :param traces: the traces to run this operation on
        :return the sum of total squares for this set of traces
        """

        dimension_of_trace = len(traces[0])
        sst = np.zeros(dimension_of_trace)

        for i in range(dimension_of_trace):
            column = traces[:, i]
            mu = np.mean(column)
            sst[i] = np.sum(np.square(column - mu))

        return sst
