import numpy as np
from scipy.stats.stats import pearsonr
import warnings

from sca.util.hamming_weight import HammingWeight


class Pearson:
    """
    This class contains methods for performing Pearson correlation for feature
    selection. The indices of traces with the highest correlation coefficient
    can be retrieved using `Pearson.pearson_coefficients`. Use `Pearson.run`
    to save those to a file.
    """

    @staticmethod
    def run(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, points_of_interest: int, save_traces: bool,
            subkey: int, aes_round: int = 1, aes_operation: int = 0, leakage_model: bool = True):
        """
        This method performs the full Pearson correlation analysis.

        :param traces: the traces used for the tool
        :param keys: the keys used for the tool
        :param plain: the plain used for the tool
        :param points_of_interest: the number of points of interest to extract.
        :param save_traces: whether to save the full traces or just their indices.
        :param subkey: the subkey to correlate on. Used for hamming weight
            calculation, must be in the range [0-15].
        :param aes_round: the AES round to 'attack'.
        :param aes_operation: the AES operation to 'attack', represented as an integer from 0 to 3.
        :param leakage_model: the leakage model to use.
        """

        assert 0 <= subkey < 16, "Subkey must be in the range [0-15]"

        hamming_weights = HammingWeight.hamming_weights(plain, keys, subkey, aes_round, aes_operation, leakage_model)
        indices = Pearson.pearson_coefficients(hamming_weights, points_of_interest, traces)

        if save_traces:
            print('Saving interesting traces... ', end='')
            np.save('out/pearson_correlation_selected_traces', traces[:, indices])
        else:
            print('Saving indices of interesting traces... ', end='')
            np.save('out/pearson_correlation_selected_indices', indices)

        print('Done')

    @staticmethod
    def pearson_coefficients(hamming_weights: np.ndarray, points_of_interest: int, traces: np.ndarray):
        """Calculate the Pearson correlation coefficients and return the indices
        of those with the highest correlation.

        :param hamming_weights: the hamming weights of the plaintext combined
            with the key, can be calculated using `Pearson.hamming_weights`.
        :param points_of_interest: the number of points of interest (features)
            to extract from the coefficients.
        :param traces: the values to calculate correlation with hamming_weights.

        :returns: the indices of the traces with the highest Pearson correlation
            coefficient.
        """

        warnings.simplefilter("ignore", category=RuntimeWarning)

        pearson_coefficients = np.zeros(len(traces[0]))
        for i in range(len(traces[0])):
            pearson_coefficients[i] = abs(pearsonr(hamming_weights, traces[:, i])[0])

        warnings.simplefilter("default")

        return np.argsort(-pearson_coefficients)[:points_of_interest][::-1]

    @staticmethod
    def pearson_cpa(hamming_weights: np.ndarray, traces: np.ndarray):
        """Calculate the Pearson correlation coefficients and return the indices
        of those with the highest correlation.

        :param hamming_weights: the hamming weights of the plaintext combined
            with the key, can be calculated using `Pearson.hamming_weights`.
        :param traces: the values to calculate correlation with hamming_weights.

        :returns: the indices of the traces with the highest Pearson correlation
            coefficient.
        """

        warnings.simplefilter("ignore", category=RuntimeWarning)

        pearson_coefficients = np.zeros(len(traces[0]))
        for i in range(len(traces[0])):
            pearson_coefficients[i] = abs(pearsonr(hamming_weights, traces[:, i])[0])

        warnings.simplefilter("default")
        return pearson_coefficients

    @staticmethod
    def best_indices(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, points_of_interest: int,
                     subkey: int, aes_round: int = 1, aes_operation: int = 0, leakage_model: bool = True):
        """
        This method performs the full Pearson correlation analysis.

        :param traces: the traces used for the tool
        :param keys: the keys used for the tool
        :param plain: the plain used for the tool
        :param points_of_interest: the number of points of interest to extract.
        :param subkey: the subkey to correlate on. Used for hamming weight
            calculation, must be in the range [0-15].
        :param aes_round: the AES round to 'attack'.
        :param aes_operation: the AES operation to 'attack', represented as an integer from 0 to 3.
        :param leakage_model: whether to use hamming weight leakage model
        """

        hamming_weights = HammingWeight.hamming_weights(plain, keys, subkey, aes_round, aes_operation, leakage_model)
        if leakage_model:
            counts = np.zeros(9)
        else:
            counts = np.zeros(256)
        for i in hamming_weights:
            counts[i] += 1

        for i in counts:
            if i < 2:
                print('WARNING: Pearson does not have enough data points of certain value, '
                      'results may not be accurate.')
                return Pearson.pearson_coefficients(hamming_weights, points_of_interest, traces)
        return Pearson.pearson_coefficients(hamming_weights, points_of_interest, traces)
