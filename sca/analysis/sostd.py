import warnings

import numpy as np

from typing import List

from sca.util.hamming_weight import HammingWeight


class SOSTD:
    """
    This class contains methods for performing SOST and SOSD for feature
    selection.
    """

    @staticmethod
    def run(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, points_of_interest: int, save_traces: bool,
            subkey: int, aes_round: int = 1, aes_operation: int = 0, hamming_weight: bool = True,
            normalization: bool = True):
        """
        This method performs the full SOST or SOSD analysis.

        :param traces: the traces used for the tool
        :param keys: the keys used for the tool
        :param plain: the plain used for the tool
        :param points_of_interest: the number of points of interest to extract.
        :param save_traces: whether to save the full traces or just their indices.
        :param subkey: the subkey to correlate on. Used for hamming weight
            calculation, must be in the range [0-15].
        :param aes_round: the AES round to 'attack'.
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param aes_operation: the AES operation to 'attack', represented as an integer from 0 to 3.
        :param normalization: whether to normalize the coefficients or not. SOST includes normalization, SOSD does not.
        """
        indices = SOSTD.list_of_best_indices(traces, keys, plain, points_of_interest, subkey, aes_round,
                                             aes_operation, hamming_weight, normalization)
        if save_traces:
            print('Saving interesting traces... ', end='')
            np.save('out/sostd_selected_traces', traces[:, indices])
        else:
            print('Saving indices of interesting traces... ', end='')
            np.save('out/sostd_selected_indices', indices)

        print('Done')

    @staticmethod
    def list_of_indices_per_value(values: np.ndarray, traces: np.ndarray, hamming_weights: bool):
        """
        Builds a list with indices of the traces for every distinct value.

        :param values: list of hamming weights/values of the traces
        :param traces: the actual traces
        :param hamming_weights: what leakage model to use
        :return: List of indices per distinct value
        """

        if hamming_weights:
            amount_of_values = 9
        else:
            amount_of_values = 256
        indices_list = []
        for i in range(amount_of_values):
            indices_list.append([])
            for j in range(len(traces)):
                if values[j] == i:
                    indices_list[i].append(j)

        return indices_list

    @staticmethod
    def sostd_coefficients(indices: List[List[int]], traces: np.ndarray, hamming_weights: bool,
                           normalization: bool = True):
        """
        Returns the sost coefficients for every feature

        :param indices: List of list of indices for distinct values
        :param traces: the trace values
        :param hamming_weights: whether to use the hamming weight leakage model
        :param normalization: whether to normalize the coefficients or not.
        :return: SOST Values for every point in time
        """

        if normalization:
            print('Calculating SOST correlation coefficients')
        else:
            print('Calculating SOSD correlation coefficients')

        trace_length = len(traces[0])
        if hamming_weights:
            amount_of_values = 9
        else:
            amount_of_values = 256

        sostd_coefficients = np.empty(trace_length)

        for t in range(trace_length):

            means = np.empty(amount_of_values)
            variances = np.empty(amount_of_values)
            counts = np.empty(amount_of_values)
            result = 0

            for i in range(amount_of_values):

                values = traces[indices[i], t]
                means[i] = np.mean(values)
                variances[i] = np.var(values)
                counts[i] = len(values)

            for i in range(amount_of_values):

                for j in range(i):
                    temp = pow(means[i] - means[j], 2)

                    if normalization:
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                        temp /= np.sqrt(variances[i]/counts[i] + variances[j]/counts[j])

                        warnings.simplefilter("default")

                    result += temp

            sostd_coefficients[t] = result

        return sostd_coefficients

    @staticmethod
    def list_of_best_indices(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, points_of_interest: int,
                             subkey: int, aes_round: int = 1, aes_operation: int = 0, hamming_weight: bool = True,
                             normalization: bool = True):
        """
        This method performs the full SOST or SOSD analysis.

        :param traces: the traces used for the tool
        :param keys: the keys used for the tool
        :param plain: the plain used for the tool
        :param points_of_interest: the number of points of interest to extract.
        :param subkey: the subkey to correlate on. Used for hamming weight
            calculation, must be in the range [0-15].
        :param aes_round: the AES round to 'attack'.
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param aes_operation: the AES operation to 'attack', represented as an integer from 0 to 3.
        :param normalization: whether to normalize the coefficients or not. Normalization is part of SOST, without it,
            SOSD is run.
        :return List of best n indices
        """
        hamming_weights = HammingWeight.hamming_weights(plain, keys, subkey,
                                                        aes_round, aes_operation, hamming_weight)
        indices = SOSTD.list_of_indices_per_value(hamming_weights, traces, hamming_weight)
        sostd_coefficients = SOSTD.sostd_coefficients(indices, traces, hamming_weight, normalization)

        return np.argsort(-sostd_coefficients)[:points_of_interest][::-1]
