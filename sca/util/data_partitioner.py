import numpy as np
import random
from sca.analysis.pearson import Pearson
from sca.analysis.sostd import SOSTD
from sca.analysis.nicv import NICV
from sca.analysis.snr import SNR


class DataPartitioner:
    """ Class that helps partition and process data for the attacks."""

    @staticmethod
    def get_traces(traces: np.ndarray, keys: np.ndarray, plaintexts: np.ndarray, num_traces: int, subkey: int,
                   feature_select: int, attack_traces: int, num_features: int,
                   aes_round: int, aes_operation: int, hamming_weight: bool = True):
        """ Gets a subset of traces.

        :param traces: The measurements.
        :param keys: the keys.
        :param plaintexts: the plaintexts.
        :param num_traces: the number of traces for profiling.
        :param subkey: the subkey to feature select for.
        :param feature_select: which feature select method to use.
        :param attack_traces: number of traces for attack phase of algorithm.
        :param num_features: the number of features to select.
        :param aes_round: what aes round to attack.
        :param aes_operation: what aes operation to attack.
        :param hamming_weight: whether to use the hamming weight leakage model.
        :return: a 2d array that has the traces and keys.
        """

        if num_traces + attack_traces > len(traces):
            temp = num_traces + attack_traces
            print('ERROR: Less traces provided than specified. (Info:  %d traces specified, only %d provided)'
                  % (temp, len(traces)))
            exit(1)

        indices = random.sample(list(range(0, len(traces))), num_traces + attack_traces)
        indices_for_profiling = indices[:num_traces]
        indices_for_attack = indices[num_traces:]

        profiling_traces = traces[indices_for_profiling, :]
        profiling_keys = keys[indices_for_profiling, :]
        profiling_plaintexts = plaintexts[indices_for_profiling, :]
        attack_traces = traces[indices_for_attack, :]
        attack_keys = keys[indices_for_attack, :]
        attack_plaintexts = plaintexts[indices_for_attack, :]

        if feature_select > 0:
            features = DataPartitioner.select_features(profiling_traces, profiling_keys, profiling_plaintexts, subkey,
                                                       feature_select, num_features, aes_round, aes_operation,
                                                       hamming_weight)
            return profiling_traces[:, features], profiling_keys, profiling_plaintexts, attack_traces[:, features], \
                attack_keys, attack_plaintexts

        return traces[indices_for_profiling, :], keys[indices_for_profiling, :], plaintexts[indices_for_profiling, :], \
            traces[indices_for_attack, :], keys[indices_for_attack, :], plaintexts[indices_for_attack, :]

    @staticmethod
    def select_features(traces: np.ndarray, keys: np.ndarray, plaintexts: np.ndarray, subkey: int,
                        feature_select: int, num_features: int, aes_round: int, aes_operation: int,
                        hamming_weight: bool = True):
        """ Selects the right feature selection method and runs it.

        :param traces: The measurements.
        :param keys: the keys.
        :param plaintexts: the plaintexts.
        :param subkey: the subkey to feature select for.
        :param feature_select: which feature select method to use.
        :param num_features: the number of features to select.
        :param aes_round: what aes round to attack.
        :param aes_operation: what aes operation to attack.
        :param hamming_weight: whether to use the hamming weight leakage model.
        :return: the best features.
        """

        if feature_select == 1:
            return Pearson.best_indices(traces, keys, plaintexts, num_features, subkey, aes_round, aes_operation,
                                        hamming_weight)
        if feature_select == 2:
            return SOSTD.list_of_best_indices(traces, keys, plaintexts, num_features, subkey,
                                              aes_round, aes_operation, True)
        if feature_select == 3:
            return NICV.get_points_of_interest_indices(plaintexts, traces, num_features, subkey, keys, hamming_weight)
        if feature_select == 4:
            return SOSTD.list_of_best_indices(traces, keys, plaintexts, num_features, subkey, aes_round, aes_operation,
                                              False)
