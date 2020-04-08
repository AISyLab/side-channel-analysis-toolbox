import numpy as np
from sca.util import progress_bar_util, aes, data_partitioner
from sca.loghandler.loghandler import LogHandler
from sca.util.hamming_weight import HammingWeight
import progressbar
from sca.analysis.pearson import Pearson


class CPA:
    """" This class contains methods for performing a MIA (Mutual Information
    Analysis) side-channel attack."""

    RANGE = 5000
    OFFSET = 0
    NUMBER_OF_TRACES = 10000
    KEY_SIZE = 256

    def __init__(self, traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, online: bool):
        """ Constructor for mia class

        :param traces: the traces to use
        :param keys: the keys to use
        :param plain: the plaintexts to use
        :param online: use online correlation calculation
        """
        self.traces = traces
        self.plain = plain
        self.keys = keys
        self.online = online
        self.NUMBER_OF_TRACES = self.traces.shape[0]
        self.RANGE = self.traces.shape[1]

        self.log_handler = LogHandler("mia", False)

    def set_range(self, range_: int):
        """ Set the range of the amount of data points to process
        :param range_: amount of data points to process in mia, selecting [offset, range_] of the traces
        """
        self.RANGE = range_

    def set_offset(self, offset: int):
        """ Set the offset at which to start reading the data points
        :param offset: Number at which to start reading the data points
        """
        self.OFFSET = offset

    @staticmethod
    def run(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, attack_traces: np.ndarray,
            attack_keys: np.ndarray, attack_plain: np.ndarray, round_: int, operation: int,
            subkey: int, feature_select: int, num_features: int, hamming_weight: bool = False,
            debug_mode_enabled: bool = False, online: bool = False, conditional_averaging: bool = False) -> np.ndarray:

        """ The run method of cpa

        :param traces: the traces to use
        :param keys: the keys to use
        :param plain: the plaintexts to use
        :param attack_traces: the traces to use for attacking
        :param attack_keys: the keys to use for attacking
        :param attack_plain: the plaintexts to use for attacking
        :param round_: the AES round to attack
        :param operation: the AES operation to attack, represented as an integer from 0 to 3
        :param feature_select: which feature select method to use.
        :param num_features: the number of features to select
        :param subkey: the subkey index to analyze. Must be in the range [0-15].
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param debug_mode_enabled: whether to enable debug mode
        :param online: use online correlation calculation.
        :param conditional_averaging: whether to use conditional averaging.
        :return: the calculated subkey corresponding to the subkey index specified
        """

        print("Executing Correlation Power Analysis")

        result = np.zeros(16, dtype=int)
        # Init progress bar with amount of iterations.
        if subkey == 16:

            bar = progressbar.ProgressBar(max_value=16 * 255,
                                          widgets=progress_bar_util.get_widgets(debug_mode_enabled))
            for i in range(16):

                if feature_select > 0:
                    indices = data_partitioner.DataPartitioner.select_features(traces, keys, plain, i, feature_select,
                                                                               num_features, round_, operation,
                                                                               hamming_weight)
                    cpa = CPA(attack_traces[:, indices], attack_keys, attack_plain, online)
                else:
                    cpa = CPA(attack_traces, attack_keys, attack_plain, online)
                result[i] = cpa.solve_subkey(i, hamming_weight, debug_mode_enabled, round_=round_,
                                             operation=operation, bar=bar, conditional_averaging=conditional_averaging)

        else:
            bar = progressbar.ProgressBar(max_value=255,
                                          widgets=progress_bar_util.get_widgets(debug_mode_enabled))

            if feature_select > 0:
                indices = data_partitioner.DataPartitioner.select_features(traces, keys, plain, subkey, feature_select,
                                                                           num_features, round_, operation,
                                                                           hamming_weight)
                cpa = CPA(attack_traces[:, indices], attack_keys, attack_plain, online)
            else:
                cpa = CPA(attack_traces, attack_keys, attack_plain, online)
            result[subkey] = cpa.solve_subkey(subkey, hamming_weight, debug_mode_enabled, round_=round_,
                                              operation=operation, bar=bar, conditional_averaging=conditional_averaging)
        bar.finish()
        print('The final key is: ', result)
        return result

    def solve_subkey(self, subkey: int, hamming_leakage: bool = True,
                     debug_mode_enabled: bool = False, round_: int = 1, operation: int = 0,
                     bar: progressbar.ProgressBar = None, conditional_averaging: bool = False):
        """ Solve a subkey
        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param hamming_leakage: whether to use the hamming_weight leakage model
        :param debug_mode_enabled: whether to enable debug mode
        :param round_: round of aes.
        :param operation: operation of aes.
        :param bar: the progressbar to update
        :param conditional_averaging: whether to use conditional averaging or not.
        :return: the calculated subkey corresponding to the subkey index specified
        """

        num_values = 256
        if hamming_leakage:
            num_values = 9

        if conditional_averaging:
            return CPA.conditional_avg(self, subkey, hamming_leakage, round_, operation, bar)

        log_handler = LogHandler("cpa", debug_mode_enabled)

        log_handler.log_to_debug_file('RUNNING cpa attack with \n'
                                      'NUM_FEATURES: {} \n'
                                      'ATTACK_TRACES: {} \n'
                                      'SUBKEY: {}.'
                                      .format(self.RANGE, self.NUMBER_OF_TRACES, subkey))
        correlation = np.zeros(self.KEY_SIZE)

        for key in range(self.KEY_SIZE):

            for i in range(len(self.traces)):
                self.keys[i][subkey] = key
            leakage_table = HammingWeight.hamming_weights(self.plain, self.keys, subkey, round_,
                                                          operation, hamming_leakage)/num_values-1

            if self.online:
                correlation[key] = max(self.online_coeffs(leakage_table))
            else:
                correlation[key] = max(Pearson.pearson_cpa(leakage_table, self.traces))
            if bar is not None and bar.value < bar.max_value:
                bar.update(bar.value + 1)
        return np.argmax(correlation)

    def online_coeffs(self, leakage_table: np.ndarray):
        """
        Calculates coefficients using online calculation method.

        :param leakage_table: the leakage table used.
        :return: the coefficients, calculated in an online manner.
        """
        num_traces = len(self.traces)
        results = np.zeros(len(self.traces[0]))
        for j in range(len(self.traces[0])):
            traces = self.traces[:, j]
            traces /= np.max(traces)

            ht = 0
            h = 0
            t = 0
            hh = 0
            tt = 0
            for d in range(num_traces):
                h += leakage_table[d]
                hh += np.power(leakage_table[d], 2)
                t += traces[d]
                tt = np.power(traces[d], 2)
                ht += leakage_table[d] * traces[d]
            denominator = (np.power(h, 2) - (num_traces * hh)) * (np.power(t, 2) - (num_traces * tt))
            numerator = num_traces * ht - h * t
            results[j] = np.abs(numerator/np.sqrt(np.abs(denominator)))
        return results

    def conditional_avg(self, subkey: int, hamming_leakage: bool, round_: int = 1, operation: int = 0,
                        bar: progressbar.ProgressBar = None):
        """
        Calculates coefficients using conditional averaging.

        :param subkey: the subkey to calculate.
        :param hamming_leakage: signifies which leakage model is used.
        :param round_: round of aes.
        :param operation: operation of aes.
        :param bar: the progressbar to update.
        :return: the coefficients, calculated using conditional averaging.
        """
        num_traces = len(self.traces)

        num_values = 256
        if hamming_leakage:
            num_values = 9

        results = np.zeros(self.KEY_SIZE)

        for key in range(self.KEY_SIZE):
            conditional_avg = np.zeros(num_values)
            count = np.zeros(num_values)
            for i in range(len(self.traces)):
                self.keys[i][subkey] = key
            leakage_table = HammingWeight.hamming_weights(self.plain, self.keys, subkey, round_,
                                                          operation, hamming_leakage)

            var_range = np.var(leakage_table)
            mean_range = np.mean(leakage_table)
            covariance_index = np.zeros(len(self.traces[0]))

            for x in range(len(self.traces[0])):
                for i in range(num_traces):
                    conditional_avg[leakage_table[i]] = conditional_avg[leakage_table[i]] + self.traces[i][x]
                    count[leakage_table[i]] = count[leakage_table[i]] + 1
                for i in range(np.power(2, num_traces)):
                    conditional_avg[i] = conditional_avg[i] / count[i]

            for x in range(len(self.traces[0])):
                mean = np.mean(conditional_avg)
                variance = np.var(conditional_avg)
                covariance = 0

                for i in range(num_values):
                    covariance = covariance + i * conditional_avg[i]

                covariance_index[x] = np.abs(covariance / num_values - mean * mean_range) / np.sqrt(
                    variance * var_range)

            if bar is not None and bar.value < bar.max_value:
                bar.update(bar.value + 1)

            results[key] = np.max(covariance_index)

        return np.argmax(results)
