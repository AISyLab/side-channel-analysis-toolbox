import multiprocessing
import numpy as np
from collections import Counter
from pyitlib import discrete_random_variable as drv
from tabulate import tabulate
import progressbar  # This actually imports progressbar2 but `import progressbar2' itself doesn't work. Weird package.
from typing import Tuple, List
from sca.util import progress_bar_util, aes, data_partitioner
from sca.loghandler.loghandler import LogHandler
from functools import partial


class Mia:
    """" This class contains methods for performing a MIA (Mutual Information
    Analysis) side-channel attack."""

    RANGE = 5000
    OFFSET = 0
    NUMBER_OF_TRACES = 10000
    KEY_SIZE = 256

    def __init__(self, traces: np.ndarray, keys: np.ndarray, plain: np.ndarray):
        """ Constructor for mia class

        :param traces: the traces to use
        :param keys: the keys to use
        :param plain: the plaintexts to use
        """
        self.traces = traces
        self.plain = plain
        self.keys = keys

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
            subkey: int, feature_select: int, num_features: int,
            hamming_weight: bool = False, debug_mode_enabled: bool = False) -> int:

        """ The run method of mia

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
        :return: the calculated subkey corresponding to the subkey index specified
        """

        print("Executing Mutual Information Analysis")

        result = np.zeros(16, dtype=int)
        # Init progress bar with amount of iterations.
        if subkey == 16:

            bar = progressbar.ProgressBar(max_value=16 * 256,
                                          widgets=progress_bar_util.get_widgets(debug_mode_enabled))
            for i in range(16):

                if feature_select > 0:
                    indices = data_partitioner.DataPartitioner.select_features(traces, keys, plain, i, feature_select,
                                                                               num_features, round_, operation,
                                                                               hamming_weight)
                    mia = Mia(attack_traces[:, indices], attack_keys, attack_plain)
                else:
                    mia = Mia(attack_traces, attack_keys, attack_plain)

                result[i] = mia.solve_subkey(i, hamming_weight, debug_mode_enabled, round_=round_,
                                             operation=operation, bar=bar)

        else:
            if feature_select > 0:
                indices = data_partitioner.DataPartitioner.select_features(traces, keys, plain, subkey, feature_select,
                                                                           num_features, round_, operation,
                                                                           hamming_weight)
                attack_traces = attack_traces[:, indices]

            mia = Mia(attack_traces, attack_keys, attack_plain)

            bar = progressbar.ProgressBar(max_value=mia.KEY_SIZE + 1,
                                          widgets=progress_bar_util.get_widgets(debug_mode_enabled))
            result[subkey] = mia.solve_subkey(subkey, hamming_weight, debug_mode_enabled,
                                              round_=round_, operation=operation, bar=bar)

        bar.finish()
        print('The final key is: ', result)

        return result

    def solve_subkey(self, subkey: int, hamming_leakage: bool = True,
                     debug_mode_enabled: bool = False, round_: int = 1, operation: int = 0,
                     bar: progressbar.ProgressBar = None) -> int:
        """ Solve a subkey
        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param hamming_leakage: whether to use the hamming_weight leakage model
        :param debug_mode_enabled: whether to enable debug mode
        :param round_: round of aes.
        :param operation: operation of aes.
        :param bar: the progressbar to update
        :return: the calculated subkey corresponding to the subkey index specified
        """

        log_handler = LogHandler("mia", debug_mode_enabled)

        log_handler.log_to_debug_file('RUNNING mia attack with \n'
                                      'NUM_FEATURES: {} \n'
                                      'ATTACK_TRACES: {} \n'
                                      'SUBKEY: {}.'
                                      .format(self.RANGE, self.NUMBER_OF_TRACES, subkey))

        self.calculate_leakage_table(subkey, bar, round_, operation, hamming_leakage)

        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result = []
        calculate_point_in_time_subkey = partial(self.calculate_point_in_time)
        best_key = -1
        best_mutual_information = 0
        best_time = -1

        for i, value in enumerate(
                pool.map(calculate_point_in_time_subkey, range(self.OFFSET, self.OFFSET + self.RANGE)), 2):
            mutual_information, key = value

            # Update the most probable key guess
            if mutual_information > best_mutual_information:
                best_mutual_information = mutual_information
                best_key = key
                best_time = i

            # Print and log the necessary stuff while debugging
            string = "MI at: t=%d is %f with key guess: %d | Best so far: t=%d, MI=%f, key=%d"
            data = (i, mutual_information, int(key), best_time, best_mutual_information, int(best_key))
            debug_string = string % data

            log_handler.log_to_debug_file(debug_string)
            log_handler.log_to_debug_progressbar(bar, debug_string)
            if bar is not None and bar.value < bar.max_value:
                bar.update(bar.value + 1)
            result.append(value)

        pool.close()

        log_handler.log_to_debug_file(self.tabulate_top_n(result, 10))

        return best_key

    def calculate_point_in_time(self, time: int) -> Tuple[float, int]:
        """ Calculate the mutual_information on a given moment in time

        :param time: number representing a timestamp, should be in range
        :return: tuple containing the max mutual information and the best key guess given the time stamp
        """

        max_mutual = 0
        best_key_guess = -1
        for key in range(self.KEY_SIZE):

            # Estimate observations using a histogram
            observations_at_time = self.get_observations_at_time(time)
            occ, bin_edges = np.histogram(observations_at_time, bins='auto')
            bin_values = np.digitize(observations_at_time, bin_edges)

            # Convert discrete random variables to int numpy arrays
            # so they can be used to compute the mutual information
            o = np.array(bin_values, dtype=int)
            h = np.array(self.leakage[key], dtype=int)

            # Calculate mutual information.
            mutual_information = self.compute_mutual_information(o, h)
            if mutual_information > max_mutual:
                max_mutual = mutual_information
                best_key_guess = key

        return max_mutual, best_key_guess

    def calculate_leakage_table(self, subkey: int, bar: progressbar.ProgressBar = None, aes_round: int = 1,
                                aes_operation: int = 0, hamming_weight: bool = True):
        """Prepare a multidimensional table containing all the pre-calculated hamming weights for every data point
        at every point in time

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param aes_round: the AES round to attack
        :param bar: The progressbar to update
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param aes_operation: the AES operation to attack, represented as an integer from 0 to 3
        """

        full_key = np.zeros(16, dtype=int)

        self.leakage = np.zeros(shape=(self.KEY_SIZE, self.NUMBER_OF_TRACES), dtype=int)
        aes_object = aes.AES(full_key)
        for key in range(self.leakage.shape[0]):
            for trace in range(self.leakage.shape[1]):
                if aes_round == 1 and aes_operation == 0:
                    box = aes.sbox[key ^ self.plain[trace][subkey]]
                else:
                    if not (trace > 1 and np.array_equal(self.keys[trace], self.keys[trace - 1])):
                        full_key = self.keys[trace]
                        full_key[subkey] = key
                        aes_object.change_key(full_key)
                    box = aes_object.encrypt(self.plain[trace], round_output=aes_round,
                                             operation=aes_operation, single_byte=True, result_byte=subkey)
                if hamming_weight:
                    result = bin(box).split('b')[1].count('1')
                else:
                    result = box
                self.leakage[key, trace] = result
            if bar is not None:
                bar.update(bar.value + 1)

    @staticmethod
    def compute_mutual_information(data_points: np.ndarray, hamming_weights: np.ndarray) -> int:
        """ This method computes the mutual information of two discrete data sets using Shannon entropy

        :param data_points: data points to compute the mutual information on
        :param hamming_weights: hamming weights used
        :return: mutual information value
        """

        return drv.information_mutual(data_points, hamming_weights)

    @staticmethod
    def extract_key(data: List[Tuple[float, int]]) -> int:
        """ Extract the most probable key guess from the highest mutual_information

        :param data: list of mutual information points
        :return: the most probable key guess
        """
        best_mutual_information = 0
        best_key_guess = -1
        for mutual_information, key in data:
            if mutual_information > best_mutual_information:
                best_mutual_information = mutual_information
                best_key_guess = key

        return best_key_guess

    @staticmethod
    def tabulate_top_n(data: List[Tuple[float, int]], n: int) -> str:
        """"" This method prints the most occurring key guesses

        :param data: list of mutual information points
        :param n: amount of key guesses to log
        :return: log string
        """

        counter = Counter([entry[1] for entry in data])
        return tabulate(counter.most_common(n), headers=['Key', 'Occurrences'])

    def get_observations_at_time(self, time: int) -> np.ndarray:
        """ Get the array of all observations at a certain point in time

        :param time: the timestamp to get
        :return: observations given the timestamp
        """

        if time < 0 or time >= self.NUMBER_OF_TRACES:
            return np.array([])

        return self.traces[:, time]
