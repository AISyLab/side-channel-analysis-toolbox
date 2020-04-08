import numpy as np
import progressbar  # This actually imports progressbar2 but `import progressbar2' itself doesn't work. Weird package.
import torch
from typing import List
from sca.util import progress_bar_util, aes
from sca.util.data_partitioner import DataPartitioner
from sca.loghandler.loghandler import LogHandler


class SA:
    """"This class contains methods for performing Stochastic Attack."""

    log_handler: LogHandler = None

    profiling_traces: np.ndarray
    profiling_keys: np.ndarray
    profiling_plain: np.ndarray
    attack_traces: np.ndarray
    attack_keys: np.ndarray
    attack_plain: np.ndarray

    def __init__(self, template_traces: np.ndarray, template_keys: np.ndarray, template_plaintext: np.ndarray,
                 attack_traces: np.ndarray, attack_keys: np.ndarray, attack_plaintext: np.ndarray):
        self.profiling_traces = template_traces
        self.profiling_keys = template_keys
        self.profiling_plain = template_plaintext
        self.attack_traces = attack_traces
        self.attack_keys = attack_keys
        self.attack_plain = attack_plaintext

    # TODO: Return key to main CLI to be displayed.
    @staticmethod
    def run(profiling_traces: np.ndarray, profiling_keys: np.ndarray, profiling_plaintext: np.ndarray,
            attack_traces: np.ndarray, attack_keys: np.ndarray, attack_plaintext: np.ndarray, round_: int,
            operation: int, num_traces: int, num_attack_traces: int, subkey: int, feature_select: int,
            num_features: int, use_gpu: bool = False, hamming_weight: bool = False, debug_mode_enabled: bool = False) \
            -> List[int]:

        """
        Runs a full stochastic attack

        :param profiling_traces: the traces to use
        :param profiling_keys: the keys to use
        :param profiling_plaintext: the plaintexts to use
        :param attack_traces: the traces to use for attacking
        :param attack_keys: the keys to use for attacking
        :param attack_plaintext: the plaintexts to use for attacking
        :param round_: the AES round to attack
        :param operation: the AES operation to attack, represented as an integer from 0 to 3
        :param num_traces: number of data points to process
        :param num_attack_traces: number of data points to attack
        :param feature_select: which feature select method to use.
        :param num_features: the number of features to select
        :param subkey: the subkey index to analyze. Must be in the range [0-16], 16 signaling the whole key.
        :param use_gpu: whether or not to use gpu acceleration
        :param hamming_weight: whether or not to use the hamming_weight leakage model
        :param debug_mode_enabled: whether or not to enable debug mode
        :returns: list containing the (sub)key
        """

        print('This performs a stochastic attack.')

        # Instantiate a SA (Stochastic Attack) object to call dynamic method.
        sa = SA(profiling_traces, profiling_keys, profiling_plaintext, attack_traces, attack_keys, attack_plaintext)
        sa.log_handler = LogHandler('sa', debug_mode_enabled)

        if sa.log_handler is not None:
            sa.log_handler.log_to_debug_file('RUNNING stochastic attack with \n'
                                             'TEMPLATE_TRACES: {} \n'
                                             'NUM_FEATURES: {} \n'
                                             'ATTACK_TRACES: {} \n'
                                             'SUBKEY: {}.'
                                             .format(len(sa.profiling_traces),
                                                     num_features, len(sa.attack_traces), subkey))

        # Init progress bar with estimated amount of iterations
        num_subkeys = 16 if subkey == 16 else 1
        max_value = num_attack_traces * 256 * num_subkeys * 15
        bar = progressbar.ProgressBar(max_value=max_value, widgets=progress_bar_util.get_widgets(debug_mode_enabled))

        result = [0 for _ in range(16)]

        # call method that returns the key, currently with no arguments taken from the user.
        if subkey == 16:
            for i in range(subkey):
                result[i] = sa.solve_subkey(i, use_gpu, bar, feature_select=feature_select,
                                            num_features=num_features,
                                            num_traces=num_traces,
                                            num_attack_traces=num_attack_traces, hamming_weight=hamming_weight,
                                            aes_round=round_, aes_operation=operation)
        else:
            result[subkey] = sa.solve_subkey(subkey, use_gpu, bar, feature_select=feature_select,
                                             num_features=num_features, num_traces=num_traces,
                                             num_attack_traces=num_attack_traces, hamming_weight=hamming_weight,
                                             aes_round=round_, aes_operation=operation)

        bar.finish()

        print('The final key is: ', result)
        return result

    def solve_subkey(self, subkey: int, use_gpu: bool, bar: progressbar.ProgressBar, feature_select: int,
                     num_features: int, aes_round: int = 1, aes_operation: int = 0, num_traces: int = 5000,
                     num_attack_traces: int = 30, hamming_weight: bool = False):
        """
        This is the over-arching method for finding the key of an AES-encrypted cypher text.
        For detailed explanation of the algorithm see: https://link.springer.com/chapter/10.1007/11545262_3

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param use_gpu: whether to use gpu acceleration
        :param bar: the progressbar to update
        :param aes_round: the AES round to attack
        :param aes_operation: the AES operation to attack, represented as an integer from 0 to 3
        :param num_traces: number of data points to process
        :param num_attack_traces: number of data points to attack
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param feature_select: which feature select method to use.
        :param num_features: the number of features to select
        :return: the calculated subkey corresponding to the subkey index specified
        """

        # Input sanitization
        # pt. 1 : our inputs must not be empty
        if self.profiling_traces.size <= 0 or self.profiling_keys.size <= 0 or self.profiling_plain.size <= 0:
            raise (ValueError("One or more of the following inputs was or were empty:\n-traces\n-keys\n-plain"))
        # pt. 2 : our inputs must be of equal sizes
        if self.profiling_traces.size - 1 != self.profiling_keys.size != self.profiling_plain.size:
            raise (ValueError("One or more of the following inputs were of unequal size:\n-traces\n-keys\n-plain"))
        # pt. 3 : our key consists of 16 subkeys. thus indices range from 0 to 15
        if 0 > subkey or subkey > 15:
            raise (ValueError("Subkey index out of range, should be between 0 and 15"))

        # Feature selection
        if feature_select > 0:
            indices = DataPartitioner.select_features(self.profiling_traces, self.profiling_keys, self.profiling_plain,
                                                      subkey, feature_select, num_features, aes_round, aes_operation,
                                                      hamming_weight)
            traces = self.profiling_traces[:, indices]
            attack_traces = self.attack_traces[:, indices]
        else:
            traces = self.profiling_traces
            attack_traces = self.attack_traces

        # Actual method calls
        bit_matrix = self.poa_output(subkey, num_traces, hamming_weight, aes_round, aes_operation)
        estimates = SA.parameter_estimation(traces, bit_matrix, use_gpu, num_traces=num_traces,
                                            hamming_weight=hamming_weight)

        best_key = self.key_extraction(estimates, attack_traces, subkey, bar,
                                       num_attack_traces=num_attack_traces,
                                       hamming_weight=hamming_weight, aes_round=aes_round, aes_operation=aes_operation)

        # Print the necessary stuff while debugging
        debug_string = \
            "Subkey " + str(subkey) + " is: " + str(best_key) + " | first 10 POI indices: " + str(indices[:10])
        self.log_handler.log_to_debug_file(self.log_handler.CONST_SEPARATOR)
        self.log_handler.log_to_debug_progressbar(bar, debug_string)

        return best_key

    # TODO: add point of attack (poa) parameter. Then add comment explaining this functionality
    def poa_output(self, subkey: int, num_traces: int, hamming_weight: bool = False, aes_round: int = 1,
                   aes_operation: int = 0) -> np.ndarray:

        """ Method that returns the output of the AES algorithm we will perform analysis on.

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param num_traces: number of data points to process
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param aes_operation: operation of aes to attack.
        :param aes_round: round of aes to attack.
        :return: output of the aes algorithm
        """

        # Initialize output array, with a row of 1's for noise, which is always present.
        if hamming_weight:
            bit_matrix = np.zeros((num_traces, 2))
        else:
            bit_matrix = np.zeros((num_traces, 9))
        bit_matrix[:, 0] = 1

        aes_object = aes.AES(np.zeros(16, dtype=int))

        # Put our dummy plaintext through the sbox and return the result.
        for i in range(num_traces):
            aes_object.change_key(self.profiling_keys[i])
            temp = aes_object.encrypt(self.profiling_plain[i], round_output=aes_round, single_byte=True,
                                      result_byte=subkey, operation=aes_operation)

            if hamming_weight:
                bit_matrix[i][1] = temp
            else:
                # Use string to calculate which bits are on
                binary_temp = format(temp, '08b')
                for j in range(len(binary_temp)):
                    bit_matrix[i][j + 1] = int(binary_temp[j])
        return bit_matrix

    @staticmethod
    def parameter_estimation(traces: np.ndarray, bit_matrix: np.ndarray, use_gpu: bool, num_traces: int,
                             hamming_weight: bool = False) -> np.ndarray:
        """ Method that returns a matrix that, for every point in time, has an estimation of the dependency between
        a bit value and the value of the trace at that time.

        :param traces: the traces to use
        :param bit_matrix: bit_matrix containing aes information
        :param use_gpu: whether to use gpu acceleration
        :param num_traces: number of data points to process
        :param hamming_weight: whether to use the hamming_weight leakage model
        :return: matrix containing estimations of dependencies for every point in time
        """

        # Initialize our matrices
        measurements = np.zeros(num_traces)
        if hamming_weight:
            estimates = np.zeros((len(traces[0]), 2))
        else:
            estimates = np.zeros((len(traces[0]), 9))
        # Fill matrix with measurements for the every available point in time [t], for every trace [i].
        for t in range(len(traces[0])):
            for i in range(num_traces):
                measurements[i] = traces[i][t]

            # From our measurements and our bit matrix we can calculate our bit dependencies.
            estimates[t] = SA.matrix_calculations(bit_matrix, measurements, use_gpu)
        return estimates

    @staticmethod
    def matrix_calculations(bit_matrix: np.ndarray, measurements: np.ndarray, use_gpu: bool) -> np.ndarray:
        """ Method that handles the main matrix calculations for this attack.

        :param bit_matrix: bit_matrix containing aes information
        :param measurements: trace measurements
        :param use_gpu: whether to use gpu acceleration
        :return: matrix containing results of matrix calculations according to the stochastic attack
        """

        if use_gpu:

            # init of matrices on gpu as tensors
            t_bt = torch.as_tensor(bit_matrix).cuda()
            t_bt_t = torch.t(t_bt)
            t_ms = torch.as_tensor(measurements).cuda()

            # computations
            res1 = torch.mm(t_bt_t, t_bt)
            res2 = torch.as_tensor(np.linalg.inv(res1.cpu().numpy())).cuda()
            res3 = torch.mm(res2, t_bt_t)
            res4 = torch.mv(res3, t_ms)
            res = res4.cpu().numpy()

            return res
        else:

            # we stay on cpu, so we can start computations immediately
            res1 = np.dot(bit_matrix.T, bit_matrix)
            res2 = np.linalg.inv(res1)
            res3 = np.dot(res2, bit_matrix.T)
            res = np.dot(res3, measurements)

            return res

    # TODO: Poi selection is currently done somewhat arbitrarily, based on some empirical testing.
    # TODO: This should be done more systematically.
    @staticmethod
    def poi_selection(estimates: np.ndarray) -> List[int]:
        """ Method to select points of interest (poi's).
        These are points in which the dependency between the traces and the value of our bits is the highest.

        :param estimates: points to get points of interest from
        :return: list containing indices of points of interest
        """

        # Sum dependencies per individual bit for every point in time.
        euclidian_length_vectors = np.arange(0.0, len(estimates))
        for i in range(len(estimates)):
            euclidian_length_vectors[i] = np.linalg.norm(estimates[i][1:])

        # Determine poi's by taking points that differ more than a certain amount of standard deviations from the mean.
        # This amount was empirically chosen.
        standard_deviation = np.std(euclidian_length_vectors)
        average_length_vectors = np.mean(euclidian_length_vectors)
        points_of_interest = []
        for i in range(len(euclidian_length_vectors)):
            if np.abs(euclidian_length_vectors[i] - average_length_vectors) >= 3.5 * standard_deviation:
                points_of_interest.append(i)

        return points_of_interest

    # TODO: we currently use traces from our own data to perform the attack on. This should be modified so we take
    # TODO: a set of traces as an argument.
    def key_extraction(self, estimates: np.ndarray, traces: np.ndarray, subkey: int, bar: progressbar.ProgressBar,
                       num_attack_traces: int = 30, top_n_guesses=1, hamming_weight: bool = False,
                       aes_round: int = 1, aes_operation: int = 0) -> int:
        """
        Method for key extraction. Given a set of traces from a system we want to attack,
        we go through the same profiling process as before, then compare the output to our profiles.
        The key that matches the closest is chosen as the best candidate key.

        :param estimates: matrix containing estimations of dependencies
        :param traces: the traces to use
        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param bar: the progressbar to update
        :param num_attack_traces: number of data points to attack
        :param top_n_guesses: amount of key guesses to log
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param aes_operation: operation of aes to attack.
        :param aes_round: round of aes to attack.
        :return: the calculated subkey
        """

        min_sums = 255e155
        best_key = -1
        full_key = self.attack_keys[0]
        aes_object = aes.AES(full_key)
        scores = np.zeros(256)
        for key in range(256):
            full_key[subkey] = key
            aes_object.change_key(full_key)
            temp_sum = 0
            for i in range(len(traces[0])):
                for j in range(num_attack_traces):
                    atk_trace = traces[j]
                    temp = aes_object.encrypt(self.attack_plain[j], round_output=aes_round,
                                              single_byte=True, result_byte=subkey,
                                              operation=aes_operation)
                    # due to format of 0bXXXXXXXX we split on b and take the second part
                    binary_temp = format(temp, '08b')
                    inner_temp_sum = estimates[i][0]
                    if hamming_weight:
                        inner_temp_sum += estimates[i][1] * temp
                    else:
                        for l in range(len(binary_temp)):
                            inner_temp_sum += int(binary_temp[l]) * estimates[i][l + 1]

                    temp_sum += pow(atk_trace[i] - inner_temp_sum, 2)
                    if bar.max_value is None or bar.value + 1 < bar.max_value:
                        bar.update(bar.value + 1)

            if temp_sum < min_sums:
                min_sums = temp_sum
                best_key = key
            scores[key] = temp_sum

        if self.log_handler is not None:
            self.log_handler.log_to_debug_file('Best 5 key guesses are {}.'.
                                               format(np.argsort(scores)[:5]))

        if top_n_guesses == 1:
            return best_key
        return np.argsort(scores)[:top_n_guesses]
