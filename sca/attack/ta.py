import numpy as np
import torch
import progressbar  # This actually imports progressbar2 but `import progressbar2' itself doesn't work. Weird package.
from scipy.stats import multivariate_normal
from typing import List
from tabulate import tabulate
from sca.util import progress_bar_util, aes
from sca.loghandler.loghandler import LogHandler
from sca.util.hamming_weight import HammingWeight
from sca.util.data_partitioner import DataPartitioner


class TA:
    """" This class contains methods for performing a Template Attack, pooled or normal."""

    hw = [bin(x).count('1') for x in range(256)]
    log_handler: LogHandler = None

    template_traces: np.ndarray
    template_keys: np.ndarray
    template_plain: np.ndarray
    attack_traces: np.ndarray
    attack_keys: np.ndarray
    attack_plain: np.ndarray

    def __init__(self, template_traces: np.ndarray, template_keys: np.ndarray, template_plaintext: np.ndarray,
                 attack_traces: np.ndarray, attack_keys: np.ndarray, attack_plaintext: np.ndarray):
        self.template_traces = template_traces
        self.template_keys = template_keys
        self.template_plain = template_plaintext
        self.attack_traces = attack_traces
        self.attack_keys = attack_keys
        self.attack_plain = attack_plaintext

    @staticmethod
    def run(template_traces: np.ndarray, template_keys: np.ndarray, template_plaintext: np.ndarray,
            attack_traces: np.ndarray, attack_keys: np.ndarray, attack_plain: np.ndarray, pooled: bool,
            num_points_of_interest: int, spacing_points_of_interest: int, subkey: int, gpu: bool = False,
            leakage_model: bool = True, debug_mode_enabled: bool = False, feature_select: int = 0) -> np.array:
        """ Method used to select correct version of Template Attack.

        :param template_traces: the traces to use
        :param template_keys: the keys to use
        :param template_plaintext: the plaintexts to use
        :param attack_traces: the traces to use for attacking
        :param attack_keys: the keys to use for attacking
        :param attack_plain: the plaintexts to use for attacking
        :param pooled: whether to do a pooled attack
        :param num_points_of_interest: number of points of interest to use
        :param spacing_points_of_interest: spacing between the points of interest
        :param subkey: the subkey index to analyze. Must be in the range [0-16]. 16 signals the full key.
        :param gpu: whether or not to use gpu for this attack
        :param leakage_model: the leakage model to use
        :param debug_mode_enabled: whether to enable debug mode
        :param feature_select: which feature selection method to use, see main for which number is which.
        :return: array containing the calculated key
        """

        # Init progress bar with rough amount of iterations
        num_subkeys = 16 if subkey == 16 else 1  # Attack takes roughly equal time per subkey so * 16 for whole key
        max_value = len(attack_traces) * 256 * num_subkeys
        bar = progressbar.ProgressBar(max_value=max_value, widgets=progress_bar_util.get_widgets(debug_mode_enabled))

        ta = TA(template_traces, template_keys, template_plaintext, attack_traces, attack_keys, attack_plain)

        indices = []

        if feature_select > 0:

            print("Feature selection is being calculated...")

            if num_subkeys == 16:
                for i in range(16):

                    temp = DataPartitioner.select_features(template_traces, template_keys, template_plaintext, i,
                                                           feature_select, num_points_of_interest, 1, 0, leakage_model)
                    for j in temp:
                        indices.append(j)
            else:
                # Select at least 10 features
                num_features = max(num_points_of_interest, 10)
                indices = DataPartitioner.select_features(template_traces, template_keys, template_plaintext, subkey,
                                                          feature_select, num_features, 1, 0, leakage_model)

            ta.template_traces, ta.attack_traces = template_traces[:, indices], attack_traces[:, indices]

        if pooled:
            result = ta.run_pooled(num_points_of_interest, spacing_points_of_interest, subkey,
                                   bar, gpu, leakage_model, debug_mode_enabled)
        else:
            result = ta.run_normal(num_points_of_interest, spacing_points_of_interest, subkey,
                                   bar, gpu, leakage_model, debug_mode_enabled)

        bar.finish()

        print('The final key is: ', result)
        return result

    def run_normal(self, num_points_of_interest: int, spacing_points_of_interest: int, selected_subkey: int,
                   bar: progressbar.ProgressBar, gpu: bool, leakage_model: bool,
                   debug_mode_enabled: bool = False) -> List[int]:
        """ Run method used to run the normal Template Attack.

        :param num_points_of_interest: number of points of interest to use
        :param spacing_points_of_interest: spacing between the points of interest
        :param selected_subkey: the subkey index to analyze. Must be in the range [0-16], 16 signals the whole key.
        :param bar: the progressbar to update
        :param gpu: whether or not to use gpu
        :param leakage_model: the leakage model to use
        :param debug_mode_enabled: whether to enable debug mode
        :return: list containing the calculated (sub)key
        """

        print('Template attack is being executed...')

        self.log_handler = LogHandler('ta', debug_mode_enabled)
        self.log_handler.log_to_debug_file('RUNNING template attack with \n'
                                           'TEMPLATE_TRACES: {} \n'
                                           'NUM_FEATURES: {} \n'
                                           'ATTACK_TRACES: {}\n'
                                           'SUBKEY: {}.'
                                           .format(len(self.template_traces), num_points_of_interest,
                                                   len(self.attack_traces), selected_subkey))

        num_attack_traces = len(self.attack_traces)

        if selected_subkey == 16:
            return [self.normal_solve_subkey(x, num_points_of_interest, spacing_points_of_interest,
                                             num_attack_traces, bar, gpu, leakage_model) for x in range(16)]
        else:
            result = [0 for _ in range(16)]
            result[selected_subkey] = self.normal_solve_subkey(selected_subkey, num_points_of_interest,
                                                               spacing_points_of_interest, num_attack_traces, bar, gpu,
                                                               leakage_model)
            return result

    def run_pooled(self, num_points_of_interest: int, spacing_points_of_interest: int, selected_subkey: int,
                   bar: progressbar.ProgressBar, gpu: bool, leakage_model: bool,
                   debug_mode_enabled: bool = False) -> List[int]:
        """ Run method used to run the pooled Template Attack.

        :param num_points_of_interest: number of points of interest to use
        :param spacing_points_of_interest: spacing between the points of interest
        :param selected_subkey: the subkey index to analyze. Must be in the range [0-16]
        :param bar: the progressbar to update
        :param gpu: whether or not to use gpu for this attack
        :param leakage_model: the leakage model to use
        :param debug_mode_enabled: whether to enable debug mode
        :return: list containing the calculated (sub)key
        """

        print('Pooled template attack is being executed...')

        self.log_handler = LogHandler('pooled_ta', debug_mode_enabled)

        self.log_handler.log_to_debug_file('RUNNING pooled template attack with \n'
                                           'TEMPLATE_TRACES: {} \n'
                                           'NUM_FEATURES: {} \n'
                                           'ATTACK_TRACES: {} \n'
                                           'SUBKEY: {}.'
                                           .format(len(self.template_traces), num_points_of_interest,
                                                   len(self.attack_traces), selected_subkey))

        num_attack_traces = len(self.attack_traces)
        result = [0 for _ in range(16)]

        mean_list, poi_list, pooled_covariance_matrix = \
            self.pooled_covariance_matrix(num_points_of_interest, spacing_points_of_interest, gpu, leakage_model)

        if selected_subkey == 16:
            for x in range(16):
                template_means = mean_list[x]
                points_of_interest = poi_list[x]

                subkey_result = self.pooled_solve_subkey(x, points_of_interest, template_means, num_points_of_interest,
                                                         num_attack_traces, pooled_covariance_matrix, bar)

                result[x] = subkey_result
        else:
            template_means = mean_list[selected_subkey]
            points_of_interest = poi_list[selected_subkey]

            result[selected_subkey] = self.pooled_solve_subkey(selected_subkey, points_of_interest, template_means,
                                                               num_points_of_interest, num_attack_traces,
                                                               pooled_covariance_matrix, bar)

        return result

    def normal_solve_subkey(self, subkey: int, num_poi: int, spacing_poi: int, num_attack_traces: int,
                            bar: progressbar.ProgressBar, gpu: bool, leakage_model: bool) -> int:
        """ Calls all methods needed to calculate subkey for specified cypher text

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param num_poi: number of points of interest to use
        :param spacing_poi: spacing between the points of interest
        :param num_attack_traces: number of data points to attack
        :param bar: the progressbar to update
        :param gpu: flag used to enable gpu.
        :param leakage_model: the leakage model to use
        :return: the calculated subkey
        """
        # Calculate all important data for the attack.
        template_traces_hamming_weight = self.calc_traces_hamming_weight(subkey, leakage_model, num_poi)
        template_means = self.find_average(template_traces_hamming_weight, gpu)
        template_sum_difference = self.find_sum_of_differences(template_means)
        points_of_interest = self.find_points_of_interest(num_poi, spacing_poi, template_sum_difference)
        mean_matrix = self.calc_mean_matrix(num_poi, template_means, points_of_interest)
        covariance_matrix = self.calc_covariance_matrix(num_poi, template_traces_hamming_weight, points_of_interest)

        # Execute the attack.
        return self.execute_attack(subkey, points_of_interest, mean_matrix, covariance_matrix, num_attack_traces, bar)

    def calc_traces_hamming_weight(self, subkey: int, leakage_model: bool,
                                   min_points_of_interest: int = 2) -> List[np.ndarray]:
        """
        Method that calculates intermediate sbox and hamming weight traces.

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param leakage_model: the leakage model to use
        :param min_points_of_interest: minimum amount of points required per class.
        :return: list containing matrix per hamming weight
        """

        # Create the right hamming weight structure.
        template_hamming_weight = HammingWeight.hamming_weights(self.template_plain, self.template_keys, subkey,
                                                                hamming_leakage=leakage_model)

        if leakage_model:
            hamming_weight_size = 9
        else:
            hamming_weight_size = 256

        # Put each trace in the right category and change into NumPy array.
        template_traces_hamming_weight = [[] for _ in range(hamming_weight_size)]

        for i in range(len(self.template_traces)):
            temp = template_hamming_weight[i]
            template_traces_hamming_weight[temp].append(self.template_traces[i])

        template_traces_hamming_weight = [np.array(template_traces_hamming_weight[i])
                                          for i in range(hamming_weight_size)]

        min_length = min_points_of_interest
        for i in range(len(template_traces_hamming_weight)):
            if len(template_traces_hamming_weight[i]) < min_length:
                min_length = len(template_traces_hamming_weight[i])

        if min_length < min_points_of_interest:
            print('ERROR: Not enough values of certain class. Maximum amount of points of interest for this '
                  'dataset is %d' % min_length)
            exit(1)
        return template_traces_hamming_weight

    def find_average(self, template_traces_hamming_weight: List[np.ndarray], gpu: bool) -> np.ndarray:
        """ Method for finding the averages of the traces.

        :param template_traces_hamming_weight: list containing matrix per hamming weight
        :param gpu: whether or not to use gpu for this operation
        :return: matrix containing averages for the traces given the hamming weights
        """

        if gpu:
            # Find an average trace for each weight.
            template_means = torch.zeros((len(template_traces_hamming_weight), len(self.template_traces[0])))
            tthw_t = []
            for i in range(len(template_traces_hamming_weight)):
                tthw_t.append(torch.as_tensor(template_traces_hamming_weight[i]).cuda())
            for i in range(len(template_traces_hamming_weight)):
                template_means[i] = torch.mean(tthw_t[i], dim=0)

            return template_means.cpu().numpy()

        else:
            # Find an average trace for each weight.
            template_means = np.zeros((len(template_traces_hamming_weight), len(self.template_traces[0])))
            for i in range(len(template_traces_hamming_weight)):
                template_means[i] = np.average(template_traces_hamming_weight[i], 0)

        return template_means

    def find_sum_of_differences(self, template_means: np.ndarray) -> np.ndarray:
        """ Method that find the sum of the differences of the traces.

        :param template_means: the means of the traces
        :return: matrix containing sum of differences of the traces
        """

        # Use the sum of differences method to find points of interest.
        template_sum_difference = np.zeros(len(self.template_traces[0]))
        for i in range(len(template_means)):
            for j in range(i):
                template_sum_difference += np.abs(template_means[i] - template_means[j])

        return template_sum_difference

    @staticmethod
    def find_points_of_interest(num_poi: int, spacing_poi: int, template_sum_difference: np.ndarray) -> List[int]:
        """ Method for finding the points of interest of the traces.

        :param num_poi: number of points of interest to use
        :param spacing_poi: spacing between the points of interest
        :param template_sum_difference: matrix containing sum of differences of the traces
        :return: list containing indices of points of interest
        """

        points_of_interest = []

        for i in range(num_poi):
            # Add the biggest sum of difference to the points of interest.
            next_point_of_interest = template_sum_difference.argmax()
            points_of_interest.append(next_point_of_interest)

            # Make sure the next point of interest is the only peak in its spacing.
            min_point_of_interest = max(0, next_point_of_interest - spacing_poi)
            max_point_of_interest = min(next_point_of_interest + spacing_poi, len(template_sum_difference))
            for j in range(min_point_of_interest, max_point_of_interest):
                template_sum_difference[j] = 0

        return points_of_interest

    @staticmethod
    def calc_mean_matrix(num_points_of_interest: int, template_means: np.ndarray, points_of_interest: List[int]):
        """ Method that calculates the mean matrix used in the Template Attack.

        :param num_points_of_interest: number of points of interest to use
        :param template_means: the means of the traces
        :param points_of_interest: list containing indices of points of interest
        :return: mean matrix
        """

        mean_matrix = np.zeros((len(template_means), num_points_of_interest))
        for HW in range(len(template_means)):
            for i in range(num_points_of_interest):
                mean_matrix[HW][i] = template_means[HW][int(points_of_interest[i])]

        return mean_matrix

    @staticmethod
    def calc_covariance_matrix(num_points_of_interest: int, template_traces_hamming_weight: List[np.ndarray],
                               points_of_interest: List[int]):
        """ Method that calculates the covariance matrix used in the Template Attack.

        :param num_points_of_interest: number of points of interest to use
        :param template_traces_hamming_weight: list containing matrix per hamming weight
        :param points_of_interest: list containing indices of points of interest
        :return: covariance matrix
        """

        covariance_matrix = np.zeros((len(template_traces_hamming_weight), num_points_of_interest,
                                      num_points_of_interest))
        for hamming_weight in range(len(template_traces_hamming_weight)):
            for i in range(num_points_of_interest):
                for j in range(num_points_of_interest):
                    x = template_traces_hamming_weight[hamming_weight][:, points_of_interest[i]]
                    y = template_traces_hamming_weight[hamming_weight][:, points_of_interest[j]]
                    covariance_matrix[hamming_weight, i, j] = np.cov(x, y)[0][1]

        return covariance_matrix

    def execute_attack(self, subkey: int, points_of_interest: List[int], mean_matrix: np.ndarray,
                       covariance_matrix: np.ndarray, num_attack_traces: int, bar: progressbar.ProgressBar) -> int:
        """ Method that executes the normal Template Attack after all necessary data has been found and calculated.

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param points_of_interest: points of interest to use
        :param mean_matrix: mean_matrix to use
        :param covariance_matrix: covariance_matrix to use
        :param num_attack_traces: number of data points to attack
        :param bar: the progressbar to update
        :return: the calculated subkey
        """

        # Log the points of interest and mean matrix whilst debugging
        self.log_handler.log_to_debug_file(
            "POI indices for subkey " + str(subkey) + ": " + str(points_of_interest) + "\n")
        self.log_handler.log_to_debug_file(
            "Template Means for subkey " + str(subkey) + ":\n" + np.array2string(mean_matrix) + "\n")

        key_guesses = np.zeros(256)

        for j in range(num_attack_traces):
            # Grab key points and put them in a matrix
            a = [self.attack_traces[j][int(points_of_interest[i])] for i in range(len(points_of_interest))]

            # Test each key
            for k in range(256):
                # Find HW coming out of sbox
                hamming_weight = self.hw[aes.sbox[self.attack_plain[j][subkey] ^ k]]

                # Find p_{k,j}
                rv = multivariate_normal(mean_matrix[hamming_weight], covariance_matrix[hamming_weight])
                p_kj = rv.pdf(a)

                # Add it to running totalSS
                if p_kj != 0:
                    key_guesses[k] += np.log(p_kj)

                bar.update(bar.value + 1)

        # Print the top 5 subkeys whilst we are loghandler
        self.log_handler.log_to_debug_file(self.tabulate_top_n(key_guesses, 5, subkey))

        subkey_result = int(key_guesses.argsort()[-1:])

        # Print the subkey while debugging
        debug_string = \
            "Subkey " + str(subkey) + " is: " + str(subkey_result) + " | POI indices: " + str(points_of_interest)
        self.log_handler.log_to_debug_progressbar(bar, debug_string)
        self.log_handler.log_to_debug_file(self.log_handler.CONST_SEPARATOR)

        return subkey_result

    @staticmethod
    def tabulate_top_n(data: np.ndarray, n: int, subkey: int) -> str:
        """"" This method prints the most occurring key guesses

        :param data: key_guesses of the template attack
        :param n: amount of key guesses to log
        :param subkey: the subkey index which has been analyzed
        :return: log string
        """

        maximum = np.max(data)
        data /= maximum
        parsed_data = [(x, data[x]) for x in reversed(np.argsort(data)[-n:])]
        return tabulate(parsed_data, headers=['Subkey: ' + str(subkey), 'Relative score'])

    def pooled_solve_subkey(self, subkey: int, points_of_interest: List[int], template_means: np.ndarray, num_poi: int,
                            num_attack_traces: int, pooled_covariance_matrix: np.ndarray, bar: progressbar.ProgressBar):
        """ Calls all methods needed to calculate subkey for specified cypher text

        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param points_of_interest: list containing indices of points of interest
        :param template_means: the means of the traces
        :param num_poi: number of points of interest to use
        :param num_attack_traces: number of data points to attack
        :param pooled_covariance_matrix: covariance_matrix to use
        :param bar: the progressbar to update
        :return: the calculated subkey
        """
        # Calculate all important data for the attack, most of it is already calculated whilst calculating the pooled
        # covariance matrix, thus we pass that information on and don't calculate it again.

        mean_matrix = self.calc_mean_matrix(num_poi, template_means, points_of_interest)

        # Execute the attack.
        return self.execute_attack(subkey, points_of_interest, mean_matrix, pooled_covariance_matrix, num_attack_traces,
                                   bar)

    def pooled_covariance_matrix(self, num_poi: int, spacing_poi: int, gpu: bool, leakage_model: bool):
        """ Calculates the pooled covariance matrix and stores all needed intermediate results for future use

        :param num_poi: number of points of interest to use
        :param spacing_poi: spacing between the points of interest
        :param gpu: whether to use GPU for matrix computations
        :param leakage_model: the leakage model to use
        :return: pooled covariance matrix
        """

        if leakage_model:
            length_leakage_model = 9
        else:
            length_leakage_model = 256

        mean_list = np.zeros((16, length_leakage_model, self.template_traces.shape[1]))
        poi_list = np.zeros((16, num_poi))
        covariance_matrices = np.zeros((16, length_leakage_model, num_poi, num_poi))

        for subkey in range(16):
            template_traces_hamming_weight = self.calc_traces_hamming_weight(subkey, leakage_model, num_poi)
            template_means = self.find_average(template_traces_hamming_weight, gpu)
            template_sum_difference = self.find_sum_of_differences(template_means)
            points_of_interest = self.find_points_of_interest(num_poi, spacing_poi, template_sum_difference)
            covariance_matrix = self.calc_covariance_matrix(num_poi, template_traces_hamming_weight, points_of_interest)

            mean_list[subkey] = template_means
            poi_list[subkey] = points_of_interest
            covariance_matrices[subkey] = covariance_matrix

        if gpu:
            t_c = torch.as_tensor(np.array([i for i in covariance_matrices])).cuda()
            pooled_covariance_matrix = torch.mean(t_c, dim=0).cpu()
        else:
            pooled_covariance_matrix = np.mean(np.array([i for i in covariance_matrices]), axis=0)

        return mean_list, poi_list, pooled_covariance_matrix
