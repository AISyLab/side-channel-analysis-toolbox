import numpy as np
import warnings

from sca.util.hamming_weight import HammingWeight

CONST_PLAIN_TEXT_LENGTH = 256


class NICV:
    """"
    This class contains methods for performing a NICV (Normalized
    Inter-Class Variance) for accelerating side-channel attacks.
    """

    @staticmethod
    def run(traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, hamming_weight: bool = True) -> np.ndarray:
        """ Runs a full NICV analysis

        :param traces: the traces to use
        :param keys: the keys to use
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param plain: the plaintexts to use
        """

        print('Performing NICV analysis...')

        # Analyze the sub byte specified by the user
        # TODO: This should not be hardcoded but be passed via the CLI
        points_of_interest = NICV.get_points_of_interest_indices(plain, traces, 10, 0, keys, hamming_weight)

        # Output the points of interest to a file.
        np.save("out/points_of_interest", points_of_interest)

        return points_of_interest

    @staticmethod
    def calculate_mean_x_given_y_matrix(
            plain_texts: np.ndarray, traces: np.ndarray,
            sub_byte_index_to_analyze: int, keys: np.ndarray,
            hamming_weight: bool = True, aes_round: int = 1,
            aes_operation: int = 0) -> np.ndarray:
        """Mean_y_given_x means that we calculate for every possible sub byte x
        and a given timestamp the mean of y at that point
        x specifies a sub_byte in the plain_text, y specifies an output trace

        :param plain_texts: the plaintexts to use
        :param traces: the traces to use
        :param keys: the keys to use
        :param sub_byte_index_to_analyze: the subkey index to analyze. Must be in the range [0-15]
        :param hamming_weight: whether to use the hamming_weight leakage model
        :param aes_round; the round of aes to analyze
        :param aes_operation: the operation of aes to analyze
        :return: a numpy matrix containing means for every sub byte
        """

        length_trace = len(traces[0])
        hamming_weights = HammingWeight.hamming_weights(plain_texts, keys, sub_byte_index_to_analyze,
                                                        aes_round, aes_operation, hamming_weight)
        if hamming_weight:
            amount_of_values = 9
        else:
            amount_of_values = 256

        mean_y_given_x_matrix = np.empty((amount_of_values, length_trace))

        for i in range(amount_of_values):

            traces_with_x = []

            # Filter out the traces which contain the current sub byte
            for j in range(len(plain_texts)):
                if hamming_weights[j] == i:
                    traces_with_x.append(j)

            # Now calculate the mean of this specific x for every given timestamp
            for j in range(length_trace):
                list_values = np.empty((len(traces_with_x)))
                for k in range(len(traces_with_x)):
                    list_values[k] = traces[traces_with_x[k]][j]
                if len(list_values) > 0:
                    mean_y_given_x_matrix[i][j] = np.mean(list_values)
                # If there are no traces which contain the following x set the mean at 0, this will not happen
                # if the trace size is much bigger than 255
                # TODO: maybe find a more sophisticated solution to this problem
                else:
                    mean_y_given_x_matrix[i][j] = 0

        return mean_y_given_x_matrix

    @staticmethod
    def calculate_single_nicv(mean_x_given_y: np.ndarray, y: np.ndarray) -> int:
        """ Calculates the nicv, which is given by the formula nicv = var(mean(X | Y)) / var(Y)
        mean_x_given_y should be an array of means of x given a certain timestamp, y should be the possible values of y
        given that certain timestamp.

        :param mean_x_given_y: the mean of the sub byte given a certain timestamp
        :param y: the values y can take on given a certain timestamp
        :return: the Normalised Interclass Variance(NICV)

        """
        return np.var(mean_x_given_y) / np.var(y)

    @staticmethod
    def calculate_nicv_array(plain_texts: np.ndarray, traces: np.ndarray,
                             sub_byte_index_to_analyze: int, keys: np.ndarray,
                             hamming_weight: bool = True) -> np.ndarray:
        """ This method calculates the nicv for every possible timestamp and outputs it to a ndarray
        :param plain_texts: the plaintexts to use
        :param traces: the traces to use
        :param sub_byte_index_to_analyze: The subkey to analyze. Must be in the range [0-15]
        :param keys: keys to use
        :param hamming_weight: whether to use the hamming_weight leakage model
        :return: a matrix containing NICV values for every feature/timestamp

        """

        length_trace = len(traces[0])
        mean_y_given_x_matrix = NICV.calculate_mean_x_given_y_matrix(plain_texts, traces,
                                                                     sub_byte_index_to_analyze, keys, hamming_weight)

        # Now that there is a matrix for every sub byte and every timestamp the mean of the trace value
        # The nicv can be calculated, this is done for every timestamp.
        nicv_values_over_time = np.empty(length_trace)

        warnings.simplefilter("ignore", category=RuntimeWarning)

        for i in range(length_trace):
            nicv_values_over_time[i] = NICV.calculate_single_nicv(mean_y_given_x_matrix[:, i], traces[:, i])

        warnings.simplefilter("default")
        return nicv_values_over_time

    @staticmethod
    def get_points_of_interest_indices(plain_texts: np.ndarray, traces: np.ndarray, amount_of_points: int,
                                       sub_byte_index_to_analyze: int, keys: np.ndarray,
                                       hamming_weight: bool = True) -> np.ndarray:
        """
        Method which returns a certain amount of points of interest,
        traces which have the highest nicv value and thus are probably the most relevant to analyze.

        :param plain_texts: the plaintexts to use
        :param traces: the traces to use
        :param amount_of_points: The amount of point of interests needed
        :param sub_byte_index_to_analyze: The subkey to analyze. Must be in the range [0-15]
        :param keys: keys to use
        :param hamming_weight: whether to use the hamming_weight leakage model
        :return: matrix containing the most relevant indices of the traces file
        """

        # Get the nicv values
        nicv_values_over_time = NICV.calculate_nicv_array(plain_texts, traces,
                                                          sub_byte_index_to_analyze, keys, hamming_weight)
        points_of_interest_indices = np.argsort(-nicv_values_over_time)[:amount_of_points][::-1]

        return points_of_interest_indices

    @staticmethod
    def get_points_of_interest(plain_texts: np.ndarray, traces: np.ndarray,
                               amount_of_points: int, sub_byte_index_to_analyze: int,
                               keys: np.ndarray, hamming_weight: bool = True) -> np.ndarray:
        """
        Method which returns a certain amount of points of interest indices,
        traces which have the highest nicv value and thus are probably the most relevant to analyze.

        :param plain_texts: the plaintexts to use
        :param traces: the traces to use
        :param amount_of_points: The amount of point of interests needed
        :param sub_byte_index_to_analyze: The subkey to analyze. Must be in the range [0-15]
        :param keys: keys to use
        :param hamming_weight: whether to use the hamming_weight leakage model
        :return: matrix containing the most relevant traces of the traces file
        """

        indices = NICV.get_points_of_interest_indices(plain_texts, traces, amount_of_points,
                                                      sub_byte_index_to_analyze, keys, hamming_weight)
        return traces[:, indices]
