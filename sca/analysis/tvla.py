"""
Implementation of the Test Vector Leakage Assessment.
TVLA is defined as "a statistical tool, which can be used to detect information leakage in devices".
This definition is cited from the paper:
    "Test Vector Leakage Assessment Development", 2016.
    Valme R.

This implementation is based on the findings of the following paper:
    "A testing methodology for side­channel resistance validation", n.d.
    Goodwill G., Jun B., Jaffe J., Rohatgi P.
"""

import numpy as np
import warnings


class TVLA:

    THRESHOLD = 4.5

    @staticmethod
    def run(traces: np.ndarray, output_file=None, threshold=THRESHOLD):
        leakages_indices = TVLA.tvla(traces, threshold)

        print(str(traces.shape[1]) + " features analysed")
        print(str(len(leakages_indices)) + " leaking features found")
        print(str((len(leakages_indices)/traces.shape[1])*100) + "% of features leak")

        if output_file is not None:
            np.save(output_file, traces[:, leakages_indices])

    @staticmethod
    def tvla(traces: np.ndarray, threshold: float):
        """
        Given some traces, the Test Vector Leakage Assessment can show whether or not
        the device that has been running the traces is sensitive to leakage. Initially,
        the method returns only True or False, but a plot is shown, to indicate how much
        leakage there is. In the plot, if the red line is exceeded, it means there is
        leakage.
        :param traces: np.ndarray        Is the path to the numpy object that contains the
                                         power traces of a machine running AES.
        :param threshold:   int          Define a custom threshold. Default is 4.5
        :return:       bool              True for leakage, and False for no leakage
        """

        warnings.simplefilter("ignore", category=RuntimeWarning)

        # Store the traces and plain data in numpy matrices and their sizes
        num_point = traces.shape[1]

        # Binary random vector expresses whether a query is demanded(if the flag is 1)
        flag = np.random.randint(2, size=(traces.shape[0], num_point))
        # Index of the 'fixed plaintext' traces:
        fixed_index = flag == 1
        # Index of the 'random plaintext' traces:
        random_index = flag == 0
        # Number of queries with fixed class:
        fixed_size = fixed_index.sum()
        # Number of queries with random class:
        random_size = random_index.sum()

        # print(fixed_size)
        # print(random_size)

        # Sample mean of the 'fixed plaintext' traces
        mean_fixed = np.mean(traces[fixed_index[:, 0], :], axis=0)
        # Sample mean of the 'random plaintext' traces
        mean_random = np.mean(traces[random_index[:, 0], :], axis=0)
        # Sample variance of the 'fixed plaintext' traces
        variance_fixed = np.var(traces[fixed_index[:, 0], :], axis=0)
        # Sample variance of the 'random plaintext' traces
        variance_random = np.var(traces[random_index[:, 0], :], axis=0)
        # The formula for Welch's t-statistic:
        t_test = (mean_fixed - mean_random) / np.sqrt(variance_fixed / fixed_size +
                                                      variance_random / random_size)

        t_test_result = np.array(np.abs(t_test[0:num_point]))

        filtered_result = np.where(t_test_result > threshold)
        return filtered_result[0]
