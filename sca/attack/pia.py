import numpy as np
from pyitlib import discrete_random_variable as drv
import progressbar
from sca.loghandler.loghandler import LogHandler
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
from sca.util import progress_bar_util
import warnings
from typing import List, Tuple
from scipy.interpolate import interp1d


class Pia:
    """" This class contains methods for performing a PIA """

    KEY_SIZE = 256

    @staticmethod
    def run(traces: np.ndarray, plains: np.ndarray, keys: np.ndarray, attack_traces: np.ndarray, subkey: int,
            debug_mode_enabled: bool = False) -> List[int]:

        """ The run method of pia

        :param traces: the traces to use
        :param plains: the plaintexts to use
        :param keys: the keys to use
        :param attack_traces: the traces to use for attacking
        :param subkey: the subkey index to analyze. Must be in the range [0-15].
        :param debug_mode_enabled: whether to enable debug mode
        :return: the calculated subkey corresponding to the subkey index specified
        """

        print("Executing Perceived Information Analysis")

        bar = progressbar.ProgressBar(max_value=len(attack_traces[0]) * (16 if subkey == 16 else 1),
                                      widgets=progress_bar_util.get_widgets(debug_mode_enabled))
        warnings.simplefilter("ignore", category=RuntimeWarning)

        perceived_information = [0] * len(attack_traces[0])
        max_pia = -float("inf")

        bar.start()

        indices = [subkey]
        if subkey == 16:
            indices = range(subkey)

        for i in indices:

            subkeys = [0] * len(keys)

            for j in range(len(keys)):
                subkeys[j] = int(keys[j][i])

            dummy_interp1d = interp1d(range(2), range(2))

            leakage_per_byte_value_matrix = [list() for _ in range(Pia.KEY_SIZE)]
            model_sampled_pdf_per_byte_value_array = np.array([dummy_interp1d for _ in range(Pia.KEY_SIZE)])

            for j in range(len(traces[0])):
                for k in range(len(traces)):
                    key = subkeys[k]
                    plain = plains[k][i]
                    byte = key ^ plain
                    leakage_per_byte_value_matrix[byte].append(traces[k][j])

            for j in range(len(leakage_per_byte_value_matrix)):
                model_mu, model_std = norm.fit(leakage_per_byte_value_matrix[j])
                model_sampled_pdf_per_byte_value_array[j] = Pia.sample_pdf(model_mu, model_std, 10)

            scaling_factor = 1.0 / (len(traces) * len(attack_traces))

            for j in range(len(attack_traces[0])):
                column = attack_traces[:, j]

                chip_mu, chip_std = norm.fit(column)
                chip_sampled_pdf = Pia.sample_pdf(chip_mu, chip_std, 10)

                pia = 0
                for cell in column:
                    for k in range(Pia.KEY_SIZE):
                        model_sampled_pdf = model_sampled_pdf_per_byte_value_array[k]

                        if not np.isclose(chip_std, 0.0):

                            model_probability = model_sampled_pdf(cell)
                            chip_probability = chip_sampled_pdf(cell)

                            # The sampling sometimes returns a negative probability. Correct this.
                            if model_probability <= 0.0:
                                model_probability = 0.000001

                            if chip_probability <= 0.0:
                                chip_probability = 0.0

                            pia += chip_probability * math.log2(model_probability)

                pia *= scaling_factor

                _, bin_edges = np.histogram(column, bins='auto')
                bin_values = np.digitize(column, bin_edges)

                o = np.array(bin_values, dtype=int)
                shannon_entropy = drv.entropy(o)

                pia = shannon_entropy - pia

                perceived_information[j] += pia

                if bar.value < bar.max_value:
                    bar.update(bar.value + 1)

                if pia > max_pia:
                    max_pia = pia

        for i in range(len(perceived_information)):
            perceived_information[i] /= max_pia * (16 if subkey == 16 else 1)

        bar.finish()
        plt.plot(perceived_information)
        plt.show()

        warnings.simplefilter("default")

        print("Done!")

        return perceived_information

    @staticmethod
    def sample_pdf(mu: float, std: float, number_of_points: int, outer_bounds: Tuple = (-1, 1)) -> interp1d:
        """

        Samples a normal distribution quadratically.
        Assumes the distribution is largely within the domain x:[outer_bounds_0, outer_bounds_1]

        :param mu: The mean of the normal distribution
        :param std: The standard deviation of the normal distribution
        :param number_of_points: The amount of points to sample with
        :param outer_bounds: The bounds to stop sampling, after these the pdf returns 0.
        :return: The interpolated normal distribution
        """
        distribution = norm(mu, std)

        if np.isclose(0.0, std):
            std = mu / 3

        x = np.linspace(mu - 4 * std, mu + 4 * std, num=number_of_points, endpoint=True)
        y = np.zeros(number_of_points)

        for l in range(math.ceil(len(y) / 2)):
            y[l] = distribution.pdf(x[l])

        reversed_y = np.flip(y)

        if number_of_points % 2 == 1:
            np.delete(reversed_y, obj=0, axis=0)

        # After sampling half of the distribution copy it over to the other side, as normal distributions are
        # symmetric.

        for l in range(math.ceil(len(y) / 2), len(y)):
            y[l] = reversed_y[l]

        # Put bounds to the normal distribution
        negative_bound, positive_bound = outer_bounds

        x[0] = negative_bound
        x[number_of_points - 1] = positive_bound

        y[0] = 0.000001
        y[number_of_points - 1] = 0.000001

        return interp1d(x, y, kind='quadratic', copy=False)
