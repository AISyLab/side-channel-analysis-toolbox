import numpy as np
from sca.util import progress_bar_util
from sca.loghandler.loghandler import LogHandler
from sca.util.hamming_weight import HammingWeight
import progressbar


class DPA:
    """" This class contains methods for performing a DPA (differential) side-channel attack."""

    RANGE = 5000
    OFFSET = 0
    NUMBER_OF_TRACES = 10000
    KEY_SIZE = 256

    def __init__(self, traces: np.ndarray, keys: np.ndarray, plain: np.ndarray, bit: int):
        """ Constructor for mia class

        :param traces: the traces to use
        :param keys: the keys to use
        :param plain: the plaintexts to use
        :param bit: which bit to use.
        """
        self.traces = traces
        self.plain = plain
        self.keys = keys
        self.bit = bit
        self.NUMBER_OF_TRACES = self.traces.shape[0]
        self.RANGE = self.traces.shape[1]

        self.log_handler = LogHandler("dpa", False)

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
    def run(attack_traces: np.ndarray, attack_keys: np.ndarray, attack_plain: np.ndarray, round_: int, operation: int,
            subkey: int, offset: int = 15, single_order: bool = False,
            debug_mode_enabled: bool = False, bit: int = 0) -> np.ndarray:

        """ The run method of dpa

        :param attack_traces: the traces to use for attacking
        :param attack_keys: the keys to use for attacking
        :param attack_plain: the plaintexts to use for attacking
        :param round_: the AES round to attack
        :param operation: the AES operation to attack, represented as an integer from 0 to 3
        :param subkey: the subkey index to analyze. Must be in the range [0-15].
        :param offset: offset to use for dpa.
        :param single_order: only use single order.
        :param debug_mode_enabled: whether to enable debug mode
        :param bit: which bit to select.
        :return: the calculated subkey corresponding to the subkey index specified
        """

        print("Executing Differential Power Analysis")

        result = np.zeros(16, dtype=int)
        if single_order:
            offset = 0
        # Init progress bar with amount of iterations.
        if subkey == 16:

            bar = progressbar.ProgressBar(max_value=16 * (255),
                                          widgets=progress_bar_util.get_widgets(debug_mode_enabled))
            for i in range(16):
                dpa = DPA(attack_traces, attack_keys, attack_plain, bit)
                dpa.set_offset(offset)
                result[i] = dpa.solve_subkey(i, debug_mode_enabled, round_=round_,
                                             operation=operation, bar=bar)

        else:
            bar = progressbar.ProgressBar(max_value=255,
                                          widgets=progress_bar_util.get_widgets(debug_mode_enabled))

            dpa = DPA(attack_traces, attack_keys, attack_plain, bit)
            dpa.set_offset(offset)
            result[subkey] = dpa.solve_subkey(subkey, debug_mode_enabled,
                                              round_=round_, operation=operation, bar=bar)
        bar.finish()
        print('The final key is: ', result)
        return result

    def solve_subkey(self, subkey: int,
                     debug_mode_enabled: bool = False, round_: int = 1, operation: int = 0,
                     bar: progressbar.ProgressBar = None):
        """ Solve a subkey
        :param subkey: the subkey index to analyze. Must be in the range [0-15]
        :param debug_mode_enabled: whether to enable debug mode
        :param round_: round of aes.
        :param operation: operation of aes.
        :param bar: the progressbar to update
        :return: the calculated subkey corresponding to the subkey index specified
        """

        log_handler = LogHandler("mia", debug_mode_enabled)

        log_handler.log_to_debug_file('RUNNING dpa attack with \n'
                                      'NUM_FEATURES: {} \n'
                                      'ATTACK_TRACES: {} \n'
                                      'SUBKEY: {}.'
                                      .format(self.RANGE, self.NUMBER_OF_TRACES, subkey))
        differential = np.zeros(self.KEY_SIZE)
        means = np.zeros((self.RANGE, 2))

        for key in range(self.KEY_SIZE):

            self.keys[:, subkey] = key
            leakage_table = HammingWeight.bitwise_hamming(self.plain, self.keys, subkey, round_,
                                                          operation, self.bit)

            list1 = []
            list0 = []
            for i in range(self.NUMBER_OF_TRACES):
                if leakage_table[i] == 1:
                    list1.append(i)
                else:
                    list0.append(i)
            if self.OFFSET == 0:
                for i in range(self.RANGE):
                    means[i][0] = np.mean(self.traces[list0, i])
                    means[i][1] = np.mean(self.traces[list1, i])
            else:
                for i in range(self.RANGE - self.OFFSET):
                    means[i][0] = np.mean(np.abs(self.traces[list0, i] - self.traces[list0, i + self.OFFSET]))
                    means[i][1] = np.mean(np.abs(self.traces[list1, i] - self.traces[list1, i + self.OFFSET]))
            temp = np.zeros(self.RANGE - self.OFFSET)
            for i in range(self.RANGE - self.OFFSET):
                temp[i] = np.abs(means[i][1] - means[i][0])
            differential[key] = np.max(temp)
            if bar is not None and bar.value < bar.max_value:
                bar.update(bar.value + 1)

        return np.argmax(differential)
