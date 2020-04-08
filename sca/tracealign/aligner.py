import numpy as np
from sca.util import progress_bar_util

import progressbar
import fastdtw
import scipy.signal


class Aligner:
    """" This class contains methods for aligning traces."""

    @staticmethod
    def run(aligned: np.ndarray, unaligned: np.ndarray, algorithm: int, output: str, debug_mode_enabled: bool):
        """Aligns traces with specified algorithm and outputs an aligned file

        :param aligned: File of 1 or more aligned traces.
        :param unaligned: File of unaligned traces
        :param algorithm: algorithm to use
        :param output: filename to output to.
        :param debug_mode_enabled: debug mode flag.
        :return:
        """

        print("Executing trace alignment.")

        bar = progressbar.ProgressBar(max_value=len(unaligned),
                                      widgets=progress_bar_util.get_widgets(debug_mode_enabled))
        if algorithm == 0:
            outputarray = Aligner.dtw(aligned, unaligned, bar)
            np.save(output, outputarray)
        elif algorithm == 1:
            outputarray = Aligner.fft(aligned, unaligned, bar)
            np.save(output, outputarray)
        bar.finish()
        return 0

    @staticmethod
    def dtw(aligned: np.ndarray, unaligned: np.ndarray, bar: progressbar.ProgressBar):
        """ Aligns 2 arrays using fastdtw algorithm.

        :param aligned: File of 1 or more aligned traces.
        :param unaligned: File of unaligned traces
        :param bar: The progressbar
        :return:
        """

        aligned2 = np.zeros(((len(aligned) + len(unaligned)), len(aligned[0])))
        for i in range(len(aligned)):
            aligned2[i] = aligned[i]
        for i in range(len(unaligned)):
            distance, alignment = fastdtw.fastdtw(aligned[0], unaligned[i])
            new_alignment = np.zeros(len(aligned[0]))
            bar.update(bar.value + 1)
            k = 0
            for j in range(len(alignment)):
                if alignment[j][0] != k:
                    j += 1
                else:
                    new_alignment[k] = unaligned[i][alignment[j][1]]
                    k += 1
            aligned2[i + len(aligned)] = new_alignment
        return aligned2

    @staticmethod
    def fft(aligned: np.ndarray, unaligned: np.ndarray, bar: progressbar.ProgressBar):
        """ Aligns 2 arrays using fastdtw algorithm.

        :param aligned: File of 1 or more aligned traces.
        :param unaligned: File of unaligned traces
        :param bar: The progressbar
        :return:
        """

        aligned2 = np.zeros(((len(aligned) + len(unaligned)), len(aligned[0])))
        for i in range(len(aligned)):
            aligned2[i] = aligned[i]

        for i in range(len(unaligned)):
            alignment = scipy.signal.fftconvolve(aligned[0], unaligned[i])
            offset = np.argmax(alignment) - len(aligned[0])
            bar.update(bar.value + 1)
            new_alignment = np.zeros(len(aligned[0]))
            for j in range(len(aligned[0])):
                if not (j + offset < 0 or j + offset >= len(aligned[0])):
                    new_alignment[j] = unaligned[i][j + offset]

            aligned2[i + len(aligned)] = new_alignment
        return aligned2
