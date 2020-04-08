import numpy as np
import warnings
import matplotlib.pyplot as plt


class SNR:
    """
    This class implements signal to noise feature selection, usable in the different attacks of this package.
    """

    @staticmethod
    def run(traces: np.ndarray, num_features: int) -> np.ndarray:
        """
        The run method for SNR.

        :param traces: the traces to be used in the SNR.
        :param num_features: the amount of features to be returned by SNR.
        :return: the signal to noise ratio to be used for an attack.
        """
        result = np.abs(SNR.signaltonoise(traces))
        warnings.simplefilter("default")
        plt.plot(result)
        plt.show()
        return np.argsort(result)[: num_features]

    @staticmethod
    def signaltonoise(a: np.ndarray, axis: int = 0, ddof: int = 0) -> np.ndarray:
        """
        Helper method for SNR.

        :param a: the array to be used.
        :param axis: axis along which the mean is to be computed. By default axis = 0.
        :param ddof: degree of freedom correction for Standard Deviation.
        :return:
        """
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)
