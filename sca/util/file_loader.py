import numpy as np


class FileLoader:
    """ This class handles loading the traces, keys and plaintext required to execute the attacks and analysis tools."""

    @staticmethod
    def load_traces(traces_file: str) -> np.array:
        """ This method loads the traces.

        :param traces_file is the filepath which contains the traces used for making the attack and profiling traces.
        :return returns the traces as a numpy array which were contained in the traces_file.
        """

        try:
            traces = np.load(traces_file)
        except(FileNotFoundError, IOError):
            print("ERROR: File not found or wrong path.")
            exit(1)

        return traces

    @staticmethod
    def load_keys(keys_file: str) -> np.array:
        """ This method loads the keys.

        :param keys_file is the filepath which contains the keys used for making the attack and profiling traces.
        :return returns the keys as a numpy array which were contained in the keys_file.
        """

        try:
            keys = np.load(keys_file)
        except(FileNotFoundError, IOError):
            print("ERROR: File not found or wrong path.")
            exit(1)

        return keys

    @staticmethod
    def load_plain(plain_file: str) -> np.array:
        """ This method loads the keys.

        :param plain_file is the filepath which contains the plain used for making the attack and profiling traces.
        :return returns the plain as a numpy array which were contained in the plain_file.
        """

        try:
            plain = np.load(plain_file)
        except(FileNotFoundError, IOError):
            print("ERROR: File not found or wrong path.")
            exit(1)

        return plain

    @staticmethod
    def main(traces_file: str, keys_file: str, plain_file: str) -> (np.array, np.array, np.array):
        """ This is the main method of this file loader class, used to load traces.

        :param traces_file is the filepath which contains the traces used for making the attack and profiling traces.
        :param keys_file is the filepath which contains the keys used for making the attack and profiling traces.
        :param plain_file is the filepath which contains the plain used for making the attack and profiling traces.
        :returns the traces, keys and plain as a tuple of numpy arrays.
        """

        print("The needed files are being loaded...")

        fl = FileLoader()

        traces = fl.load_traces(traces_file)
        keys = fl.load_keys(keys_file)
        plain = fl.load_plain(plain_file)

        return traces, keys, plain
