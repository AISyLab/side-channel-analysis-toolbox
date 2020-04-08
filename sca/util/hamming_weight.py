import numpy as np

from sca.util import aes


class HammingWeight:
    """ A utility class to calculate the hamming weights of data."""

    @staticmethod
    def hamming_weights(plaintext: np.ndarray, key: np.ndarray, subkey: int,
                        aes_round: int = 1, aes_operation: int = 0, hamming_leakage: bool = True):
        """ Calculates the hamming weights of all plaintext values for a specific subkey in combination with the key and
        optionally the AES round and operation.

        :param plaintext: the plaintext values.
        :param key: the key the plaintext is to be encrypted with.
        :param subkey: the specific subkey.
        :param aes_round: the AES round to get hamming weights for.
        :param aes_operation: the AES operation to get hamming weights for.
        :param hamming_leakage: whether to use the hamming weight leakage model.
        :returns: the calculated hamming weights.
        """

        if aes_round == 1 and aes_operation == 0:
            # This is the trivial implementation which doesn't require any key expansion or intermediate keys
            return HammingWeight._normal_hw(plaintext, key, subkey, hamming_leakage)
        else:
            return HammingWeight._alternate_hw(plaintext, key, subkey, hamming_leakage, aes_round, aes_operation)

    @staticmethod
    def _normal_hw(plaintext: np.ndarray, key: np.ndarray, subkey: int, hamming_leakage: bool) -> np.ndarray:
        """ The default implementation without AES key expansion.

        :param plaintext: the plaintext values.
        :param key: the key the plaintext is to be encrypted with.
        :param subkey: the specific subkey.
        :param hamming_leakage: whether to use the hamming weight leakage model.
        :return the calculated hamming weights.
        """

        length = len(plaintext)
        hamming_weights = np.empty(length, int)

        for i in range(length):
            index = plaintext[i][subkey] ^ key[i][subkey]
            hamming_weights[i] = aes.sbox[index]

        if hamming_leakage:
            return HammingWeight._hamming_leakage(hamming_weights)

        return hamming_weights

    @staticmethod
    def _alternate_hw(plaintext: np.ndarray, key: np.ndarray, subkey: int, hamming_leakage: bool,
                      aes_round: int, aes_operation: int):
        """ The implementation that uses key expansion and intermediate AES keys for different round or operations.

        :param plaintext: the plaintext values.
        :param key: the key the plaintext is to be encrypted with.
        :param subkey: the specific subkey.
        :param aes_round: the AES round to get hamming weights for.
        :param aes_operation: the AES operation to get hamming weights for.
        :param hamming_leakage: whether to use the hamming weight leakage model.
        :returns: the calculated hamming weights.
        """

        length = len(plaintext)
        hamming_weights = np.empty(length, int)
        aes_object = aes.AES(key[0])

        for i in range(length):
            if i <= 1 or not np.array_equal(key[i], key[i - 1]):
                full_key = key[i]
                aes_object.change_key(full_key)

            # single_byte=True asserts that `encrypt` returns an int instead of an ndarray
            hamming_weights[i] = aes_object.encrypt(plaintext[i], aes_round, aes_operation,
                                                    single_byte=True, result_byte=subkey)

        if hamming_leakage:
            return HammingWeight._hamming_leakage(hamming_weights)

        return hamming_weights

    @staticmethod
    def bitwise_hamming(plaintext: np.ndarray, key: np.ndarray, subkey: int,
                        aes_round: int = 1, aes_operation: int = 0, bit: int = 0):
        """ Calculates the hamming weights of all plaintext values for a specific subkey in combination with the key and
        optionally the AES round and operation.

        :param plaintext: the plaintext values.
        :param key: the key the plaintext is to be encrypted with.
        :param subkey: the specific subkey.
        :param aes_round: the AES round to get hamming weights for.
        :param aes_operation: the AES operation to get hamming weights for.
        :param bit: which bit to select.
        :returns: the calculated hamming weights.
        """

        if aes_round == 1 and aes_operation == 0:
            # This is the trivial implementation which doesn't require any key expansion or intermediate keys
            num = HammingWeight._normal_hw(plaintext, key, subkey, False)
        else:
            num = HammingWeight._alternate_hw(plaintext, key, subkey, False, aes_round, aes_operation)
        return HammingWeight._bit_select(num, bit)

    @staticmethod
    def _bit_select(hamming_weights: np.ndarray, bit: int) -> np.ndarray:
        """ Returns on bit.

        :param hamming_weights: the hamming weights whose bits should be counted.
        :param bit: which bit to select.
        :returns: the counted bits.
        """

        return np.vectorize(lambda hw: format(hw, "08b")[bit].count('1'))(hamming_weights)

    @staticmethod
    def _hamming_leakage(hamming_weights: np.ndarray) -> np.ndarray:
        """ Counts the number of ones in each of the hamming weights and returns those values.

        :param hamming_weights: the hamming weights whose bits should be counted.
        :returns: the counted bits.
        """

        return np.vectorize(lambda hw: bin(hw).split('b')[1].count('1'))(hamming_weights)
