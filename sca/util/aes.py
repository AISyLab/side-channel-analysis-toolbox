""" Code adapted for use in our project from https://github.com/bozhu/AES-Python/blob/master/aes.py"""

import numpy as np

CONST_NUMBER_ROUNDS = 10

""" Substitution box used for encryption."""
sbox = (
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
)

""" Substitution box used for decryption."""
inv_sbox = (
    0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
    0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
    0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
    0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
    0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
    0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
    0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
    0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
    0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
    0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
    0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
    0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
    0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
    0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
    0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D,
)

""" Rcon array used for key expansion."""
rcon = (
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40,
    0x80, 0x1B, 0x36, 0x6C, 0xD8, 0xAB, 0x4D, 0x9A,
    0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A,
    0xD4, 0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39,
)


class AES:

    def __init__(self, key: np.ndarray):
        """ Constructor for AES class.

        :param key: the key to use in encryption.
        """

        self.keys = np.zeros((4, 4 * 11), dtype=int)
        self.change_key(key)
        self.state = np.zeros((4, 4), dtype=int)

    def change_key(self, key: np.ndarray):
        """ Takes Key and expands it as described in section 3.6 of Design of Rijndael.

        :param key: the key to use in encryption.
        """

        self.keys = np.zeros((4, 4 * 11), dtype=int)
        for i in range(4):
            for j in range(4):
                self.keys[j][i] = key[i * 4 + j]
        for i in range(4, 11 * 4):
            if i % 4 == 0:
                self.keys[0][i] = self.keys[0][i - 4] ^ sbox[self.keys[1][i - 1]] ^ rcon[i // 4]
                for j in range(1, 4):
                    self.keys[j][i] = self.keys[j][i - 4] ^ sbox[self.keys[(j + 1) % 4][i - 1]]
            else:
                for j in range(4):
                    self.keys[j][i] = self.keys[j][i - 4] ^ self.keys[j][i - 1]

    @staticmethod
    def xtime(a: int):
        """ Code taken from https://github.com/bozhu/AES-Python/blob/master/aes.py, method used for mixcolumns.

        :param a: index to shift.
        :return: index to shift to.
        """
        return (((a << 1) ^ 0x1B) & 0xFF) if (a & 0x80) else (a << 1)

    def encrypt(self, plain: np.ndarray, round_output: int = 11, operation: int = 0, single_byte: bool = False,
                result_byte: int = 0):
        """ Encrypts given plaintext with the current key.

         :param plain: the plaintexts to use.
         :param round_output: round for which to return result, defaults to the last round.
         :param operation: operation on which to return result, default is 1, shift rows.
         :param single_byte: if set to true returns single byte.
         :param result_byte: result byte is byte to return, must be in range [0-15], default is 0.
         :return: encrypted plaintext.
         """

        self.state = np.zeros((4, 4), dtype=int)

        for i in range(4):
            for j in range(4):
                self.state[j][i] = plain[i * 4 + j]

        self.add_round_key(0)
        for i in range(1, CONST_NUMBER_ROUNDS):
            if round_output == i:
                self.subst()
                if operation == 0:
                    return self.output(single_byte, result_byte)
                self.shift_rows()
                if operation == 1:
                    return self.output(single_byte, result_byte)
                self.mix_columns()
                if operation == 2:
                    return self.output(single_byte, result_byte)
                self.add_round_key(i)
                if operation == 3:
                    return self.output(single_byte, result_byte)
            self.subst()
            self.shift_rows()
            self.mix_columns()
            self.add_round_key(i)

        self.subst()
        self.shift_rows()
        self.add_round_key(10)

        return self.output(single_byte, result_byte)

    def output(self, single_byte: bool = False, result_byte: int = 0):
        """ Outputs selected byte(s) from state array.

        :param single_byte: if set to true returns single byte.
        :param result_byte: result byte is byte to use, must be in range [0-15], default is 0.
        :return: output of state array.
        """
        if single_byte:
            return self.state[result_byte % 4][result_byte // 4]
        else:
            output = np.zeros(16, dtype=int)
            for i in range(4):
                for j in range(4):
                    output[i * 4 + j] = self.state[j][i]

            return output

    def add_round_key(self, r: int):
        """ Adds round key for round r to state array.

        :param r: round to add key to
        """

        for i in range(4):
            for j in range(4):
                self.state[j][i] = self.state[j][i] ^ self.keys[j][i + 4 * r]

    def subst(self):
        """ Does substitution phase of AES."""

        for i in range(4):
            for j in range(4):
                self.state[j][i] = sbox[self.state[j][i]]

    def inv_subst(self):
        """ Does inverse of substitution phase of AES."""

        for i in range(4):
            for j in range(4):
                self.state[j][i] = inv_sbox[self.state[j][i]]

    def shift_rows(self):
        """ Does shift rows phase of AES."""

        self.state[1][0], self.state[1][1], self.state[1][2], self.state[1][3] = \
            self.state[1][1], self.state[1][2], self.state[1][3], self.state[1][0]
        self.state[2][0], self.state[2][1], self.state[2][2], self.state[2][3] = \
            self.state[2][2], self.state[2][3], self.state[2][0], self.state[2][1]
        self.state[3][0], self.state[3][1], self.state[3][2], self.state[3][3] = \
            self.state[3][3], self.state[3][0], self.state[3][1], self.state[3][2]

    def inv_shift_rows(self):
        """ Does inv shift rows phase of AES."""

        self.state[1][1], self.state[1][2], self.state[1][3], self.state[1][0] = \
            self.state[1][0], self.state[1][1], self.state[1][2], self.state[1][3]
        self.state[2][2], self.state[2][3], self.state[2][0], self.state[2][1] = \
            self.state[2][0], self.state[2][1], self.state[2][2], self.state[2][3]
        self.state[3][3], self.state[3][0], self.state[3][1], self.state[3][2] = \
            self.state[3][0], self.state[3][1], self.state[3][2], self.state[3][3]

    def mix_columns(self):
        """ Does mix columns phase of AES.

         Code adapted from https://github.com/bozhu/AES-Python/blob/master/aes.py"""

        for i in range(4):
            t = self.state[0][i] ^ self.state[1][i] ^ self.state[2][i] ^ self.state[3][i]
            u = self.state[0][i]
            self.state[0][i] ^= t ^ AES.xtime(self.state[0][i] ^ self.state[1][i])
            self.state[1][i] ^= t ^ AES.xtime(self.state[1][i] ^ self.state[2][i])
            self.state[2][i] ^= t ^ AES.xtime(self.state[2][i] ^ self.state[3][i])
            self.state[3][i] ^= t ^ AES.xtime(self.state[3][i] ^ u)

    def inv_mix_columns(self):
        """ Does inverse of mix columns phase of AES.

         Code adapted from https://github.com/bozhu/AES-Python/blob/master/aes.py"""

        for i in range(4):
            u = AES.xtime(AES.xtime(self.state[0][i] ^ self.state[2][i]))
            v = AES.xtime(AES.xtime(self.state[1][i] ^ self.state[3][i]))
            self.state[0][i] ^= u
            self.state[1][i] ^= v
            self.state[2][i] ^= u
            self.state[3][i] ^= v
        self.mix_columns()

    def decrypt(self, cipher: np.ndarray, round_output: int = 11, operation: int = 0, single_byte=False, result_byte=0):
        """ Decrypts given cipher text with the current key.
         Round is round for which to return result.
         Operation is operation on which to return result.
         Result byte is byte to return 0-15.
         If single byte is set to true returns single byte.

         :param cipher: cipher text to use.
         :param round_output: round for which to return result, defaults to the last round.
         :param operation: operation on which to return result, default is 1, shift rows.
         :param single_byte: if set to true returns single byte.
         :param result_byte: result byte is byte to return, must be in range [0-15], default is 0.
         :return: decrypted cipher text.
         """

        self.state = np.zeros((4, 4), dtype=int)
        for i in range(4):
            for j in range(4):
                self.state[j][i] = cipher[i * 4 + j]

        self.add_round_key(CONST_NUMBER_ROUNDS)
        self.inv_shift_rows()
        self.inv_subst()

        for i in range(CONST_NUMBER_ROUNDS - 1, 0, -1):
            if round_output == i:
                self.add_round_key(i)
                if operation == 3:
                    return self.output(single_byte, result_byte)
                self.inv_mix_columns()
                if operation == 2:
                    return self.output(single_byte, result_byte)
                self.inv_shift_rows()
                if operation == 1:
                    return self.output(single_byte, result_byte)
                self.add_round_key(i)
                if operation == 0:
                    return self.output(single_byte, result_byte)
            self.add_round_key(i)
            self.inv_mix_columns()
            self.inv_shift_rows()
            self.inv_subst()

        self.add_round_key(0)

        output = np.zeros(16, dtype=int)
        for i in range(4):
            for j in range(4):
                output[i * 4 + j] = self.state[j][i]
        return output
