from .Utility import Utility
from .IDecoder import IDecoder
import numpy as np

class HadamardTransform(IDecoder):
    def __init__(self, m):
        self.m = m

    def generate_H_i_m(self, i):
        I = Utility.generate_unitary_matrix(2 ** (self.m - i))
        H = [[1, 1],[1,-1]]
        H_i_m = Utility.generate_kronecher_product(I, H)
        I = Utility.generate_unitary_matrix(2 ** (i - 1))
        H_i_m = Utility.generate_kronecher_product(H_i_m, I)
        return H_i_m

    def fast_hadamard_transform(self, message):
        vector = self.convert_to_pm1(message)
        for i in range(self.m + 1):
            if i == 0:
                continue
            H_i_m = self.generate_H_i_m(i)
            vector = Utility.vector_by_matrix(vector, H_i_m)
        return vector
    
    def convert_to_pm1(self, message):
        return [1 if bit == 1 else -1 for bit in message]
    
    def decode(self, message):
        message = self.fast_hadamard_transform(message)
        position, sign = self.find_largest_component_position(message)
        position_in_bits = HadamardTransform.int_to_unpacked_bit_list(position, self.m)
        position_in_bits = HadamardTransform.reverse_bit_list(position_in_bits)
        position_reversed = position_in_bits
        while len(position_reversed) < self.m + 1:
            position_reversed.append(0)
        # position_reversed = ''.join(map(str, position_reversed)).ljust(self.m+1, '0')
        if sign == 1:
            position_reversed = [1] + position_reversed[:-1]
        else:
            position_reversed = [0] + position_reversed[:-1]
        return position_reversed
    
    @staticmethod
    def int_to_unpacked_bit_list(n, length=None):
        if n < 0:
            raise ValueError("n must be greater than or equal to 0")
        if length is not None and length < 0:
            raise ValueError("length must be greater than or equal to 0")
        if length is not None and length < n.bit_length():
            raise ValueError("length must be greater than or equal to n.bit_length()")
        if length is None:
            length = n.bit_length()
        bit_list = np.unpackbits(np.array([n], dtype=np.uint8))
        bit_list = bit_list.tolist()
        while bit_list[0] == 0:
            bit_list = bit_list[1:]
        return bit_list

    @staticmethod
    def reverse_bit_list(bit_list):
        bit_list.reverse()
        return bit_list
    
    def find_largest_component_position(self, vector):
        max_value = abs(vector[0])
        position = 0
        sign = 1 if vector[0] > 0 else -1
        for i in range(1, len(vector)):
            if abs(vector[i]) > max_value:
                max_value = abs(vector[i])
                sign = 1 if vector[i] > 0 else -1
                position = i
        return position, sign