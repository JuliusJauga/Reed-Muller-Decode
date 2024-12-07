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
        # print(
        #     f"Decoding message: {message}"
        # )
        message = self.fast_hadamard_transform(message)
        # print(f"Mesage after fast hadamard: {message}")
        # print(
        #     f"After fast hadamard message: {message}"
        # )
        position, sign = self.find_largest_component_position(message)
        # print(f"Position: {position}, sign: {sign}")

        # print(
        #     f"Position: {position}, Sign: {sign}"
        # )
        try:
            position_in_bits = HadamardTransform.int_to_unpacked_bit_list(position, self.m)
            # print(f"Position in bits: {position_in_bits}")
        except Exception as e:
            # print(
            #     f"Error: {e}"
            # )
            # print("With position: ", position)
            return

        # print(
        #     f"Position in bits: {position_in_bits}"
        # )
        position_in_bits = HadamardTransform.reverse_bit_list(position_in_bits)
        

        position_reversed = position_in_bits
        # print(f"Position reversed1: {position_reversed}")

        while len(position_reversed) < self.m + 1:
            position_reversed.append(0)
        # position_reversed = ''.join(map(str, position_reversed)).ljust(self.m+1, '0')
        if sign == 1:
            position_reversed = [1] + position_reversed[:-1]
        else:
            position_reversed = [0] + position_reversed[:-1]
        # print(f"Position reversed2: {position_reversed}")
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
        if n == 0:
            return np.zeros(length, dtype=np.uint8).tolist()
        
        bit_list = []
        while n > 0:
            bit_list.insert(0, n % 2)
            n = n // 2
        
        
    # Ensure the bit_list length is a multiple of 8 for packing
        # while len(bit_list) % 8 != 0:
            # bit_list.insert(0, 0)  # Pad with leading zeros

        # Convert to numpy array and pack the bits
        bit_array = np.array(bit_list, dtype=np.uint8)

        return bit_array

    @staticmethod
    def reverse_bit_list(bit_list):
        new_bit_list = []
        for bit in reversed(bit_list):
            new_bit_list.append(bit)
        return new_bit_list
    
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