from .Utility import Utility
from .IDecoder import IDecoder
import numpy as np
import time
# Hadamard Transform class implementing the IDecoder interface
class HadamardTransform(IDecoder):
    def __init__(self, m):
        self.m = m
    
    def change_m(self, m):
        '''
        Change the value of m.
        
        Args:
            m: The new value of m.
        '''
        self.m = m

    def generate_H_i_m(self, i):
        '''
        Generate the H_i_m matrix for a given i.

        Args:
            i: The value of i.

        Returns:
            The H_i_m matrix.
        '''
        I = Utility.generate_unitary_matrix(2 ** (self.m - i))
        H = [[1, 1],[1,-1]]
        H_i_m = Utility.generate_kronecher_product(I, H)
        I = Utility.generate_unitary_matrix(2 ** (i - 1))
        H_i_m = Utility.generate_kronecher_product(H_i_m, I)
        return H_i_m

    def fast_hadamard_transform_old(self, message):
        '''
        Perform the Fast Hadamard Transform on a boolean message of length 2^m.

        Args:
            message: The message to transform.
        
        Returns:
            The transformed message.
        '''
        vector = self.convert_to_pm1(message)
        for i in range(self.m + 1):
            if i == 0:
                continue
            H_i_m = self.generate_H_i_m(i)
            vector = Utility.vector_by_matrix(vector, H_i_m)
        return vector


    def convert_to_pm1(self, message):
        '''
        Convert a message to a message of 1s and -1s.

        Args:
            message: The message to convert.

        Returns:
            The converted message.
        '''
        return [1 if bit == 1 else -1 for bit in message]
    
    def decode(self, message):
        '''
        Decode a message using the Fast Hadamard Transform.

        Args:
            message: The message to decode.

        Returns:
            The decoded message.
        '''
        message = self.fast_hadamard_transform_old(message)
        position, sign = self.find_largest_component_position(message)
        try:
            position_in_bits = HadamardTransform.int_to_unpacked_bit_list(position, self.m)
        except Exception:
            return
        position_in_bits = HadamardTransform.reverse_bit_list(position_in_bits)
        position_reversed = position_in_bits
        while len(position_reversed) < self.m + 1:
            position_reversed.append(0)
        if sign == 1:
            position_reversed = [1] + position_reversed[:-1]
        else:
            position_reversed = [0] + position_reversed[:-1]
        return position_reversed
    
    @staticmethod
    def int_to_unpacked_bit_list(n, length=None):
        '''
        Convert an integer to an unpacked binary list.

        Args:
            n: The integer to convert.
            length: The length of the binary list.
        
        Returns:
            The unpacked binary list.
        '''
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
        bit_array = np.zeros(length, dtype=np.uint8)
        for i in range(length):
            bit_array[length - i - 1] = n % 2
            n = n // 2
        return bit_array

    @staticmethod
    def reverse_bit_list(bit_list):
        '''
        Reverse a bit list. 

        Args:
            bit_list: The bit list to reverse.
        
        Returns:
            The reversed bit list.
        '''
        new_bit_list = []
        for bit in reversed(bit_list):
            new_bit_list.append(bit)
        return new_bit_list
    
    def find_largest_component_position(self, vector):
        '''
        Find the position of the largest component in a vector.

        Args:
            vector: The vector to search.
            
        Returns:
            The position of the largest component and its sign.
        '''
        max_value = abs(vector[0])
        position = 0
        sign = 1 if vector[0] > 0 else -1
        for i in range(1, len(vector)):
            if abs(vector[i]) > max_value:
                max_value = abs(vector[i])
                sign = 1 if vector[i] > 0 else -1
                position = i
        return position, sign
    
    def fast_hadamard_transform_recursive(self, vec, start, end):
        '''
        Perform the Fast Hadamard Transform recursively on a vector.
        
        Args:
            vec: The vector to transform.
            start: The start index of the vector.
            end: The end index of the vector.
        
        Returns:
            The transformed vector.
        '''
        if end - start == 1:  # Base case: single element
            return

        mid = start + (end - start) // 2
        # Transform the first half
        self.fast_hadamard_transform_recursive(vec, start, mid)
        # Transform the second half
        self.fast_hadamard_transform_recursive(vec, mid, end)

        for i in range(start, mid):
            a = vec[i]
            b = vec[i + (mid - start)]
            vec[i] = a + b  # Combine results (sum)
            vec[i + (mid - start)] = a - b  # Combine results (difference)


    def fast_hadamard_transform(self, message, m):
        '''
        Perform the Fast Hadamard Transform on a boolean message of length 2^m.

        Args:
            message: The message to transform.
            m: The value of m.
        
        Returns:
            The transformed message.
        '''
        N = 1 << m  # Length of the vector (2^m)
        vector = self.convert_to_pm1(message)

        # Perform in-place recursive Hadamard Transform
        self.fast_hadamard_transform_recursive(vector, 0, N)

        return vector