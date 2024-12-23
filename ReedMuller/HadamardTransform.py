from .Utility import Utility
from .IDecoder import IDecoder
import numpy as np
import time
import threading
# Hadamard Transform class implementing the IDecoder interface
class HadamardTransform(IDecoder):
    def __init__(self, m):
        self.m = m
        self.Hadamard_matrices = None
        self.generate_Hadamard_matrices()

    def change_m(self, m):
        '''
        Change the value of m.
        
        Args:
            m: The new value of m.
        '''
        self.m = m
        self.generate_Hadamard_matrices()

    def generate_Hadamard_matrices(self):
        '''
        Generate the Hadamard matrices for all values of i.

        Returns:
            None
        '''
        self.Hadamard_matrices = [None] * (self.m + 1)
        for i in range(1, self.m + 1):
            self.Hadamard_matrices[i] = self.generate_H_i_m(i)
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
        I.clear()
        H.clear()
        return H_i_m

    def fast_hadamard_transform(self, message):
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
            H_i_m = self.Hadamard_matrices[i]
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
        message = self.fast_hadamard_transform(message)
        position, sign = self.find_largest_component_position(message)
        try:
            position_in_bits = HadamardTransform.int_to_bit_list(position, self.m)
        except Exception:
            return
        position_in_bits = HadamardTransform.reverse_bit_list(position_in_bits)
        position_reversed = position_in_bits
        if sign == 1:
            position_reversed = [1] + position_reversed
        else:
            position_reversed = [0] + position_reversed
        return position_reversed
    
    @staticmethod
    def int_to_bit_list(n, length=None):
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
        if length is None:
            length = max(1, n.bit_length())
        elif length < n.bit_length():
            raise ValueError("length must be greater than or equal to n.bit_length()")
        return [int(bit) for bit in bin(n)[2:].zfill(length)]

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