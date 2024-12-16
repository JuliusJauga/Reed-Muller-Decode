import numpy as np
from concurrent.futures import ThreadPoolExecutor


# Utility class for common functions used in the Reed-Muller and Hadamard Transform classes, such as matrix operations or bit manipulation
class Utility:
    @staticmethod
    def vector_by_matrix_mod2(vector, matrix):
        '''
        Multiply a vector by a matrix modulo 2.

        Args:
            vector: The vector to multiply.
            matrix: The matrix to multiply.
        
        Returns:
            The result of the multiplication.
        '''
        result = []
        matrix_length = len(matrix)

        for j in range(len(matrix[0])):
            column = [matrix[i][j] for i in range(matrix_length)]
            result.append(Utility.dot_product_mod2(vector, column))
        return result 

    @staticmethod
    def dot_product_mod2(v1, v2):
        '''
        Calculate the dot product of two vectors modulo 2.
        
        Args:
            v1: The first vector.
            v2: The second vector.
        
        Returns:
            The result of the dot product modulo 2.
        '''
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length")
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result % 2

    @staticmethod
    def generate_unitary_matrix(n):
        '''
        Generate a unitary matrix of size n.

        Args:
            n: The size of the matrix.
        
        Returns:
            The unitary matrix.
        '''
        if n < 1:
            raise ValueError("n must be greater than 0")
        matrix_columns = n
        matrix_rows = n

        matrix = [[0] * matrix_columns for _ in range(matrix_rows)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1
        return matrix
    
    @staticmethod
    def generate_kronecher_product(A, B):
        '''
        Generate the Kronecker product of two matrices.

        Args:
            A: The first matrix.
            B: The second matrix.

        Returns:
            The Kronecker product of the two matrices.
        '''
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        
        result_rows, result_cols = rows_A * rows_B, cols_A * cols_B
        
        result = [[0] * result_cols for _ in range(result_rows)]
        
        for i in range(rows_A):
            for j in range(cols_A):
                for k in range(rows_B):
                    for l in range(cols_B):
                        result[i * rows_B + k][j * cols_B + l] = A[i][j] * B[k][l]
        
        return result
    
    @staticmethod
    def vector_by_matrix(vector, matrix):
        '''
        Multiply a vector by a matrix. Transposing the matrix before multiplication allows for easier access to columns.

        Args:
            vector: The vector to multiply.
            matrix: The matrix to multiply.

        Returns:
            The result of the multiplication.
        '''
        # Transpose the matrix to access columns easily
        columns = list(zip(*matrix))

        result = [Utility.dot_product(vector, column) for column in columns]
        return result
    
    @staticmethod
    def dot_product(v1, v2):
        '''
        Dot product of two vectors.
        
        Args:
            v1: The first vector.
            v2: The second vector.
        
        Returns:
            The result of the dot product.
        '''
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length")
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result
    
    @staticmethod
    def flip_bit(noisy_message_in_bits: list, index: int):
        '''
        Flip the bit at the specified index in the message.

        Args:
            noisy_message_in_bits: The message in bits.
            index: The index of the bit to flip.
        
        Returns:
            The message with the bit flipped.
        '''
        # Flip the bit at the specified index
        noisy_message_in_bits[index] = 1 - noisy_message_in_bits[index]
        return noisy_message_in_bits
    
    @staticmethod
    def bit_list_to_str(bits):
        '''
        Convert bit list to a string.

        Args:
            bits: list of bits
            
        Returns:
            The string.
        '''
        # Convert the list of bits to a string by processing 8 bits at a time
        message_str = ''
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            byte_str = ''.join(map(str, byte))
            character = chr(int(byte_str, 2))
            message_str += character
        return message_str
