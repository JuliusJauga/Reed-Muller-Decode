import numpy as np
import time

# Utility class for common functions used in the Reed-Muller and Hadamard Transform classes, such as matrix operations or bit manipulation
class Utility:
    @staticmethod
    def display_matrix(matrix):
        '''
        Display a matrix in a human-readable format.

        Args:
            matrix: The matrix to display.
        
        Returns:
            None
        '''
        for row in matrix:
            print(row)

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
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row)
        return matrix
    
    @staticmethod
    def generate_kronecher_product(A, B):
        '''
        Generate the Kronecker product of two matrices A and B.

        Args:
            A: The first matrix.
            B: The second matrix.
        
        Returns:
            The result of the Kronecker product.
        '''
        # Generate the Kronecker product of two matrices A and B
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        
        # Initialize the result matrix with zeros
        result = [[0] * (cols_A * cols_B) for _ in range(rows_A * rows_B)]
        
        for i in range(rows_A):
            for j in range(cols_A):
                for k in range(rows_B):
                    for l in range(cols_B):
                        result[rows_B * i + k][cols_B * j + l] = A[i][j] * B[k][l]
        return result
    
    @staticmethod
    def vector_by_matrix(vector, matrix):
        '''
        Multiply a vector by a matrix.
        
        Args:
            vector: The vector to multiply.
            matrix: The matrix to multiply.

        Returns:
            The result of the multiplication.
        '''
        result = []
        columns = []
        for j in range(len(matrix[0])):
            column = []
            for i in range(len(matrix)):
                column.append(matrix[i][j])
            columns.append(column)
        for col in columns:
            result.append(Utility.dot_product(vector, col))
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
    def int_to_bit_array(n, length=None):
        '''
        Convert an integer to a binary array.

        Args:
            n: The integer to convert.
            length: The length of the binary array.
        
        Returns:
            The binary array.
        '''
        # Convert to binary string and remove '0b' prefix
        binary_string = bin(n)[2:]
        
        # Pad with leading zeros if length is specified
        if length:
            binary_string = binary_string.zfill(length)
        
        # Convert binary string to a list of integers
        return [int(bit) for bit in binary_string]
    
    @staticmethod
    def binary_to_string(binary_string, chunk_size=8):
        '''
        Convert a binary string to a string of characters.

        Args:
            binary_string: The binary string to convert.
            chunk_size: The size of each chunk in bits.

        Returns:
            The string of characters.
        '''
        # Split the binary string into chunks of 8 bits
        characters = []
        for i in range(0, len(binary_string), chunk_size):
            character = binary_string[i:i + chunk_size]
            characters.append(chr(int(character, 2)))
        return ''.join(characters)

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
    def np_bit_array_to_str(bits):
        '''
        Convert a NumPy bit array to a string.

        Args:
            bits: The NumPy bit array.
            
        Returns:
            The string.
        '''
        # Ensure the bit array length is a multiple of 8
        if bits.size % 8 != 0:
            bits = np.pad(bits, (0, 8 - bits.size % 8), 'constant')

        # Reshape the bit array into bytes (8 bits per byte)
        bytes_array = bits.reshape(-1, 8)

        # Convert each byte to an integer
        byte_values = np.packbits(bytes_array, axis=1).flatten()

        # Decode the byte values to a string
        decoded_string = byte_values.tobytes().decode('utf-8', errors='ignore')
        
        return decoded_string
