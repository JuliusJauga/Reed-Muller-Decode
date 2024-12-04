class Utility:
    @staticmethod
    def display_matrix(matrix):
        for row in matrix:
            print(row)

    @staticmethod
    def vector_by_matrix_mod2(vector, matrix):
        result = []
        columns = []
        for j in range(len(matrix[0])):
            column = []
            for i in range(len(matrix)):
                column.append(matrix[i][j])
            columns.append(column)
        for col in columns:
            result.append(Utility.dot_product_mod2(vector, col))
        return result

    @staticmethod
    def dot_product_mod2(v1, v2):
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length")
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result % 2

    @staticmethod
    def generate_unitary_matrix(n):
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
        if len(v1) != len(v2):
            raise ValueError("Vectors must have the same length")
        result = 0
        for i in range(len(v1)):
            result += v1[i] * v2[i]
        return result
    
    @staticmethod
    def int_to_bit_array(n, length=None):
        # Convert to binary string and remove '0b' prefix
        binary_string = bin(n)[2:]
        
        # Pad with leading zeros if length is specified
        if length:
            binary_string = binary_string.zfill(length)
        
        # Convert binary string to a list of integers
        return [int(bit) for bit in binary_string]
    
    @staticmethod
    def binary_to_string(binary_string, chunk_size=8):
    # Split the binary string into chunks of 8 bits

        characters = []
        for i in range(0, len(binary_string), chunk_size):
            character = binary_string[i:i + chunk_size]
            characters.append(chr(int(character, 2)))
        return ''.join(characters)

    @staticmethod
    def flip_bit(noisy_message_in_bits: str, index: int) -> str:
        # Flip the bit at the specified index
        return noisy_message_in_bits[:index] + str(int(not int(noisy_message_in_bits[index]))) + noisy_message_in_bits[index + 1:]
