from .Utility import Utility
from .IDecoder import IDecoder
from .NoiseApplicator import NoiseApplicator
from .NoiseEnum import NoiseEnum
import concurrent.futures

class ReedMuller:
    def __init__(self, r: int, m: int, decoder: IDecoder):
        self.m = m
        self.r = r
        self.n = 2**m
        self.decoder = decoder        
        if r > 1 or r < 0:
            raise NotImplementedError("Only Reed-Muller codes with r = 1 are supported")
        if m == 0:
            raise ValueError("m must be greater than 0")
        if r > m:
            raise ValueError("r must be less than or equal to m")
        

    def set_message(self, message):
        self.message = ''.join(format(ord(char), '08b') for char in message)

    def generator_matrix(self, r, m):
        # Generate the generator matrix for a Reed-Muller code
        if m == 0:
            raise ValueError("m must be greater than 0")
        if r > m:
            raise ValueError("r must be less than or equal to m")
        if r == 0:
            return [[1] * (2 ** m)]
        if m == 1:
            return [[1 , 1], 
                    [0 , 1]]
        else:
            smaller_matrix = self.generator_matrix(r, m - 1)
            bottom = self.generator_matrix(r - 1, m - 1)
            
            top = [row + row for row in smaller_matrix]
            
            bottom = [[0] * len(smaller_matrix[0]) + row for row in bottom]

            return top + bottom
        
    def get_mistake_positions(self):
        return self.mistake_positions

    @staticmethod
    def split_message_for_encoding(message, m):
        # Split the message into chunks of size m
        # If the last chunk is smaller than m, pad it with zeros
        chunks = []
        m += 1
        for i in range(0, len(message), m):
            chunks.append(message[i:i + m])
        if len(chunks[-1]) < m:
            chunks[-1] = chunks[-1].ljust(m, '0')
        return chunks

    @staticmethod
    def split_message_for_decoding(message, m):
        # Split the message into chunks of size m
        # If the last chunk is smaller than m, pad it with zeros
        chunks = []
        for i in range(0, len(message), m):
            chunks.append(message[i:i + m])
        if len(chunks[-1]) < m:
            chunks[-1] = chunks[-1].ljust(m, '0')
        return chunks

    def encode(self):
        # Encode the message using the generator matrix
        chunks = ReedMuller.split_message_for_encoding(self.message, self.m)
        generator = self.generator_matrix(self.r, self.m)
        encoded_message = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(Utility.vector_by_matrix_mod2, [int(bit) for bit in chunk], generator): i for i, chunk in enumerate(chunks)}
            results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()
            encoded_message.extend(result for result in results if result is not None)
        self.encoded_message = ''.join([''.join(map(str, bits)) for bits in encoded_message])
        self.noisy_message = self.encoded_message
        return self.encoded_message

    def apply_noise(self, noise_type, noise_amount):
        # Apply noise to the encoded message
        self.noisy_message, self.mistake_positions = NoiseApplicator.apply_noise(self.encoded_message, noise_type, noise_amount)
        return self.noisy_message

    def decode(self):
        # Decode the message using the decoder
        chunks = ReedMuller.split_message_for_decoding(self.noisy_message, 2**self.m)
        decoded_message = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.decoder.decode, chunk): i for i, chunk in enumerate(chunks)}
            results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()
            decoded_message.extend(result for result in results if result is not None)
        return ''.join([str(bit) for bit in decoded_message])

    
