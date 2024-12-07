from typing import Any
from .Utility import Utility
from .IDecoder import IDecoder
from .NoiseApplicator import NoiseApplicator
from .NoiseEnum import NoiseEnum
import numpy as np
import concurrent.futures

class ReedMuller:
    def __init__(self, r: int, m: int, decoder: IDecoder):
        self.m = m
        self.r = r
        self.n = 2**m
        self.decoder = decoder
        self.appended_bits = 0        
        if r > 1 or r < 0:
            raise NotImplementedError("Only Reed-Muller codes with r = 1 are supported")
        if m == 0:
            raise ValueError("m must be greater than 0")
        if r > m:
            raise ValueError("r must be less than or equal to m")
        

    def set_message(self, message):
        if message is None:
            raise ValueError("Message cannot be None")
        if message == "":
            raise ValueError("Message cannot be empty")
        if isinstance(message, str):
            binary_image = np.array([int(bit) for bit in ''.join(format(ord(char), '08b') for char in message)])
            packed_message = np.packbits(binary_image.flatten())
            unpacked_message = np.unpackbits(packed_message)
            self.message = unpacked_message.tolist()
        elif all(bit in [0, 1] for bit in message):
            self.message = message
        else:
            raise ValueError("Message must be a string or a list of bits")
    
    def get_message(self):
        return self.message
    
    def get_encoded_message(self):
        return self.encoded_message
    
    def get_noisy_message(self):
        return self.noisy_message
    
    def get_mistake_positions(self):
        return self.mistake_positions
    
    def flip_mistake_position(self, index):
        if self.noisy_message is None:
            raise ValueError("No noisy message to flip")
        if index < 0 or index >= len(self.noisy_message):
            raise ValueError("Invalid index")
        self.noisy_message = Utility.flip_bit(self.noisy_message, index)
        self.compare_messages()
        return self.noisy_message
    
    def set_noisy_message(self, noisy_message):
        self.noisy_message = noisy_message
        self.compare_messages()
    
    def compare_messages(self):
        self.mistake_positions = []
        for i in range(len(self.encoded_message)):
            if self.encoded_message[i] != self.noisy_message[i]:
                self.mistake_positions.append(i)

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
        while len(chunks[-1]) < m:
            chunks[-1].append(0)
        return chunks

    @staticmethod
    def split_message_for_decoding(message, m, appended_bits=0):
        # Split the message into chunks of size m
        # If the last chunk is smaller than m, pad it with zeros
        chunks = []
        appended_bits = 0
        for i in range(0, len(message), m):
            chunks.append(message[i:i + m])
        while len(chunks[-1]) < m:
            chunks[-1].append(0)
            appended_bits += 1
        return (chunks, appended_bits)

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
            for result in results:
                if result is not None:
                    encoded_message.extend(result)
        self.encoded_message = encoded_message
        self.noisy_message = self.encoded_message
        return self.encoded_message

    def apply_noise(self, noise_type, noise_amount):
        # Apply noise to the encoded message
        self.noisy_message, self.mistake_positions = NoiseApplicator.apply_noise(self.encoded_message, noise_type, noise_amount)
        return self.noisy_message

    def decode(self):
        # Decode the message using the decoder
        chunks, self.appended_bits = ReedMuller.split_message_for_decoding(self.noisy_message, 2**self.m)
        decoded_message = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.decoder.decode, chunk): i for i, chunk in enumerate(chunks)}
            results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()
            for result in results:
                if result is not None:
                    # print(result)
                    decoded_message.extend(result)
        if self.appended_bits > 0:
            decoded_message = decoded_message[:-self.appended_bits]
        return decoded_message

    
