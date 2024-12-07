from typing import Any
from .Utility import Utility
from .IDecoder import IDecoder
from .NoiseApplicator import NoiseApplicator
from .NoiseEnum import NoiseEnum
import numpy as np
import concurrent.futures
import time

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
        start_time = time.time()
        if message is None:
            raise ValueError("Message cannot be None")
        if isinstance(message, str):
            if message == "":
                raise ValueError("Message cannot be empty")
            binary_image = np.array([int(bit) for bit in ''.join(format(ord(char), '08b') for char in message)])
            packed_message = np.packbits(binary_image.flatten())
            unpacked_message = np.unpackbits(packed_message)
            self.message = unpacked_message.tolist()
        elif all(bit in [0, 1] for bit in message):
            self.message = message
        else:
            self.message = np.unpackbits(np.array(message)).tolist()
            end_time = time.time()
            print(f"Time taken to set message: {end_time - start_time}")
    
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
        start_time = time.time()
        # Encode the message using the generator matrix
        if len(self.message) > 16*16*3*8:
            print("Message too large, using sequential encoding")
            encoded = self.encode_sequentially(self.message)
            end_time = time.time()
            print(f"Time taken to encode: {end_time - start_time}")
            return encoded
        chunks = ReedMuller.split_message_for_encoding(self.message, self.m)
        print((len(chunks), len(chunks[0])))
        start_time = time.time()
        generator = self.generator_matrix(self.r, self.m)
        end_time = time.time()
        print(f"Time taken to generate matrix: {end_time - start_time}")
        
        encoded_message = []
        # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        #     futures = {executor.submit(Utility.vector_by_matrix_mod2, chunk, generator): i for i, chunk in enumerate(chunks)}
        #     results = [None] * len(chunks)
        #     for future in concurrent.futures.as_completed(futures):
        #         index = futures[future]
        #         results[index] = future.result()
        #     for result in results:
        #         if result is not None:
        #             encoded_message.extend(result)
        for chunk in chunks:
            encoded_message.extend(Utility.vector_by_matrix_mod2(chunk, generator))
        self.encoded_message = encoded_message
        self.noisy_message = self.encoded_message
        return self.encoded_message

    def encode_sequentially(self, message):
        chunks = self.split_into_16x16_chunks_for_encoding(message)
        encoded_message = []
        generator = self.generator_matrix(self.r, self.m)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.encode_big_chunk, chunk, generator): i for i, chunk in enumerate(chunks)}
            results = [None] * len(chunks)
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()
            for result in results:
                if result is not None:
                    encoded_message.extend(result)
        self.noisy_message = encoded_message
        self.encoded_message = encoded_message
        return encoded_message
    
    def encode_big_chunk(self, chunk, generator):
        chunks = ReedMuller.split_message_for_encoding(chunk, self.m)
        encoded_message = []
        # start_time = time.time()
        for chunk in chunks:
            encoded_message.extend(Utility.vector_by_matrix_mod2(chunk, generator))
        # end_time = time.time()
        # print(f"Time taken to encode big chunk: {end_time - start_time}")
        return encoded_message

    def apply_noise(self, noise_type, noise_amount):
        # Apply noise to the encoded message
        self.noisy_message, self.mistake_positions = NoiseApplicator.apply_noise(self.encoded_message, noise_type, noise_amount)
        return self.noisy_message

    def decode(self):
        # Decode the message using the decoder
        if len(self.noisy_message) == 0:
            raise ValueError("No noisy message to decode")
        
        if len(self.noisy_message) > 16*16*3*8*3:
            print("Message too large, using sequential decoding")
            return self.decode_sequentially(self.noisy_message)
        chunks, self.appended_bits = ReedMuller.split_message_for_decoding(self.noisy_message, 2**self.m)
        decoded_message = []
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = {executor.submit(self.decoder.decode, chunk): i for i, chunk in enumerate(chunks)}
        #     results = [None] * len(chunks)
        #     for future in concurrent.futures.as_completed(futures):
        #         index = futures[future]
        #         results[index] = future.result()
        #     for result in results:
        #         if result is not None:
        #             # print(result)
        #             decoded_message.extend(result)
        for chunk in chunks:
            decoded_message.extend(self.decoder.decode(chunk))
        if self.appended_bits > 0:
            decoded_message = decoded_message[:-self.appended_bits]
        return decoded_message

    def decode_rgb(self):
        chunks = self.split_into_16x16_chunks(self.noisy_message)
        decoded_message = []
        for chunk in chunks:
            decoded_message.extend(self.decode_big_chunk(chunk))
        return decoded_message


    def decode_sequentially(self, message):
        big_chunks = self.split_into_16x16_chunks(message)
        decoded_message = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.decode_big_chunk, chunk): i for i, chunk in enumerate(big_chunks)}
            results = [None] * len(big_chunks)
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                results[index] = future.result()
            for result in results:
                if result is not None:
                    decoded_message.extend(result)
        return decoded_message

    def decode_big_chunk(self, chunk):
        chunks, appended_bits = ReedMuller.split_message_for_decoding(chunk, 2**self.m)
        decoded_message = []
        for chunk in chunks:
            decoded_message.extend(self.decoder.decode(chunk))
        if appended_bits > 0:
            decoded_message = decoded_message[:-appended_bits]
        return decoded_message
    
    def split_into_16x16_chunks(self, message):
        chunks = []
        for i in range(0, len(message), 16*16*3*8*3):
            chunks.append(message[i:i + 16*16*3*8*3])
        return chunks
    
    def split_into_16x16_chunks_for_encoding(self, message):
        chunks = []
        for i in range(0, len(message), 16*16*3*8):
            chunks.append(message[i:i + 16*16*3*8])
        return chunks
    
    
