from typing import Any
from .Utility import Utility
from .IDecoder import IDecoder
from .NoiseApplicator import NoiseApplicator
import numpy as np
import time
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from bitarray import bitarray

def decode_worker(start, end, shm_name, shape, dtype, m, decoder):
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    chunk = shared_data[start:end]
    chunks, appended_bits = ReedMuller.split_message_for_decoding(chunk, 2**m)
    decoded_message = []
    for chunk in chunks:
        decoded_message.extend(decoder.decode(chunk))
    if appended_bits > 0:
        decoded_message = decoded_message[:-appended_bits]
    return decoded_message

def encode_worker(start, end, generator, shm_name, shape, dtype, m):
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    chunk = shared_data[start:end].tolist()
    chunks = ReedMuller.split_message_for_encoding(chunk, m)
    encoded_message = []
    for chunk in chunks:
        encoded_message.extend(Utility.vector_by_matrix_mod2(chunk, generator))
    bitarray_encoded_message = bitarray(encoded_message)
    return bitarray_encoded_message


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
        
    def reset(self):
        self.message = None
        self.noisy_message = None
        self.encoded_message = None
        self.decoded_message = None
        self.mistake_positions = None
        self.noisy_original_message = None
        self.original_message = None
        self.appended_bits = 0

    def change_m(self, new_m):
        self.m = new_m
        self.decoder.change_m(new_m)
    
    def get_m(self):
        return self.m

    def set_message(self, message):
        if message is None:
            raise ValueError("Message cannot be None")
        if isinstance(message, str):
            if message == "":
                raise ValueError("Message cannot be empty")
            binary_image = np.array([int(bit) for bit in ''.join(format(ord(char), '08b') for char in message)])
            packed_message = np.packbits(binary_image.flatten())
            unpacked_message = np.unpackbits(packed_message)
            self.message = unpacked_message.tolist()
        else:
            self.message = message
            self.original_message = message.copy()
        self.noisy_message = None
        self.encoded_message = None

    def set_vector(self, vector):
        if vector is None:
            raise ValueError("Vector cannot be None")
        self.message = [int(bit) for bit in vector]
        self.original_message = [int(bit) for bit in vector]
        self.noisy_message = None
        self.encoded_message = None
    
    def get_message(self):
        try:
            return self.message
        except:
            return None
    def get_encoded_message(self):
        try:
            return self.encoded_message
        except:
            return None
    def get_noisy_message(self):
        try:
            return self.noisy_message
        except:
            return None
    
    def get_noisy_original_message(self):
        try:
            return self.noisy_original_message
        except:
            return self.original_message
    def get_decoded_message(self):
        try:
            return self.decoded_message
        except:
            return None
    def get_mistake_positions(self):
        return self.mistake_positions
    
    def get_original_message(self):
        return self.original_message
    
    def apply_noise_to_original_message(self, noise_type, noise_amount):
        if self.original_message is None:
            raise ValueError("No original message to apply noise to")
        noisy_message, mistake_positions = NoiseApplicator.apply_noise(self.original_message, noise_type, noise_amount)
        self.noisy_original_message = noisy_message
        return noisy_message, mistake_positions
    
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
        return self.mistake_positions

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
            if isinstance(chunks[-1], np.ndarray):
                chunks[-1] = chunks[-1].tolist()
            chunks[-1].append(0)
            appended_bits += 1
        return (chunks, appended_bits)

    def encode(self):
        start_time = time.time()
        if len(self.message) > 16*16*3*8:
            encoded = self.encode_sequentially(self.message)
            end_time = time.time()
            return encoded
        chunks = ReedMuller.split_message_for_encoding(self.message, self.m)
        start_time = time.time()
        generator = self.generator_matrix(self.r, self.m)
        end_time = time.time()
        encoded_message = []
        for chunk in chunks:
            encoded_message.extend(Utility.vector_by_matrix_mod2(chunk, generator))
        self.encoded_message = encoded_message
        self.noisy_message = self.encoded_message
        chunks.clear()
        return self.encoded_message

    def encode_sequentially(self, message):
        chunk_size = self.m + 1
        appended_bits = 0
        generator = self.generator_matrix(self.r, self.m)
        length = len(message)
        flat_message = np.array(message, dtype=np.uint8)
        if length % (self.m+1) != 0:
            appended_bits = ((self.m+1) - (length % (self.m+1)))
            flat_message = np.pad(flat_message, (0, appended_bits))
        shared_mem = shared_memory.SharedMemory(create=True, size=flat_message.nbytes)
        shared_array = np.ndarray(flat_message.shape, dtype=flat_message.dtype, buffer=shared_mem.buf)
        np.copyto(shared_array, flat_message)
        try:
            ranges = self.calculate_ranges(len(flat_message), chunk_size)
            encoded_message = bitarray()
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {
                    executor.submit(encode_worker, start, end, generator, shared_mem.name, flat_message.shape, flat_message.dtype, self.m): i
                    for i, (start, end) in enumerate(ranges)
                }
                results = [None] * len(ranges)
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                for result in results:
                    if result is not None:
                        encoded_message.extend(result)
        finally:
            shared_mem.close()
            shared_mem.unlink()
            ranges.clear()
        
        if appended_bits > 0:
            encoded_message = encoded_message[:-appended_bits]
        self.noisy_message = encoded_message
        self.encoded_message = encoded_message
        return encoded_message
    
    def apply_noise(self, noise_type, noise_amount):
        # Apply noise to the encoded message
        self.noisy_message, self.mistake_positions = NoiseApplicator.apply_noise(self.encoded_message, noise_type, noise_amount)
        return self.noisy_message

    def decode(self):
        # Decode the message using the decoder
        if len(self.noisy_message) == 0:
            raise ValueError("No noisy message to decode")
        
        if len(self.noisy_message) > 16*16*3*8:
            return self.decode_sequentially(self.noisy_message)
        chunks, self.appended_bits = ReedMuller.split_message_for_decoding(self.noisy_message, 2**self.m)
        decoded_message = []
        for chunk in chunks:
            decoded_message.extend(self.decoder.decode(chunk))
        if self.appended_bits > 0:
            decoded_message = decoded_message[:-self.appended_bits]
        chunks.clear()
        return decoded_message

    def decode_sequentially(self, message):
        """Decode a message using shared memory and concurrent processing."""
        # Create a flat NumPy array from the message
        if isinstance(message, bitarray):
            message = message.tolist()
        chunk_size = 2**self.m
        flat_message = np.array(message, dtype=np.uint8)
        if len(flat_message) % (2**self.m) != 0:
            self.appended_bits = ((2**self.m) - (len(flat_message) % (2**self.m)))
            flat_message = np.pad(flat_message, (0, self.appended_bits))
        shared_mem = shared_memory.SharedMemory(create=True, size=flat_message.nbytes)
        shared_array = np.ndarray(flat_message.shape, dtype=flat_message.dtype, buffer=shared_mem.buf)
        np.copyto(shared_array, flat_message)
        try:
            # Calculate optimal ranges for processing based on os.cpu_count()
            ranges = self.calculate_ranges(len(flat_message), chunk_size)
            # Process chunks in parallel
            decoded_message = []
            with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {
                    executor.submit(decode_worker, start, end, shared_mem.name, flat_message.shape, flat_message.dtype, self.m, self.decoder): i
                    for i, (start, end) in enumerate(ranges)
                }

                results = [None] * len(ranges)
                for future in as_completed(futures):
                    index = futures[future]
                    results[index] = future.result()
                for result in results:
                    if result is not None:
                        decoded_message.extend(result)

        finally:
            # Clean up shared memory
            shared_mem.close()
            shared_mem.unlink()
            ranges.clear()
        
        if self.appended_bits > 0:
            decoded_message = decoded_message[:-self.appended_bits]
        elif isinstance(message, list):
            message = bitarray(message)
        self.decoded_message = decoded_message
            
        return self.decoded_message


    def calculate_ranges(self, length_of_message, chunk_size):
        cpu_count = os.cpu_count()
        num_chunks = (length_of_message + chunk_size - 1) // chunk_size

        cpu_count = min(cpu_count, num_chunks)

        ranges = []
        chunks_per_cpu = num_chunks // cpu_count
        remainder_chunks = num_chunks % cpu_count

        start = 0
        for i in range(cpu_count):
            extra_chunk = 1 if i < remainder_chunks else 0
            end = start + (chunks_per_cpu + extra_chunk) * chunk_size
            end = min(end, length_of_message)
            if start >= length_of_message:
                break
            ranges.append((start, end))
            start = end

        return ranges
    
    
