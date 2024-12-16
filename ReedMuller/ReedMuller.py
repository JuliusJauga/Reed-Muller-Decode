from typing import Any
from .Utility import Utility
from .IDecoder import IDecoder
from .NoiseApplicator import NoiseApplicator
import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from bitarray import bitarray

# Worker functions for concurrent processing
def decode_worker(start, end, shm_name, shape, dtype, m, decoder):
    '''
    Decode a message using shared memory and concurrent processing.

    Args:
        start (int): The start index of the message.
        end (int): The end index of the message.
        shm_name (str): The name of the shared memory.
        shape (tuple): The shape of the message.
        dtype (type): The data type of the message.
        m (int): The m value of the encoder.
        decoder (IDecoder): The decoder to use.
    
    Returns:
        list: The decoded message.
    '''
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    chunk = shared_data[start:end]
    chunks = ReedMuller.split_message_for_decoding(chunk, 2**m)
    decoded_message = []
    for chunk in chunks:
        decoded_message.extend(decoder.decode(chunk))
    return decoded_message

def encode_worker(start, end, generator, shm_name, shape, dtype, m):
    '''
    Encode a message using shared memory and concurrent processing.
    
    Args:
        start (int): The start index of the message.
        end (int): The end index of the message.
        generator (list): The generator matrix.
        shm_name (str): The name of the shared memory.
        shape (tuple): The shape of the message.
        dtype (type): The data type of the message.
        m (int): The m value of the encoder.

    Returns:
        list: The encoded message.
    '''
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_data = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    chunk = shared_data[start:end].tolist()
    chunks = ReedMuller.split_message_for_encoding(chunk, m)
    encoded_message = []
    for chunk in chunks:
        encoded_message.extend(Utility.vector_by_matrix_mod2(chunk, generator))
    bitarray_encoded_message = bitarray(encoded_message)
    return bitarray_encoded_message


# Reed-Muller code class with the ability to encode, decode, and apply noise to messages
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
        
    def reset(self):
        '''
        Reset the Reed-Muller object to its initial state.

        Returns:
            None
        '''
        self.message = None
        self.noisy_message = None
        self.encoded_message = None
        self.decoded_message = None
        self.mistake_positions = None
        self.noisy_original_message = None
        self.original_message = None

    def change_m(self, new_m):
        '''
        Change the value of m for the Reed-Muller code.
        
        Args:
            new_m (int): The new value of m.
        
        Returns:
            None
        '''
        self.m = new_m
        self.decoder.change_m(new_m)

    def set_message(self, message):
        '''
        Set the message for the Reed-Muller code.

        Args:
            message (str): The message to set.
        
        Returns:
            None
        '''
        if message is None:
            raise ValueError("Message cannot be None")
        if isinstance(message, str):
            if message == "":
                raise ValueError("Message cannot be empty")
            self.message = [int(bit) for char in message for bit in format(ord(char), '08b')]
            self.original_message = [int(bit) for char in message for bit in format(ord(char), '08b')]
        else:
            self.message = message
            self.original_message = message.copy()
        self.noisy_message = None
        self.encoded_message = None

    def set_vector(self, vector):
        '''
        Set the message for the Reed-Muller code.

        Args:
            vector (list): The message to set.
        '''
        if vector is None:
            raise ValueError("Vector cannot be None")
        self.message = [int(bit) for bit in vector]
        self.original_message = [int(bit) for bit in vector]
        self.noisy_message = None
        self.encoded_message = None
    
    def get_m(self):
        '''
        Get the value of m for the Reed-Muller code.
        
        Returns:
            int: The value of m.
        '''    
        return self.m

    def get_message(self):
        '''
        Get the message for the Reed-Muller code.

        Returns:
            str: The message.
        '''
        try:
            return self.message
        except:
            return None
    def get_encoded_message(self):
        '''
        Get the encoded message for the Reed-Muller code.
        
        Returns:
            list: The encoded message.
        '''
        try:
            return self.encoded_message
        except:
            return None
    def get_noisy_message(self):
        '''
        Get the noisy message for the Reed-Muller code.

        Returns:
            list: The noisy message.
        '''
        try:
            return self.noisy_message
        except:
            return None
    
    def get_noisy_original_message(self):
        '''
        Get the noisy original message

        Returns:
            list: The noisy original message
        '''
        try:
            return self.noisy_original_message
        except:
            return self.original_message
    def get_decoded_message(self):
        '''
        Get the decoded message for the Reed-Muller code.

        Returns:
            list: The decoded message.
        '''
        try:
            return self.decoded_message
        except:
            return None
    def get_mistake_positions(self):
        '''
        Get the positions of the mistakes in the noisy message.

        Returns:
            list: A list of the positions of the mistakes in the noisy message.
        '''
        return self.mistake_positions
    
    def get_original_message(self):
        '''
        Get the original message for the Reed-Muller code.

        Returns:
            list: The original message.
        '''
        return self.original_message
    
    def apply_noise_to_original_message(self, noise_type, noise_amount):
        '''
        Apply noise to the original message.

        Args:
            noise_type (NoiseEnum): The type of noise to apply.
            noise_amount (float): The amount of noise to apply.
        
        Returns:
            list: The noisy original message.
        '''
        if self.original_message is None:
            raise ValueError("No original message to apply noise to")
        noisy_message, mistake_positions = NoiseApplicator.apply_noise(self.original_message, noise_type, noise_amount)
        self.noisy_original_message = noisy_message
        return noisy_message, mistake_positions
    
    def flip_mistake_position(self, index):
        '''
        Flip the bit at the specified mistake position in the noisy message.

        Args:
            index (int): The index of the mistake position to flip.
        
        Returns:
            list: The noisy message with the bit flipped.
        '''
        if self.noisy_message is None:
            raise ValueError("No noisy message to flip")
        if index < 0 or index >= len(self.noisy_message):
            raise ValueError("Invalid index")
        self.noisy_message = Utility.flip_bit(self.noisy_message, index)
        self.compare_messages()
        return self.noisy_message
    
    def set_noisy_message(self, noisy_message):
        '''
        Set the noisy message for the Reed-Muller code.

        Args:
            noisy_message (list): The noisy message to set.
        
        Returns:
            None
        '''
        self.noisy_message = noisy_message
        self.compare_messages()
    
    def compare_messages(self):
        '''
        Compare the noisy message with the original message and set the mistake positions.

        Returns:
            list: A list of the positions of the mistakes in the noisy message.
        '''
        self.mistake_positions = []
        for i in range(len(self.encoded_message)):
            if self.encoded_message[i] != self.noisy_message[i]:
                self.mistake_positions.append(i)
        return self.mistake_positions

    def generator_matrix(self, r, m):
        # Generate the generator matrix for a Reed-Muller code
        '''
        Generate the generator matrix for a Reed-Muller code.

        Args:
        r (int): The order of the Reed-Muller code
        m (int): The number of variables in the Reed-Muller code

        Returns:
        list: A list of lists representing the generator matrix

        Example:
        >>> ReedMuller.generator_matrix(1, 1)
        [[1, 1], 
         [0, 1]]

        >>> ReedMuller.generator_matrix(1, 2)
        [[1, 1, 1, 1],
         [0, 1, 0, 1],
         [0, 0, 1, 1]]
        
        >>> ReedMuller.generator_matrix(1, 3)
        [[1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 0, 1, 0, 1, 0, 1],
         [0, 0, 1, 1, 0, 0, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1]]
        '''
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
        '''
        Get the positions of the mistakes in the noisy message.

        Returns:
        list: A list of the positions of the mistakes in the noisy message.
        '''
        return self.mistake_positions

    @staticmethod
    def split_message_for_encoding(message, m):
        '''
        Split a message into chunks of size m+1.
        
        Args:
            message (list): The message to split.
            m (int): The m value of encoder.

        Returns:
            list: A list of chunks.
        '''
        # If the last chunk is smaller than m, pad it with zeros
        chunks = []
        m += 1
        for i in range(0, len(message), m):
            chunks.append(message[i:i + m])
        while len(chunks[-1]) < m:
            chunks[-1].append(0)
            appended_bits += 1
        return chunks

    @staticmethod
    def split_message_for_decoding(message, n):
        '''
        Split a message into chunks of size n precisely 2**m.

        Args:
            message (list): The message to split.
            n (int): The size of each chunk.
            appended_bits (int): The number of bits appended to the message.
        
        Returns:
            list: A list of chunks.
        '''
        # Split the message into chunks of size n
        # If the last chunk is smaller than m, pad it with zeros
        chunks = []
        for i in range(0, len(message), n):
            chunks.append(message[i:i + n])
        while len(chunks[-1]) < n:
            if isinstance(chunks[-1], np.ndarray):
                chunks[-1] = chunks[-1].tolist()
            chunks[-1].append(0)
        return chunks

    def encode(self):
        '''
        Encode the message using the generator matrix. Different encoding methods are used based on the size of the message.

        Returns:
            list: The encoded message.
        '''
        self.original_message_length = len(self.message)
        if len(self.message) > 16*16*3*8:
            encoded = self.encode_sequentially(self.message)
            return encoded
        chunks = ReedMuller.split_message_for_encoding(self.message, self.m)
        generator = self.generator_matrix(self.r, self.m)
        encoded_message = []
        for chunk in chunks:
            encoded_message.extend(Utility.vector_by_matrix_mod2(chunk, generator))
        self.encoded_message = encoded_message
        self.noisy_message = self.encoded_message
        chunks.clear()
        return self.encoded_message

    def encode_sequentially(self, message):
        '''
        Encode a message using shared memory and concurrent processing.

        Args:
            message (list): The message to encode.
        
        Returns:
            list: The encoded message.
        '''
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
        
        self.noisy_message = encoded_message
        self.encoded_message = encoded_message
        return encoded_message
    
    def apply_noise(self, noise_type, noise_amount):
        '''
        Apply noise to the encoded message.

        Args:
            noise_type (NoiseEnum): The type of noise to apply.
            noise_amount (float): The amount of noise to apply.
        
        Returns:
            list: The noisy message.
        '''
        # Apply noise to the encoded message
        self.noisy_message, self.mistake_positions = NoiseApplicator.apply_noise(self.encoded_message, noise_type, noise_amount)
        return self.noisy_message

    def decode(self):
        '''
        Decode the message using the decoder. Different decoding methods are used based on the size of the encoded message.

        Returns:
            list: The decoded message.    
        '''
        if len(self.noisy_message) == 0:
            raise ValueError("No noisy message to decode")
        
        if len(self.noisy_message) > 2**self.m * 8:
            return self.decode_sequentially(self.noisy_message)
        chunks = ReedMuller.split_message_for_decoding(self.noisy_message, 2**self.m)
        decoded_message = []
        for chunk in chunks:
            decoded_message.extend(self.decoder.decode(chunk))
        if self.original_message_length is not None:
            difference = len(decoded_message) - self.original_message_length
            if difference > 0:
                decoded_message = decoded_message[:-difference]
        chunks.clear()
        return decoded_message

    def decode_sequentially(self, message):
        '''
        Decode a message using shared memory and concurrent processing.

        Args:
            message (list): The message to decode.

        Returns:
            list: The decoded message.
        '''
        if isinstance(message, bitarray):
            message = message.tolist()
        chunk_size = 2**self.m
        flat_message = np.array(message, dtype=np.uint8)
        if len(flat_message) % (2**self.m) != 0:
            appended_bits = ((2**self.m) - (len(flat_message) % (2**self.m)))
            flat_message = np.pad(flat_message, (0, appended_bits))
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
        
        
        if isinstance(message, list):
            message = bitarray(message)
        self.decoded_message = decoded_message
        if self.original_message_length is not None:
            difference = len(decoded_message) - self.original_message_length
            if difference > 0:
                self.decoded_message = decoded_message[:-difference]
        return self.decoded_message


    def calculate_ranges(self, length_of_message, chunk_size):
        '''
        Calculate the optimal ranges for processing based on the number of CPUs available.

        Args:
            length_of_message (int): The length of the message.
            chunk_size (int): The size of each chunk.
        
        Returns:
            list: A list of tuples representing the ranges for processing.
        '''
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
    
    
