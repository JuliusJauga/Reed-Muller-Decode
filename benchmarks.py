from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import NoiseEnum
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
R = 1


def reed_muller_error_correction_capability(r, m):
    """
    Calculate the number of errors a Reed-Muller code can correct.

    Parameters:
        r (int): The order of the Reed-Muller code.
        m (int): The number of variables in the Reed-Muller code.

    Returns:
        int: The maximum number of errors the code can correct.
    """
    # Calculate the minimum Hamming distance
    d_min = 2**(m - r)
    
    # Calculate the error correction capability
    t = (d_min - 1) // 2  # Equivalent to floor((d_min - 1) / 2)
    
    return t

class BenchmarkData:
    def __init__(self, m, message_length, encoded_length, redundant_bits, noise_amount, encoding_time, decoding_time, is_equal):
        self.m = m
        self.message_length = message_length
        self.encoded_length = encoded_length
        self.redundant_bits = redundant_bits
        self.noise_amount = noise_amount
        self.encoding_time = encoding_time
        self.decoding_time = decoding_time
        self.is_equal = is_equal
        

class BenchmarkDataContainer:
    def __init__(self):
        self.data = []
        self.old_data = []
    
    def add(self, data):
        self.data.append(data)
    
    def save(self):
        with open('benchmark_data.json', 'w') as f:
            json.dump([data.__dict__ for data in self.data], f)

    def load(self):
        with open('benchmark_data.json', 'r') as f:
            data_list = json.load(f)
            self.data = [BenchmarkData(**data) for data in data_list]

    def load_old(self):
        with open('benchmark_data.json', 'r') as f:
            data_list = json.load(f)
            self.old_data = [BenchmarkData(**data) for data in data_list]
    
    def append(self):
        self.load_old()
        self.data = self.old_data + self.data
        self.save()

def image_to_binary(image):
    # Load the image
    img = image.convert('RGB')
    img_array = np.array(img, dtype=np.uint8)  # Convert to NumPy array

    # Flatten and convert each channel to binary
    binary_array = np.unpackbits(img_array.flatten())

    return binary_array, img_array.shape  # Return binary data and original shape

def binary_to_image(binary_array, shape):
    # Reshape and convert binary back to uint8
    byte_array = np.packbits(binary_array)
    # Calculate the number of bytes needed to match the original shape
    num_bytes = np.prod(shape)
    # Ensure the byte array has the correct size before reshaping
    if len(byte_array) < num_bytes:
        byte_array = np.pad(byte_array, (0, num_bytes - len(byte_array)), 'constant', constant_values=0)
    if len(byte_array) > num_bytes:
        byte_array = byte_array[:num_bytes]

    img_array = byte_array.reshape(shape)
    # Convert back to a PIL image
    return Image.fromarray(img_array, mode='RGB')


def main():
    m_ranges_words = range(11, 12)
    benchmark_data = BenchmarkDataContainer()

    # for m in m_ranges_words:
    #     start_m_time = time.time()
    #     vector_length = m+1
    #     print(f"m: {m}")
    #     hadamardTransform = HadamardTransform(m)
    #     rm = ReedMuller(1, m, hadamardTransform)
        
    #     for i in range(1, 100):
    #         print(f"i: {i}")
    #         vector = np.random.randint(0, 2**vector_length)
    #         bit_list = [int(bit) for bit in bin(vector)[2:]]
    #         bit_list = bit_list + [0] * (vector_length - len(bit_list))
    #         rm.set_message(bit_list)
    #         start_time = time.time()
    #         encoded = rm.encode()
    #         encoded_length = len(encoded)
    #         end_time = time.time()
    #         encoding_time = end_time - start_time
    #         ranges = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.49]
    #         for noise_amount in ranges:
    #             rm.apply_noise(NoiseEnum.LINEAR, noise_amount)
    #             start_time = time.time()
    #             decoded = rm.decode()
    #             end_time = time.time()
    #             decoding_time = end_time - start_time
    #             benchmark = BenchmarkData(
    #                 m=m,
    #                 message_length=vector_length,
    #                 encoded_length=encoded_length,
    #                 redundant_bits=encoded_length - vector_length,
    #                 noise_amount=noise_amount,
    #                 encoding_time=encoding_time,
    #                 decoding_time=decoding_time,
    #                 is_equal=(bit_list == decoded)
    #             )
    #             benchmark_data.add(benchmark)
    #     end_m_time = time.time()
    #     print(f"Time taken for m={m}: {end_m_time - start_m_time} seconds")

    # # Save benchmark data
    # benchmark_data.append()

    # # Load benchmark data
    benchmark_data.load()

    # Data structures to hold aggregated metrics
    success_rate_data = defaultdict(lambda: defaultdict(list))
    encoding_time_data = defaultdict(list)
    decoding_time_data = defaultdict(list)
    redundant_bits_data = defaultdict(list)

    # Process benchmark data
    for benchmark in benchmark_data.data:
        m, noise = benchmark.m, benchmark.noise_amount
        # Success Rate
        success_rate_data[m][noise].append(benchmark.is_equal)
        # Encoding and decoding times
        encoding_time_data[m].append(benchmark.encoding_time)
        decoding_time_data[m].append(benchmark.decoding_time)
        # Redundant bits
        redundant_bits_data[m].append(benchmark.redundant_bits)

    # Calculate success rate per noise range
    success_rates = []
    for m, noise_dict in success_rate_data.items():
        for noise, outcomes in noise_dict.items():
            success_rate = sum(outcomes) / len(outcomes)
            print("sum(outcomes): ", sum(outcomes))
            print("len(outcomes): ", len(outcomes))
            success_rates.append((m, noise, success_rate))
    print(success_rates)

    # Calculate averages for encoding/decoding times and redundant bits
    averaged_encoding_times = {m: np.mean(times) for m, times in encoding_time_data.items()}
    averaged_decoding_times = {m: np.mean(times) for m, times in decoding_time_data.items()}
    averaged_redundant_bits = {m: np.mean(bits) for m, bits in redundant_bits_data.items()}

    # Unique values for axes
    ms = sorted(encoding_time_data.keys())
    noise_ranges = sorted(set(noise for _, noise, _ in success_rates))

    if not os.path.exists("plots"):
        os.makedirs("plots")

    # 1. Success Rate Graph
    plt.figure(figsize=(12, 10))

    for noise in noise_ranges:
        m_values = [m for m, n, _ in success_rates if n == noise]
        rates = [rate for m, n, rate in success_rates if n == noise]
        plt.plot(m_values, rates, label=f"Klaidos tikimybė: {noise}")

    plt.title("Sėkmės rodiklis vs m kiekvienam triukšmo kiekiui")
    plt.xlabel("m")
    plt.ylabel("Sėkmės rodiklis")
    plt.xticks(range(1, 12))
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/success_rate_vs_m.png")
    plt.close()

    # 2. Encoding Time Graph
    plt.figure(figsize=(10, 6))
    plt.plot(ms, [averaged_encoding_times[m] for m in ms], marker='o')
    plt.title("Vidutinis kodavimo laikas vs m")
    plt.xlabel("m")
    plt.ylabel("Dekodavimo laikas (s)")
    plt.xticks(range(1, 12))
    plt.grid(True)
    plt.savefig("plots/encoding_time_vs_m.png")
    plt.close()

    # 3. Decoding Time Graph
    plt.figure(figsize=(10, 6))
    plt.plot(ms, [averaged_decoding_times[m] for m in ms], marker='o', color='orange')
    plt.title("Vidutinis dekodavimo laikas vs m")
    plt.xlabel("m")
    plt.xticks(range(1, 12))
    plt.ylabel("Dekodavimo laikas (s)")
    plt.grid(True)
    plt.savefig("plots/decoding_time_vs_m.png")
    plt.close()
    
    # 4. Redundant Bits Graph
    plt.figure(figsize=(10, 6))
    plt.plot(ms, [averaged_redundant_bits[m] for m in ms], marker='o', color='green')
    plt.title("Pridėtiniai bitai vs m")
    plt.xlabel("m")
    plt.xticks(range(1, 12))
    plt.ylabel("Pridėtiniai bitai")
    plt.grid(True)
    plt.savefig("plots/redundant_bits_vs_m.png")
    plt.close()
    
    # 5. Error correction capability graph
    m_vs_error_correction = []
    for m in range(1, 12):
        t = reed_muller_error_correction_capability(R, m)
        m_vs_error_correction.append((m, t))
    m_values, t_values = zip(*m_vs_error_correction)
    plt.figure(figsize=(10, 6))
    plt.plot(m_values, t_values, marker='o', color='red')
    plt.title("Klaidų korekcijos galimybė vs m")
    plt.xlabel("m")
    plt.xticks(range(1, 12))
    plt.ylabel("Klaidų korekcijos galimybė")
    plt.grid(True)
    plt.savefig("plots/error_correction_capability_vs_m.png")
    plt.close()

    # 6. Error correction vs redundancy
    plt.figure(figsize=(10, 6))
    plt.plot([averaged_redundant_bits[m] for m in ms], t_values, marker='o', color='purple')
    plt.title("Klaidos korekcijos galimybė vs pridėtiniai bitai")
    plt.xlabel("Pridėtiniai bitai")
    plt.ylabel("Klaidos korekcijos galimybė")
    plt.grid(True)
    plt.savefig("plots/error_correction_vs_redundant_bits.png")
    plt.close()

    # 7. Success rate vs noise amount for each m
    for m in ms:
        success_rates_m = [(noise, rate) for _, noise, rate in success_rates if _ == m]
        noise_values, rate_values = zip(*success_rates_m)
        plt.figure(figsize=(10, 6))
        plt.plot(noise_values, rate_values, marker='o')
        plt.title(f"Sėkmės rodiklis vs triukšmo kiekis m={m}")
        plt.xlabel("Triukšmo kiekis")
        plt.ylabel("Sėkmės rodiklis")
        plt.grid(True)
        plt.savefig(f"plots/success_rate_vs_noise_amount_m_{m}.png")
        plt.close()

    



if __name__ == "__main__":
    main()