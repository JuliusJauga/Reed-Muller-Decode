from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import Utility
from ReedMuller import NoiseEnum
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
import json
import pandas as pd


R = 1

class BenchmarkData:
    def __init__(self, m, string, message_length, encoded_length, noise, noise_amount, encoding_time, mistake_count, decoding_time, decoded_string, is_equal):
        self.m = m
        self.string = string
        self.message_length = message_length
        self.encoded_length = encoded_length
        self.noise = noise
        self.noise_amount = noise_amount
        self.encoding_time = encoding_time
        self.mistake_count = mistake_count
        self.decoding_time = decoding_time
        self.decoded_string = decoded_string
        self.is_equal = is_equal
        

class BenchmarkDataContainer:
    def __init__(self):
        self.data = []
    
    def add(self, data):
        self.data.append(data)
    
    def save(self):
        with open('benchmark_data.json', 'w') as f:
            json.dump([data.__dict__ for data in self.data], f)

    def load(self):
        with open('benchmark_data.json', 'r') as f:
            data_list = json.load(f)
            self.data = [BenchmarkData(**data) for data in data_list]



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
    strings_to_test = [
        # "OneWord",
        # "Two Words",
        # "Three Words Here",
        # "Four Words Here Right",
        # "Five Words Here Right Now",
        # "Six Words Here Right Now Please",
        # "Seven Words Here Right Now Please Thank",
        # "Eight Words Here Right Now Please Thank You",
        # "Nine Words Here Right Now Please Thank You Very",
        "Ten Words Here Right Now Please Thank You Very Much"
    ]
    images = [
        Image.open("C:\\Users\\juliu\\Desktop\\images.bmp"),
        Image.open("C:\\Users\\juliu\\Desktop\\katukai.bmp")
    ]
    m_ranges_words = range(1, 12)
    m_ranges_images = range(1, 8)
    benchmark_data = BenchmarkDataContainer()
    for m in m_ranges_words:
        for string in strings_to_test:
            noise = NoiseEnum.LINEAR
            for noise_amount in np.arange(0.1, 0.2, 0.1):

                # Setup
                hadamardTransform = HadamardTransform(m)
                rm = ReedMuller(R, m, hadamardTransform)
                rm.set_message(string)
                message = rm.get_message()
                message_length = len(message)
                # Encoding benchmark
                start_time = time.time()
                encoded = rm.encode()
                encoded_length = len(encoded)
                end_time = time.time()
                encoding_time = end_time - start_time

                # Noise benchmark
                applied_noise = rm.apply_noise(noise, noise_amount)
                mistakes = rm.compare_messages()
                mistake_count = len(mistakes)

                # Decoding benchmark
                start_time = time.time()
                decoded = rm.decode()
                end_time = time.time()
                decoding_time = end_time - start_time
                

                decoded_string = Utility.np_bit_array_to_str(np.array(decoded))


                benchmark = BenchmarkData(
                    m=m,
                    string=string,
                    message_length=message_length,
                    encoded_length=encoded_length,
                    noise=NoiseEnum.to_string(noise),
                    noise_amount=noise_amount,
                    encoding_time=encoding_time,
                    mistake_count=mistake_count,
                    decoding_time=decoding_time,
                    decoded_string=decoded_string,
                    is_equal=(string == decoded_string)
                )
                benchmark_data.add(benchmark)
            print(f"Noise {NoiseEnum.to_string(noise)} done")
            print(f"String {string} done")
        print(f"m = {m} done")
    benchmark_data.save()
    benchmark_data = BenchmarkDataContainer()
    benchmark_data.load()
    # Plotting the benchmark data
    grouped_data = {}
    for data in benchmark_data.data:
        if data.m not in grouped_data:
            grouped_data[data.m] = []
        grouped_data[data.m].append(data)
    
    encoding_times = [np.mean([d.encoding_time for d in grouped_data[m]]) for m in grouped_data]
    decoding_times = [np.mean([d.decoding_time for d in grouped_data[m]]) for m in grouped_data]
    noise_amounts = [np.mean([d.noise_amount for d in grouped_data[m]]) for m in grouped_data]
    mistake_counts = [np.mean([d.mistake_count for d in grouped_data[m]]) for m in grouped_data]
    avg_encoding_time = np.mean(encoding_times)
    avg_decoding_time = np.mean(decoding_times)
    avg_noise_amount = np.mean(noise_amounts)
    avg_mistake_count = np.mean(mistake_counts)

    print(f"Average Encoding Time: {avg_encoding_time:.4f} seconds")
    print(f"Average Decoding Time: {avg_decoding_time:.4f} seconds")
    print(f"Average Noise Amount: {avg_noise_amount:.4f}")
    print(f"Average Mistake Count: {avg_mistake_count:.4f}")

    plt.figure(figsize=(12, 8))

    # Plot m vs encoding time
    plt.subplot(2, 3, 1)
    plt.plot([data.m for data in benchmark_data.data], encoding_times, 'o-')
    plt.xlabel('m')
    plt.ylabel('Encoding Time (s)')
    plt.title('m vs Encoding Time')

    # Plot m vs decoding time
    plt.subplot(2, 3, 2)
    plt.plot([data.m for data in benchmark_data.data], decoding_times, 'o-')
    plt.xlabel('m')
    plt.ylabel('Decoding Time (s)')
    plt.title('m vs Decoding Time')

    # Plot m vs encoded length / message length
    plt.subplot(2, 3, 3)
    plt.plot([data.m for data in benchmark_data.data], 
             [data.encoded_length / data.message_length for data in benchmark_data.data], 'o-')
    plt.xlabel('m')
    plt.ylabel('Encoded Length / Message Length')
    plt.title('m vs Encoded Length / Message Length')

    # Plot m vs mistake count
    plt.subplot(2, 3, 4)
    plt.plot([data.m for data in benchmark_data.data], mistake_counts, 'o-')
    plt.xlabel('m')
    plt.ylabel('Mistake Count')
    plt.title('m vs Mistake Count')

    # Plot decoding success vs mistake count
    
    plt.subplot(2, 3, 6)
    num_successes = [sum(d.is_equal for d in grouped_data[m]) for m in grouped_data]
    plt.plot([data.m for data in benchmark_data.data], num_successes, 'o-')
    plt.xlabel('m')
    plt.ylabel('Number of Successes')
    plt.title('m vs Number of Successes')

    plt.tight_layout()
    plt.show()
    # benchmark_data = benchmark_data.data
    #     # Extract the necessary values for plotting
    # noise_types = [benchmark.noise for benchmark in benchmark_data]
    # noise_amounts = [benchmark.noise_amount for benchmark in benchmark_data]
    # encoding_times = [benchmark.encoding_time for benchmark in benchmark_data]
    # decoding_times = [benchmark.decoding_time for benchmark in benchmark_data]
    # mistake_counts = [benchmark.mistake_count for benchmark in benchmark_data]
    # success_rate = [benchmark.is_equal for benchmark in benchmark_data]

    # # Convert success_rate to numeric (1 for True, 0 for False)
    # success_rate_numeric = [1 if success else 0 for success in success_rate]

    # # Prepare data for table
    # table_data = []
    # for benchmark in benchmark_data:
    #     table_data.append([
    #         benchmark.m,
    #         benchmark.string,
    #         benchmark.message_length,
    #         benchmark.encoded_length,
    #         benchmark.noise,
    #         benchmark.noise_amount,
    #         benchmark.encoding_time,
    #         benchmark.mistake_count,
    #         benchmark.decoding_time,
    #         benchmark.decoded_string,
    #         benchmark.is_equal
    #     ])

    # # Define column headers
    # columns = [
    #     'm', 'String', 'Message Length', 'Encoded Length', 'Noise Type',
    #     'Noise Amount', 'Encoding Time', 'Mistake Count', 'Decoding Time',
    #     'Decoded String', 'Is Equal'
    # ]

    #     # Plotting the table and the graphs
    # fig = plt.figure(figsize=(14, 24))  # Adjusted figure size for better fit

    # # Table plot
    # ax1 = fig.add_subplot(2, 1, 1)
    # ax1.axis('tight')
    # ax1.axis('off')
    # table = ax1.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    # # Save the table data to an Excel file
    # # Remove illegal characters from strings
    # def clean_string(s):
    #     return ''.join(c if c.isalnum() or c.isspace() else '_' for c in s)

    # # Clean the data
    # cleaned_table_data = []
    # for row in table_data:
    #     cleaned_row = [clean_string(item) if isinstance(item, str) else item for item in row]
    #     cleaned_table_data.append(cleaned_row)

    # df = pd.DataFrame(cleaned_table_data, columns=columns)
    # df.to_excel('benchmark_data.xlsx', index=False)

    # # df = pd.DataFrame(cleaned_table_data, columns=columns)
    # # df.to_excel('benchmark_data.xlsx', index=False)
    # table.auto_set_font_size(False)
    # table.set_fontsize(8)  # Reduced font size for better fit
    # table.scale(1.5, 1.5)  # Increased scale for wider columns

    # # Adjust column widths
    # for key, cell in table.get_celld().items():
    #     cell.set_width(0.15)  # Increased width for better fit

    # # Plotting the graphs
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(noise_amounts, encoding_times, 'o-', label='Encoding Time')
    # ax2.plot(noise_amounts, decoding_times, 'o-', label='Decoding Time')
    # ax2.plot(noise_amounts, mistake_counts, 'o-', label='Mistake Count')
    # ax2.plot(noise_amounts, success_rate_numeric, 'o-', label='Success Rate')
    # ax2.set_xlabel('Noise Amount')
    # ax2.set_ylabel('Metrics')
    # ax2.set_title('Benchmark Metrics vs Noise Amount')
    # ax2.legend()

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()