from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import NoiseEnum
from ReedMuller import Utility
# Order of the polynomial, supported up to degree 1

def main():

    hadamard_decoder = HadamardTransform(3)
    reed_muller_coder = ReedMuller(1, 3, hadamard_decoder)
    message = "Change the world, my final message goodbye"
    reed_muller_coder.set_message(message)
    encoded_message = reed_muller_coder.encode()

    for noise_type in NoiseEnum.list_all():
        print(f"Applying noise type: {NoiseEnum.to_string(noise_type)}")
        reed_muller_coder.apply_noise(noise_type, 0.3)
        mistake_positions = reed_muller_coder.get_mistake_positions()
        encoded_message_with_color = list(encoded_message)
        for pos in mistake_positions:
            encoded_message_with_color[pos] = f"\033[91m{encoded_message_with_color[pos]}\033[0m"
        encoded_message_with_color = ''.join(encoded_message_with_color)
        print(f"Encoded message: {encoded_message_with_color}")
    decoded_message = reed_muller_coder.decode()
    print(f"Original message: {message}")
    print(f"Decoded message: {Utility.binary_to_string(decoded_message)}")
    
if __name__ == "__main__":
    main()