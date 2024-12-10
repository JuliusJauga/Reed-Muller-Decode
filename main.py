from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import NoiseEnum
import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import numpy as np


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

def decode_image(reed_muller_images, original_path, noise_type, noise_amount, orig_shape):
    print("Applying noise to the image...")
    reed_muller_images.apply_noise(noise_type, noise_amount)
    print("Decoding image...")
    decoded_image = binary_to_image(reed_muller_images.decode(), orig_shape)
    original_image = Image.open(original_path)
    # Display the original and decoded images in a Tkinter window
    root = tk.Tk()
    root.title("Original and Decoded Images")

    # Load and display the original image
    original_img = ImageTk.PhotoImage(original_image)
    original_label = tk.Label(root, image=original_img)
    original_label.pack(side="left", padx=10, pady=10)

    # Load and display the decoded image
    decoded_img = ImageTk.PhotoImage(decoded_image)
    decoded_label = tk.Label(root, image=decoded_img)
    decoded_label.pack(side="right", padx=10, pady=10)

    root.mainloop()


def main():
    hadamard_transform_images = HadamardTransform(1)
    hadamard_transform_messages = HadamardTransform(1)
    reed_muller_images = ReedMuller(1, 1, hadamard_transform_images)
    reed_muller_messages = ReedMuller(1, 1, hadamard_transform_messages)
    noise_enum = NoiseEnum()
    message = None
    picture_path = None
    m_value = None
    noise_type = NoiseEnum.LINEAR
    noise_amount = 0.0
    orig_shape = None

    while True:
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("Choose an option:")
        print("1. Enter message")
        print("2. Open a picture")
        print("3. Enter value of m")
        print("4. Enter noise type")
        print("5. Enter noise amount")
        if message and not picture_path:
            print("6. Encode message")
            print("7. Decode message")
        if picture_path and not message:
            print("6. Encode image")
            print("7. Decode image")
        if message and picture_path:
            print("6. Encode message")
            print("7. Decode message")
            print("8. Encode image")
            print("9. Decode image")
        if (message and not picture_path) or (picture_path and not message):
            print("8. Exit")
        if message and picture_path:
            print("10. Exit")
        
        if message:
            print(f"Message entered: {message}")
        if picture_path:
            print(f"Picture path entered: {picture_path}")
        if m_value:
            print(f"Value of m entered: {m_value}")
        if noise_type:
            print(f"Noise type: {NoiseEnum.to_string(noise_type)}")
        if noise_amount:
            print(f"Noise amount: {noise_amount}")
        if reed_muller_images.get_noisy_message():
            print(f"Image encoded")
        if reed_muller_messages.get_noisy_message():
            print(f"Message encoded")

        choice = input("Enter your choice: ")
        if choice == '1':
            message = input("Enter the message: ")
            print(f"Message entered: {message}")
            reed_muller_messages.set_message(message)
        elif choice == '2':
            root = tk.Tk()
            root.withdraw()
            picture_path = filedialog.askopenfilename(title="Select a picture", filetypes=[("Image files", "*.bmp")])
            if not picture_path:
                print("No picture selected.")
            else:
                print(f"Picture path entered: {picture_path}")
                image = Image.open(picture_path)
                print("Converting image to binary...")
                binary_array, orig_shape = image_to_binary(image)
                print("Setting message...")
                reed_muller_images.set_message(binary_array)
                continue
                
        elif choice == '3':
            try:
                m_value = int(input("Enter the value of m: "))
            except ValueError:
                print("Invalid value of m entered. Please try again.")
                continue
            hadamard_transform_images = HadamardTransform(m_value)
            hadamard_transform_messages = HadamardTransform(m_value)
            reed_muller_images = ReedMuller(1, m_value, hadamard_transform_images)
            reed_muller_messages = ReedMuller(1, m_value, hadamard_transform_messages)
            if message:
                reed_muller_messages.set_message(message)
            if picture_path:
                reed_muller_images.set_message(binary_array)
            print(f"Value of m entered: {m_value}")
        elif choice == '4':
            print("Choose a noise type:")
            i = 1
            for noise in NoiseEnum.list_all():
                print(f"{i}. {NoiseEnum.to_string(noise)}")
                i += 1
            noise_type = int(input("Enter the noise type: "))
            noise_type += 1
            noise_type = min(max(1, noise_type), len(NoiseEnum.list_all()))
            print(f"Noise type: {NoiseEnum.to_string(noise_type)}")
        elif choice == '5':
            try:
                noise_amount = input("Enter the noise amount (between 0 and 1): ").replace(',', '.')
                noise_amount = float(noise_amount)
                noise_amount = max(0, min(1, noise_amount))
                print(f"Noise amount: {noise_amount}")
            except ValueError:
                print("Invalid noise amount entered. Please try again.")
                continue
        elif choice == '6' and message and not picture_path:
            print("Encoding message...")
            reed_muller_messages.encode()
        elif choice == '7' and message and not picture_path:
            print("Decoding message...")
            reed_muller_messages.apply_noise(noise_type, noise_amount)
            reed_muller_messages.decode()
        elif choice == '6' and picture_path and not message:
            print("Encoding image...")
            reed_muller_images.encode()
        elif choice == '7' and picture_path and not message:
            decode_image(reed_muller_images, picture_path, noise_type, noise_amount, orig_shape)
        elif choice == '6' and message and picture_path:
            print("Encoding message...")
            reed_muller_messages.encode()
        elif choice == '7' and message and picture_path:
            print("Decoding message...")
            reed_muller_messages.apply_noise(noise_type, noise_amount)
            reed_muller_messages.decode()
        elif choice == '8' and message and picture_path:
            print("Encoding image...")
            reed_muller_images.encode()
        elif choice == '9' and message and picture_path:
            decode_image(reed_muller_images, picture_path, noise_type, noise_amount, orig_shape)
        elif choice == '10' and message and picture_path:
            print("Exiting...")
            break
        elif choice == '8' and (message and not picture_path) or (picture_path and not message):
            print("Exiting...")
            break
        elif choice == '6' and not message and not picture_path:
            print("Exiting...")
            break
        else:
            print("Invalid choice or action not allowed. Please try again.")

if __name__ == "__main__":
    main()