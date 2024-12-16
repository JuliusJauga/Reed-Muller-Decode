import streamlit as st
from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import NoiseEnum
from ReedMuller import Utility
import time
import numpy as np
from PIL import Image
import os
from bitarray import bitarray
from typing import Union

# Set page configuration, for full width
st.set_page_config(
    layout="wide",  # Expands the app to use the full width of the browser
)

# Convert an image to binary data
def image_to_binary(image):
    '''
    Converts an image to binary data.

    Args:
        image (PIL.Image): The image to convert
    
    Returns:
        tuple: A tuple containing the binary data and the original shape
    '''
    # Load the image
    img = image.convert('RGB')
    img_array = np.array(img, dtype=np.uint8)  # Convert to NumPy array

    # Flatten and convert each channel to binary
    binary_array = np.unpackbits(img_array.flatten())

    return binary_array, img_array.shape  # Return binary data and original shape

def binary_to_image(binary_array, shape):
    '''
    Converts binary data to an image.

    Args:
        binary_array (list): The binary data
        shape (tuple): The original shape of the image
    
    Returns:
        PIL.Image: The image created from the binary data
    '''
    if isinstance(binary_array, bitarray):
        binary_array = np.array(binary_array.tolist(), dtype=np.uint8)
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
    if isinstance(binary_array, list):
        binary_array = bitarray.bitarray(binary_array)
    return Image.fromarray(img_array, mode='RGB')

# Convert a message to bit string
def message_to_bits(message):
    '''
    Converts a message to a bit string.

    Args:
        message (list): The message to convert
    
    Returns:
        str: The bit string representation of the message
    '''
    message = bitarray(message)
    return message.to01()


# Initialize Reed-Muller and Hadamard transform objects
if 'decoder' not in st.session_state:
    st.session_state['decoder'] = HadamardTransform(3)
if 'coder' not in st.session_state:
    st.session_state['coder'] = ReedMuller(1, 3, st.session_state['decoder'])


def encode_message(message: str, coder: ReedMuller):
    '''
    Encodes a message using the Reed-Muller encoder.

    Args:
        message (str): The message to encode
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The encoded message
    '''
    coder.set_message(message)
    return coder.encode()


def encode_vector(vector: list, coder: ReedMuller):
    '''
    Encodes a vector using the Reed-Muller encoder.

    Args:
        vector (list): The vector to encode
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The encoded message
    '''
    coder.set_vector(vector)
    if len(vector) != coder.get_m() + 1:
        raise ValueError(f"Vector must be exactly {coder.get_m() + 1} bits long")
    return coder.encode()

def apply_noise(noise_type: NoiseEnum, noise_amount: float, coder: ReedMuller):
    '''
    Applies noise to the encoded message using the specified noise type and amount.

    Args:
        noise_type (NoiseEnum): The type of noise to apply
        noise_amount (float): The amount of noise to apply
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The noisy message
    '''
    coder.apply_noise(noise_type, noise_amount)
    return coder.get_noisy_message()

def apply_noise_to_original(noise_type: NoiseEnum, noise_amount: float, coder: ReedMuller):
    '''
    Applies noise to the original message using the specified noise type and amount.

    Args:
        noise_type (NoiseEnum): The type of noise to apply
        noise_amount (float): The amount of noise to apply
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The noisy message
    '''
    coder.apply_noise_to_original_message(noise_type, noise_amount)    
    return coder.get_noisy_message()

def get_mistake_positions(coder: ReedMuller):
    '''
    Returns the positions of the mistakes in the encoded message.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object

    Returns:
        list: The positions of the mistakes
    '''
    return coder.get_mistake_positions()

def set_message(message: Union[str, list], coder: ReedMuller):
    '''
    Sets the message in the Reed-Muller encoder.

    Args:
        message (str or list): The message to set
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        None
    '''
    coder.set_message(message)

def get_encoded_message(coder: ReedMuller):
    '''
    Gets the encoded message from the Reed-Muller encoder.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The encoded message
    '''
    return coder.get_encoded_message()

def get_noisy_message(coder: ReedMuller):
    '''
    Gets the noisy message from the Reed-Muller encoder.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The noisy message
    '''
    return coder.get_noisy_message()

def get_original_message(coder: ReedMuller):
    '''
    Gets the original message from the Reed-Muller encoder.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The original message
    '''
    return coder.get_original_message()

def get_noisy_original_message(coder: ReedMuller):
    '''
    Gets the noisy original message from the Reed-Muller encoder.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The noisy original message
    '''
    return coder.get_noisy_original_message()

def get_decoded_message(coder: ReedMuller):
    '''
    Gets the decoded message from the Reed-Muller encoder.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The decoded message
    '''
    return coder.get_decoded_message()

def decode_message(coder: ReedMuller):
    '''
    Decodes the message using the Reed-Muller decoder.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        list: The decoded message
    '''
    return coder.decode()

def reset(coder: ReedMuller):
    '''
    Resets the Reed-Muller encoder state.

    Args:
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        None
    '''
    coder.reset()

# Helper functions
def toggle_bit(index, coder):
    '''
    Toggles the bit at the specified index in the encoded message.

    Args:
        index (int): The index of the bit to toggle
        coder (ReedMuller): The Reed-Muller encoder object
    
    Returns:
        None
    '''
    print(f"Toggling bit at index {index}")
    coder.flip_mistake_position(index)
# Helper functions
def render_bits(start, end):
    '''
    Renders the bits in the specified range with color coding.
    Args:
        start (int): The starting index of the bits to display
        end (int): The ending index of the bits to display
    
    Returns:
        str: The HTML representation of the bits
    '''
    original_bits = st.session_state.coder.get_encoded_message()
    noisy_bits = st.session_state.coder.get_noisy_message()

    bits_html = "<div class='bit-container'>"
    for i, (orig, curr) in enumerate(zip(original_bits[start:end], noisy_bits[start:end]), start=start):
        color_class = "green" if orig == curr else "red"
        bits_html += (
            f"<div class='bit {color_class}' onclick=\"fetch('/?toggle_bit={i}').then(() => window.location.reload());\">"
            f"{curr}</div>"
        )
    bits_html += "</div>"
    return bits_html

# Add custom CSS for colored bits and image centering
st.markdown(
    """
    <style>
    .bit-container {
        display: flex;
        flex-wrap: wrap;
        overflow-x: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        max-height: 200px; /* Adjust this for scrolling height */
    }
    .bit {
        width: 20px; /* Fixed width for uniform spacing */
        height: 20px;
        margin: 2px;
        text-align: center;
        line-height: 20px;
        border-radius: 3px;
        color: white;
        cursor: pointer;
    }
    .green {
        background-color: green;
    }
    .red {
        background-color: red;
    }
    .css-1v3fvcr {
        justify-content: center;
        text-align: center;
    }
    .css-1b6v1kj {
        justify-content: center;
        text-align: center;
    }
    .stImage {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main function
def main():
    # Initialize session state
    if 'original_bits' not in st.session_state:
        st.session_state['original_bits'] = None
        st.session_state['encoded_bits'] = None
        st.session_state['decoded_message'] = None
        st.session_state['uploaded_file'] = None
        st.session_state['vector'] = False
        st.session_state.current_page = 0
        reset(st.session_state['coder'])

    # App UI
    st.title("Reed-Muller Encoding and Decoding")

    # User input section
    m_value = st.number_input("Enter the value of m:", min_value=1, max_value=100, value=3)
    d_min = 2**(m_value - 1)
    t = int((d_min - 1) / 2)
    st.write(f"Minimum distance (d_min): {d_min}")
    st.write(f"Error correction capability (t): {t}")

    # User input section
    input_type = st.radio("Choose input type:", ("Vector", "Text", "Image"))
    message = None
    vector = False
    not_allowed = True
    if input_type == "Vector":
        message = st.text_input("Enter your message (e.g., 0101):", value="", max_chars=m_value + 1)
        message = ''.join(filter(lambda x: x in '01', message))
        if len(message) != m_value + 1 or not all(bit in '01' for bit in message):
            st.error(f"Message must be exactly {m_value + 1} bits long and contain only '0' and '1'.")
            not_allowed = True
        else:
            st.session_state['uploaded_file'] = None
            st.session_state['original_image'] = None
            st.session_state['original_shape'] = None
            st.session_state['vector'] = True
            not_allowed = False
            
    elif input_type == "Text":
        message = st.text_area("Enter your message:", height=150)
        print(f"Message: {message}")
        st.session_state['uploaded_file'] = None
        st.session_state['original_image'] = None
        st.session_state['original_shape'] = None
        st.session_state['vector'] = False
        if message is not None and len(message.strip()) > 0:
            not_allowed = False

    elif input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["bmp"])
        st.session_state['uploaded_file'] = uploaded_file
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            st.session_state['original_image'] = image
            st.columns([1, 1])
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Convert image to binary
            message, orig_shape = image_to_binary(image)
            st.session_state['original_shape'] = orig_shape  # Save original shape
            st.session_state['encoded_bits'] = None
            st.session_state['not_encoded_image_bits'] = message
            st.session_state['vector'] = False
            not_allowed = False
            
    noise_amount = 0.0
    noise_type = NoiseEnum.LINEAR

    if not not_allowed and m_value and message is not None:
        if st.button("Encode"):
            st.session_state['decoder'] = HadamardTransform(m_value)
            st.session_state['coder'] = ReedMuller(1, m_value, st.session_state['decoder'])
            with st.spinner('Encoding...'):
                start_time = time.time()
                try:
                    if st.session_state['vector']:
                        try:
                            encoded_bits = encode_vector(message, st.session_state['coder'])
                        except ValueError as e:
                            st.error(f"Error: {e}")

                    else:
                        encoded_bits = encode_message(message, st.session_state['coder'])
                except ValueError as e:
                    st.error(f"Error: {e}")
                end_time = time.time()
            elapsed_time = end_time - start_time
            if st.session_state['encoded_bits'] is not None:
                st.write(f"Encoding took {elapsed_time:.2f} seconds")
                st.write(f"Original message length: {len(message)} bits")
                st.write(f"Encoded message length: {len(encoded_bits)} bits")
                st.session_state['original_bits'] = encoded_bits  # Save original bits
                st.session_state['encoded_bits'] = encoded_bits  # Save modifiable bits
                st.session_state['decoded_message'] = None  # Reset decoded message

    if st.session_state['coder'].get_encoded_message() is not None and not not_allowed:
        bit_index = st.number_input("Enter bit index to toggle:", min_value=0, max_value=len(st.session_state['coder'].get_encoded_message()) - 1, step=1)
        if st.button("Toggle Bit"):
            toggle_bit(bit_index, st.session_state['coder'])
    # Noise application
    if get_encoded_message(st.session_state['coder']) is not None and not not_allowed:
        not_allowed = False
        # Noise dropdown and button
        noise_type_list = [NoiseEnum.to_string(nt) for nt in NoiseEnum.list_all()]
        st.session_state['noise_type'] = st.selectbox("Select Noise Type", noise_type_list)
        
        # Initialize session state to store noise amount if not already initialized
        if 'noise_amount' not in st.session_state:
            st.session_state['noise_amount'] = 0.1  # Default value

        # Slider for noise amount (0.0 to 1.0)
        noise_amount_slider = st.slider(
            "Select Noise Amount", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state['noise_amount'], 
            step=0.01,
            on_change=lambda: st.session_state.update({'apply_noise': True})
        )

        # Update session state when slider changes
        st.session_state['noise_amount'] = noise_amount_slider

        # Text input for noise amount with validation (allowing both ',' and '.')
        noise_amount_str = st.text_input(
            "Enter noise amount (use ',' or '.'): ", 
            value=str(st.session_state['noise_amount']),
            on_change=lambda: st.session_state.update({'apply_noise': True})
        )

        # Validate and update noise amount based on user input
        try:
            # Replace ',' with '.' and convert to float
            noise_amount_input = float(noise_amount_str.replace(',', '.'))
            
            # Check if the input value is within the allowed range
            if 0.0 <= noise_amount_input <= 1.0:
                st.session_state['noise_amount'] = noise_amount_input
            else:
                st.error("Invalid noise amount. Please enter a number between 0.0 and 1.0.")

        except ValueError:
            # Display error if input is not a valid float
            st.error("Invalid noise amount. Please enter a valid number.")

        if st.button("Apply Noise") or st.session_state.get('apply_noise', False):
            not_allowed = False
            noise_type = st.session_state['noise_type']
            st.session_state['encoded_bits'] = apply_noise(NoiseEnum.from_string(noise_type), st.session_state['noise_amount'], st.session_state['coder'])
            mistake_positions = get_mistake_positions(st.session_state['coder'])
            mistake_count = len(mistake_positions)
            st.write(f"Mistake count: {mistake_count}")
            if mistake_count > 0:
                if mistake_count < 100:
                    st.write(f"Mistake positions: {mistake_positions}")
            print(f"Applying noise with type {noise_type} and amount {st.session_state['noise_amount']}")
            try:
                st.session_state['not_encoded_image_bits'] = apply_noise_to_original(NoiseEnum.from_string(noise_type), st.session_state['noise_amount'], st.session_state['coder'])
            except:
                pass
            st.session_state['apply_noise'] = False
            

        st.write("Encoded bits:")   
        # Display colored bits in a scrollable, wrapping container
        BITS_PER_PAGE = 100
        start = st.session_state.current_page * BITS_PER_PAGE
        end = min(start + BITS_PER_PAGE, len(st.session_state['coder'].get_noisy_message()))


        
        # Handle query parameter for toggling bits
        # Render bits
        st.write(f"Displaying bits {start + 1} to {end}:")
        bits_html = render_bits(start, end)
        st.markdown(bits_html, unsafe_allow_html=True)

        # Pagination buttons
        col1, col2 = st.columns([3, 3])  # Both columns are equally wide
        with col1:
            if st.button("Previous"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
        with col2:
            if st.button("Next"):
                if end < len(st.session_state.coder.get_encoded_message()):
                    st.session_state.current_page += 1
    # Decode section
    if get_encoded_message(st.session_state['coder']) is not None and not not_allowed:
        if st.button("Decode"):
            if st.session_state['uploaded_file'] is None:
                with st.spinner('Decoding...'):
                    start_time = time.time()
                    decoded_message = decode_message(st.session_state['coder'])
                    end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Decoding took {elapsed_time:.2f} seconds")
                st.session_state['decoded_message'] = decoded_message
                if not st.session_state['vector']:
                    st.session_state['decoded_string'] = Utility.bit_list_to_str(decoded_message)
                decoded_message = message_to_bits(decoded_message)
                st.success(f"Decoded message: {decoded_message}\n\n\n")
                if not st.session_state['vector']:
                    st.success(f"Decoded message: {Utility.bit_list_to_str(decoded_message)}")
                    st.success(f"Original message: {Utility.bit_list_to_str(st.session_state['coder'].get_noisy_original_message())}")
            else:
                with st.spinner('Decoding...'):
                    start_time = time.time()
                    decoded_message = decode_message(st.session_state['coder'])
                    end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Decoding took {elapsed_time:.2f} seconds")
                decoded_message = np.array(decoded_message)
                col1, col2 = st.columns([5, 5])
                with col1:
                    try:
                        not_encoded_image = binary_to_image(st.session_state['coder'].get_noisy_original_message(), st.session_state['original_shape'])
                        st.image(not_encoded_image, caption='Image without encoding', use_container_width=True)
                    except:
                        st.image(st.session_state['original_image'], caption='Image without encoding', use_container_width=True)
                with col2:
                    st.image(binary_to_image(decoded_message, st.session_state['original_shape']), caption='Decoded Image', use_container_width=True)
                    decoded_image = binary_to_image(decoded_message, st.session_state['original_shape'])
    # Reset functionality
    if st.button("Start Over"):
        st.session_state.clear()
        st.query_params.clear()
        st.session_state.current_page = 0
        try:
            reset(st.session_state['coder'])
        except:
            pass
        st.rerun()



if __name__ == "__main__":
    main()
