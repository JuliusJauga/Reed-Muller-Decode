import streamlit as st
from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import NoiseEnum
from ReedMuller import Utility
import time
import numpy as np
from PIL import Image
import os



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



# Initialize Reed-Muller and Hadamard transform objects
if 'decoder' not in st.session_state:
    st.session_state['decoder'] = HadamardTransform(3)
if 'coder' not in st.session_state:
    st.session_state['coder'] = ReedMuller(1, 3, st.session_state['decoder'])


# Placeholder for your backend functions
def encode_message(message: str, coder: ReedMuller) -> str:
    coder.set_message(message)
    return coder.encode()


def apply_noise(noise_type: NoiseEnum, noise_amount: float, coder: ReedMuller) -> str:
    coder.apply_noise(noise_type, noise_amount)
    return coder.get_noisy_message()

def set_message(message: str, coder: ReedMuller) -> None:
    coder.set_message(message)

def get_encoded_message(coder: ReedMuller) -> str:
    return coder.get_encoded_message()

def get_noisy_message(coder: ReedMuller) -> str:
    return coder.get_noisy_message()

def get_decoded_message(coder: ReedMuller) -> str:
    return coder.get_decoded_message()

def decode_message(coder: ReedMuller) -> str:
    return coder.decode()

# Helper functions
def toggle_bit(index, coder):
    """Toggle a noisy bit and update its value."""
    print(f"Toggling bit at index {index}")
    coder.flip_mistake_position(index)
# Helper functions
def render_bits(start, end):
    """Generate HTML for the bits display."""
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
        st.session_state.current_page = 0


    # App UI
    st.title("Reed-Muller Encoding and Decoding")

    # User input section
    input_type = st.radio("Choose input type:", ("Text", "Image"))
    message = None
    if input_type == "Text":
        message = st.text_input("Enter your message:")
        st.session_state['uploaded_file'] = None
        st.session_state['original_image'] = None
        st.session_state['original_shape'] = None
        st.session_state['MESSAGE_NOT_ENCODED'] = True
    elif input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["bmp"])
        st.session_state['uploaded_file'] = uploaded_file
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.session_state['original_image'] = image
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Convert image to binary
            message, orig_shape = image_to_binary(image)
            st.image(binary_to_image(message, orig_shape), caption='Converted Image', use_container_width=True)
            st.session_state['original_shape'] = orig_shape  # Save original shape
            st.session_state['encoded_bits'] = None
            st.session_state['MESSAGE_NOT_ENCODED'] = True
            
    m_value = st.number_input("Enter the value of m:", min_value=1, max_value=100, value=3)
    noise_amount = 0.0
    noise_type = NoiseEnum.LINEAR

    if m_value and message is not None:
        if st.button("Encode"):
            st.session_state['decoder'] = HadamardTransform(m_value)
            st.session_state['coder'] = ReedMuller(1, m_value, st.session_state['decoder'])
            with st.spinner('Encoding...'):
                start_time = time.time()
                try:
                    encoded_bits = encode_message(message, st.session_state['coder'])
                    st.session_state['MESSAGE_NOT_ENCODED'] = False
                except ValueError as e:
                    st.error(f"Error: {e}")
                end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Encoding took {elapsed_time:.2f} seconds")
            st.write(f"Original message length: {len(message)} bits")
            st.write(f"Encoded message length: {len(encoded_bits)} bits")
            st.session_state['original_bits'] = encoded_bits  # Save original bits
            st.session_state['encoded_bits'] = encoded_bits  # Save modifiable bits
            st.session_state['decoded_message'] = None  # Reset decoded message

    if st.session_state['coder'].get_encoded_message() is not None:
        bit_index = st.number_input("Enter bit index to toggle:", min_value=0, max_value=len(st.session_state['coder'].get_encoded_message()) - 1, step=1)
        if st.button("Toggle Bit"):
            toggle_bit(bit_index, st.session_state['coder'])
    # Noise application
    if get_encoded_message(st.session_state['coder']) is not None:
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
            noise_type = st.session_state['noise_type']
            st.session_state['encoded_bits'] = apply_noise(NoiseEnum.from_string(noise_type), st.session_state['noise_amount'], st.session_state['coder'])
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
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Previous"):
                if st.session_state.current_page > 0:
                    st.session_state.current_page -= 1
        with col2:
            if st.button("Next"):
                if end < len(st.session_state.coder.get_encoded_message()):
                    st.session_state.current_page += 1

    if st.session_state['uploaded_file'] is not None:
        # Checkbox for order of operations
        order_of_operations = st.checkbox("m first or noise type first", value=True)

        

        if st.button("Automatic Encode Decode and Save"):
            noise_intervals = np.arange(0, 1.1, 0.1)  # Noise intervals from 0 to 1 with step 0.1
            noise_types = NoiseEnum.list_all()
            if order_of_operations == False:
                for noise_type in noise_types:
                    for noise_amount in noise_intervals:
                        for m in range(2, 8):
                            output_dir = f"decodedImages/{NoiseEnum.to_string(noise_type)}"
                            output_path = os.path.join(output_dir, f"katukai_m{m}_noise{NoiseEnum.to_string(noise_type)}_amount{noise_amount:.1f}.bmp")
                            if os.path.exists(output_path):
                                continue                        
                            st.session_state.pop('decoder', None)
                            st.session_state.pop('coder', None)
                            st.session_state['decoder'] = HadamardTransform(m)
                            st.session_state['coder'] = ReedMuller(1, m, st.session_state['decoder'])
                            encoded_bits = encode_message(message, st.session_state['coder'])
                            st.session_state['encoded_bits'] = apply_noise(noise_type, noise_amount, st.session_state['coder'])
                            # Check if the decoded image already exists
                            if not os.path.exists(output_path):
                                decoded_message = decode_message(st.session_state['coder'])
                                decoded_message = np.array(decoded_message)
                                decoded_image = binary_to_image(decoded_message, st.session_state['original_shape'])
                                os.makedirs(output_dir, exist_ok=True)
                                decoded_image.save(output_path)
                            # Let memory clear
                            time.sleep(10)
            if order_of_operations == True:
                for m in range(1, 8):
                    st.session_state['decoder'] = HadamardTransform(m)
                    st.session_state['coder'] = ReedMuller(1, m, st.session_state['decoder'])
                    encoded_bits = encode_message(message, st.session_state['coder'])
                    for noise_type in noise_types:
                        for noise_amount in noise_intervals:
                            st.session_state['coder'].apply_noise(noise_type, noise_amount)
                            # Check if the decoded image already exists
                            output_dir = f"decodedImages/{NoiseEnum.to_string(noise_type)}"
                            output_path = os.path.join(output_dir, f"katukai_m{m}_noise{NoiseEnum.to_string(noise_type)}_amount{noise_amount:.1f}.bmp")
                            if not os.path.exists(output_path):
                                decoded_message = decode_message(st.session_state['coder'])
                                decoded_message = np.array(decoded_message)
                                decoded_image = binary_to_image(decoded_message, st.session_state['original_shape'])
                                os.makedirs(output_dir, exist_ok=True)
                                decoded_image.save(output_path)
            st.success("All decoded images saved successfully.")


    # Decode section
    if get_encoded_message(st.session_state['coder']) is not None:
        if st.button("Decode"):
            if st.session_state['uploaded_file'] is None:
                with st.spinner('Decoding...'):
                    start_time = time.time()
                    decoded_message = decode_message(st.session_state['coder'])
                    end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Decoding took {elapsed_time:.2f} seconds")
                st.session_state['decoded_message'] = decoded_message
                st.session_state['decoded_string'] = Utility.np_bit_array_to_str(np.array(decoded_message))
                st.success(f"Decoded message: {decoded_message}\n\n\nDecoded string: {st.session_state['decoded_string']}")
            else:
                with st.spinner('Decoding...'):
                    start_time = time.time()
                    decoded_message = decode_message(st.session_state['coder'])
                    end_time = time.time()
                elapsed_time = end_time - start_time
                st.write(f"Decoding took {elapsed_time:.2f} seconds")
                decoded_message = np.array(decoded_message)
                col1, col2 = st.columns([2, 2])
                with col1:
                    st.image(st.session_state['original_image'], caption='Original Image', use_container_width=True)
                with col2:
                    st.image(binary_to_image(decoded_message, st.session_state['original_shape']), caption='Decoded Image', use_container_width=True)
                    # Save the decoded image
                    decoded_image = binary_to_image(decoded_message, st.session_state['original_shape'])
                    # Create directory if it doesn't exist
                    output_dir = f"decodedImages/{noise_type}"
                    os.makedirs(output_dir, exist_ok=True)
                    # Save the decoded image
                    decoded_image.save(os.path.join(output_dir, f"decoded_image_m{m_value}_noise{noise_type}_amount{st.session_state['noise_amount']}.bmp"))
                    st.success("Decoded image saved successfully.")
                
    # Reset functionality
    if st.button("Start Over"):
        st.session_state.clear()
        st.query_params.clear()  # Updated to use the non-experimental method
        st.rerun()  # Rerun the app to reset the state



if __name__ == "__main__":
    main()
