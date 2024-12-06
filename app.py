import streamlit as st
from ReedMuller import ReedMuller
from ReedMuller import HadamardTransform
from ReedMuller import NoiseEnum
from ReedMuller import Utility
import time
import cv2
import numpy as np
from PIL import Image

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


def decode_message(coder: ReedMuller) -> str:
    return coder.decode()

# Helper functions
def toggle_bit(index):
    """Toggle a noisy bit and update its value."""
    st.session_state['encoded_bits'] = st.session_state['coder'].flip_mistake_position(index)

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

    if input_type == "Text":
        message = st.text_input("Enter your message:")
    elif input_type == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["bmp"])
        st.session_state['uploaded_file'] = uploaded_file
        if uploaded_file is not None:

            image = Image.open(uploaded_file)
            st.session_state['original_image'] = image
            st.image(image, caption='Uploaded Image', use_column_width=True)
            image = np.array(image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
            message = np.packbits(binary_image.flatten())
            unpacked_message = np.unpackbits(message)
            unpacked_bit_list = unpacked_message.tolist()
            message = unpacked_bit_list
            print(unpacked_bit_list)
    m_value = st.number_input("Enter the value of m:", min_value=1, max_value=100, value=3)

    if st.button("Encode") and message and m_value:
        st.session_state['decoder'] = HadamardTransform(m_value)
        st.session_state['coder'] = ReedMuller(1, m_value, st.session_state['decoder'])
        with st.spinner('Encoding...'):
            start_time = time.time()
            encoded_bits = encode_message(message, st.session_state['coder'])
            end_time = time.time()
        elapsed_time = end_time - start_time
        st.write(f"Encoding took {elapsed_time:.2f} seconds")
        st.session_state['original_bits'] = encoded_bits  # Save original bits
        st.session_state['encoded_bits'] = encoded_bits  # Save modifiable bits
        st.session_state['decoded_message'] = None  # Reset decoded message
        # st.success(f"Message encoded: {encoded_bits}")

    # Noise application
    if st.session_state['encoded_bits']:
        # Noise dropdown and button
        noise_type_list = [NoiseEnum.to_string(nt) for nt in NoiseEnum.list_all()]
        noise_type = st.selectbox("Select Noise Type", noise_type_list)
        
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
        
        
        
        
        
        
        
        # with st.container():
        #     original_bits = st.session_state['coder'].get_encoded_message()
        #     current_bits = st.session_state['coder'].get_noisy_message()

        #     # CSS for styling the bits display
        #     st.markdown(
        #         """
        #         <style>
        #         .bit-container {
        #             display: flex;
        #             flex-wrap: wrap;
        #             overflow-x: auto;
        #             padding: 10px;
        #             border: 1px solid #ccc;
        #             border-radius: 5px;
        #             max-height: 200px; /* Adjust this for scrolling height */
        #         }
        #         .bit {
        #             width: 20px; /* Fixed width for uniform spacing */
        #             height: 20px;
        #             margin: 2px;
        #             text-align: center;
        #             line-height: 20px;
        #             border-radius: 3px;
        #             color: white;
        #         }
        #         .green {
        #             background-color: green;
        #         }
        #         .red {
        #             background-color: red;
        #         }
        #         </style>
        #         """,
        #         unsafe_allow_html=True,
        #     )

        #     # Generate HTML for bits
        #     bits_html = "<div class='bit-container'>"
        #     for orig, curr in zip(original_bits, current_bits):
        #         color_class = "green" if orig == curr else "red"
        #         bits_html += f"<div class='bit {color_class}'>{curr}</div>"
        #     bits_html += "</div>"
            

        #     # Render the bits display
        #     st.markdown(bits_html, unsafe_allow_html=True)

    # Decode section
    if st.button("Decode") and st.session_state['encoded_bits']:
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
            print(decoded_message)
            decoded_image = np.reshape(decoded_message, st.session_state['original_image'].size[::-1])
            st.image(decoded_image, caption='Decoded Image', use_column_width=True)
            st.image(st.session_state['original_image'], caption='Original Image', use_column_width=True)

    # Reset functionality
    if st.button("Start Over"):
        st.session_state.clear()
        st.query_params.clear()  # Updated to use the non-experimental method
        st.rerun()  # Rerun the app to reset the state



if __name__ == "__main__":
    main()
