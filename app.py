import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('handwritten_improved.keras')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    img_array = tf.keras.utils.normalize(img_array, axis=1)  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI Design
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
    }
    h1, h2, h3, h4 {
        color: #1e90ff;
    }
    .stButton>button {
        background-color: #1e90ff;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #4682b4;
        color: white;
    }
    footer {
        text-align: center;
        color: #1e90ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Handwritten Digit Recognition")
st.write("**Upload an image to predict the handwritten digit using the trained AI model.**")

# File uploader for digit prediction
uploaded_file = st.file_uploader("Upload a digit image (JPG/PNG):", type=["jpg", "png", "jpeg"])

# Handle uploaded image
if uploaded_file is not None:
    # Display the uploaded image in a smaller size
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=False, width=150)

    # Preprocess and predict the digit
    img = Image.open(uploaded_file)
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Display the prediction
    st.markdown(f"<h3 style='color:#1e90ff;'>Predicted Digit: {predicted_digit}</h3>", unsafe_allow_html=True)
    st.write(f"**Confidence Level:** {confidence:.4f}")


# Footer
st.markdown("---")
st.markdown(
    """
    <footer>
        <h4>Developed for District-Level Exhibition</h4>
        <p>Group Members:</p>
        <p>1) Sara Saleem</p>
        <p>2) Aiza Bibi</p>
        <p>Class: BS IT, Information Technology, 7th Semester</p>
        <p>College: Viqar-un-Nisa Post Graduate College for Women, Rawalpindi</p>
        <p>Powered by Streamlit & TensorFlow</p>
    </footer>
    """,
    unsafe_allow_html=True
)
