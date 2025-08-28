import streamlit as st 
import cv2
import numpy as np
import pickle
from PIL import Image

st.title("🐶🐱 Cat And Dog Classification Using CNN")
st.write("Made By Subhadip 😎")

# Load the trained model
model = pickle.load(open(r"C:\Users\sinha\Desktop\CAT vs DOG\model", "rb"))

# File uploader
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])x

if uploaded_image is not None:
    st.divider()
    st.header("🖼️ Your Uploaded Image")
    
    # Open with PIL
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption=f"You uploaded: {uploaded_image.name}", use_column_width=True)

    # Convert to numpy for OpenCV
    img_array = np.array(image)

    # Resize to match model input
    test_img = cv2.resize(img_array, (256, 256))

    # Normalize (important for CNN models)
    test_img = test_img / 255.0

    # Reshape to match input shape (1, 256, 256, 3)
    img = test_img.reshape(1, 256, 256, 3)

    # Prediction button
    if st.button("Predict"):
        result = model.predict(img)

        # Handle prediction format (depends on training)
        if result[0] == 1 or result[0] > 0.5:
            st.success("🐶 It's a Dog!")
        else:
            st.success("🐱 It's a Cat!")
