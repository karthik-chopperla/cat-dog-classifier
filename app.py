import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat vs Dog Image Classifier")

uploaded_file = st.file_uploader("Choose an image of dog or cat...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).resize((224, 224))
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Classify"):
            st.info("Classifying...")
            model = MobileNetV2(weights="imagenet")
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = model.predict(x)
            decoded = decode_predictions(preds, top=1)[0][0]
            label = decoded[1]
            confidence = decoded[2] * 100

            if "cat" in label.lower():
                st.success(f"Prediction: 🐱 Cat ({confidence:.2f}%)")
            elif "dog" in label.lower():
                st.success(f"Prediction: 🐶 Dog ({confidence:.2f}%)")
            else:
                st.warning(f"Not sure! Detected: {label} ({confidence:.2f}%)")
    except Exception as e:
        st.error(f"Something went wrong with the image: {e}")
else:
    st.warning("Please upload an image to begin.")
