import streamlit as st
import numpy as np
import io
from PIL import Image
import tensorflow as tf


def load_model():
    return tf.keras.models.load_model("./models/model.keras")


model = load_model()

accuracy_of_model = 91.66

class_names = ["Early blight", "Late blight", "healthy"]


def predict_leaf_class(image_as_array):
    try:
        prediction = model.predict(image_as_array[np.newaxis, ...])
        if class_names[np.argmax(prediction)] == "healthy":
            st.markdown(f"#### The above leaf is healthy")
        else:
            st.markdown(
                f"#### The above leaf is suffering from {class_names[np.argmax(prediction)]}"
            )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


st.title("Welcome To :green[Leaf] Disease Predictor")

st.divider()

st.info(
    f"Note: This model is {accuracy_of_model}% accurate so it may make mistakes, consider confirming the information."
)

uploaded_file = st.file_uploader(
    "Drag a :leaves: image here or upload",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        image_as_array = np.array(image)
        st.image(uploaded_file)
        predict_leaf_class(image_as_array)
        st.warning(
            "Warning: If you reload the page, the uploaded image and prediction will disappear and then you have to reupload the image again to get the prediction."
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
