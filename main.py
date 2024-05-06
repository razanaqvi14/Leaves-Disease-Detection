import streamlit as st
import numpy as np
import io
from PIL import Image
import tensorflow as tf

page_title = "Potato Leaf Health Assessment System"
page_icon = ":potato:"
layout = "centered"

accuracy_of_model = 95

st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)

st.title("Welcome to :orange[Potato] :green[Leaf] Health Assessment System")

st.divider()

st.info(
    f"Note: A new and a better version of model is here with the accuracy of {accuracy_of_model}%. consider confirming the information.\n\nThings to remember:\n - The image should contain a leaf, and the leaf itself should belong to a potato; otherwise, the model will not be able to assess that leaf correctly.\n - Before reuploading, please remove the uploaded image by clicking on the cross on the right side of the image's name or it will give errors.\n - If you reload the page, the uploaded image and prediction will be lost and you have to reupload the image to assess the leaf."
)


def load_model():
    return tf.keras.models.load_model("./models/1.keras")


model = load_model()

class_names = ["Early blight", "Late blight", "healthy"]


def predict_leaf_class(image_as_array):
    try:
        prediction = model.predict(image_as_array[np.newaxis, ...])
        if class_names[np.argmax(prediction)] == "healthy":
            st.success("The above leaf is healthy")
        else:
            st.error(
                f"The above leaf is suffering from {class_names[np.argmax(prediction)]}"
            )
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


uploaded_file = st.file_uploader(
    "Drag a :leaves: image here or upload",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        image_as_array = np.array(image)
        st.image(uploaded_file)
        if st.button("Detect", type="primary"):
            predict_leaf_class(image_as_array)
    except Exception as e:
        st.error(f"An error occurred: {e}")
