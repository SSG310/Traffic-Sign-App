import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Load the trained model
model = tf.keras.models.load_model("D:/ML_proj(1)/model/model.keras")

# Dictionary for all 43 traffic sign classes
classes = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End of all speed and passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End of no passing for vehicles over 3.5 metric tons'
}

# Streamlit UI
st.title("Traffic Sign Recognition")
st.write("Upload an image of a traffic sign, and the app will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image to 30x30 as per your model's input
    image = image.resize((30, 30))
    image = np.array(image)
    if image.shape[2] == 4:  # Remove alpha channel if it exists
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize

    # Make a prediction
    predictions = model.predict(image)
    class_id = np.argmax(predictions)
    class_name = classes[class_id]

    # Display the prediction
    st.write(f"Predicted Class: {class_name}")
