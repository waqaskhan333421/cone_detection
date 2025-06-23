import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import yaml

# Load your trained model
model = YOLO(r"D:\Ayaz\projects\New folder (2)\Cone Detector Computer Vision Project\runs\detect\train3\weights\best.pt")

# Load class names from data.yaml
with open(r"D:\Ayaz\projects\New folder (2)\Cone Detector Computer Vision Project\shuttle-run-1\data.yaml", "r") as f:
    data_yaml = yaml.safe_load(f)
CLASS_NAMES = data_yaml["names"]

st.title("Automated Cone Detection for Formula student cars")

option = st.sidebar.selectbox(
    "Choose input method:",
    ("Upload Image", "Use Webcam")
)

def detect_and_display(image):
    results = model(image)
    result_img = results[0].plot()  # Draw boxes on image
    st.image(result_img, caption="Detected Cones", use_container_width=True)
    # Show detected classes and confidence
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        st.write(f"Class: {CLASS_NAMES[class_id]}, Confidence: {conf:.2f}")

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        detect_and_display(np.array(image))

elif option == "Use Webcam":
    st.write("Click 'Start' to use your webcam.")
    run = st.button("Start")
    stop = st.button("Stop")
    camera = None

    if run:
        camera = cv2.VideoCapture(0)
        stframe = st.empty()
        while True:
            ret, frame = camera.read()
            if not ret:
                st.write("Failed to grab frame")
                break
            # YOLO expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            result_img = results[0].plot()
            stframe.image(result_img, channels="RGB")
            if stop:
                break
        camera.release()
        cv2.destroyAllWindows()