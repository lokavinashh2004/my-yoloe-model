import streamlit as st
import cv2
import numpy as np
import io
import warnings
from ultralytics import YOLO
from PIL import Image

# Suppress only specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load YOLOv8 model (best version)
model = YOLO("yolov8x.pt")

# Custom CSS for better UI
st.markdown("""
    <style>
        .main {background-color: #f5f5f5;}
        .stTitle {color: #ff5733; font-size: 32px; font-weight: bold;}
        .stFileUploader label {font-size: 18px; color: #333;}
        .stImage {border-radius: 10px;}
        .download-btn {background-color: #28a745; color: white; font-size: 16px; padding: 10px; border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<h1 class='stTitle'>🔍 Object Detection</h1>", unsafe_allow_html=True)
st.write("📷 Upload an image, and the model will detect and label objects!")

# Upload image
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded image to an OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Ensure image has 3 color channels (RGB)
    if len(image.shape) == 2:  # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # Convert RGBA to RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Perform object detection
    results = model(image)

    # Process results and draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            confidence = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = model.names.get(cls, "Unknown")  # Class name with safety check

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Resize image while maintaining aspect ratio
    max_width = 800  # Adjust width as needed
    height, width, _ = image.shape
    aspect_ratio = height / width
    new_height = int(max_width * aspect_ratio)
    resized_image = cv2.resize(image, (max_width, new_height))

    # Convert image back to PIL format for Streamlit display
    result_image = Image.fromarray(resized_image)
    st.image(result_image, caption="🎯 Detected Objects", use_container_width=True)

    # Download button
    img_bytes = io.BytesIO()
    result_image.save(img_bytes, format="PNG")
    st.download_button(
        label="📥 Download Processed Image",
        data=img_bytes.getvalue(),
        file_name="detected_objects.png",
        mime="image/png",
        help="Click to download the resized image with detected objects."
    )
