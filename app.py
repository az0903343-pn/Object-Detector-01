# =========================================================================
# HARDCORE RUNTIME OVERRIDE: FORCE FIXES PY3.14 OPENCV IMPORT ERROR
# =========================================================================
import os
import sys
import subprocess

# 1. Force remove defective system path associations to isolate OpenCV
sys.path = [p for p in sys.path if "site-packages/cv2" not in p]

# 2. Programmatically force install headless server-friendly binaries
try:
    import cv2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--force-reinstall", "--no-cache-dir"])
    import cv2

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit-webrtc"])
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    from ultralytics import YOLO

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw

# Setup layout matrices
st.set_page_config(page_title="Ultra 3D Engine", layout="wide")
st.title("🤖 Ultra Pro 3D Mesh Engine & Decision Tree Classifier")
st.write("Live webcam object tracking and recognition with simulated 3D vector space graphics.")

@st.cache_resource
def load_vision_weights():
    weights_filename = "yolov8n.pt"
    if not os.path.exists(weights_filename):
        with st.spinner("Downloading base anchor model weights from secure server..."):
            import urllib.request
            url = "https://github.com"
            urllib.request.urlretrieve(url, weights_filename)
    return YOLO(weights_filename)

vision_model = load_vision_weights()

def run_decision_tree_classifier(frame_inventory):
    computer_hardware = frame_inventory.get('laptop', 0) + frame_inventory.get('keyboard', 0) + frame_inventory.get('mouse', 0)
    classroom_furniture = frame_inventory.get('chair', 0) + frame_inventory.get('book', 0) + frame_inventory.get('dining table', 0)
    home_appliances = frame_inventory.get('refrigerator', 0) + frame_inventory.get('microwave', 0) + frame_inventory.get('oven', 0) + frame_inventory.get('tv', 0)

    if computer_hardware >= 2:
        return "🌲 DECISION VERDICT: COMPUTER LABORATORY WORKSPACE"
    elif home_appliances >= 1 and computer_hardware == 0:
        return "🌲 DECISION VERDICT: SMART HOME APPLIANCE ZONE"
    elif classroom_furniture >= 1:
        return "🌲 DECISION VERDICT: CLASSROOM / ACADEMIC ENVIRONMENT"
    else:
        return "🔍 ENGINE STATUS: ACTIVE BACKGROUND SCANNING MODE..."

class Isometric3DPILProcessor(VideoProcessorBase):
    def recv(self, frame):
        img_rgb = frame.to_ndarray(format="rgb24")
        prediction_results = vision_model(img_rgb, conf=0.40, stream=True)
        current_frame_inventory = {}
        
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)
        
        for frame_data in prediction_results:
            bounding_boxes = frame_data.boxes
            for individual_box in bounding_boxes:
                class_index = int(individual_box.cls)
                detected_name = vision_model.names[class_index]
                current_frame_inventory[detected_name] = current_frame_inventory.get(detected_name, 0) + 1
                
                x_start, y_start, x_end, y_end = map(int, individual_box.xyxy)
                box_confidence = float(individual_box.conf)
                
                box_width = x_end - x_start
                box_height = y_end - y_start
                depth_offset = int(min(box_width, box_height) * 0.25)
                
                x_start_b, y_start_b = x_start + depth_offset, y_start - depth_offset
                x_end_b, y_end_b = x_end + depth_offset, y_end - depth_offset
                
                draw.rectangle([x_start_b, y_start_b, x_end_b, y_end_b], outline=(180, 105, 50), width=1)
                draw.line([(x_start, y_start), (x_start_b, y_start_b)], fill=(0, 255, 255), width=2)
                draw.line([(x_end, y_start), (x_end + depth_offset, y_start_b)], fill=(0, 255, 255), width=2)
                draw.line([(x_start, y_end), (x_start_b, y_end_b)], fill=(0, 255, 255), width=2)
                draw.line([(x_end, y_end), (x_end_b, y_end_b)], fill=(0, 255, 255), width=2)
                draw.rectangle([x_start, y_start, x_end, y_end], outline=(0, 255, 255), width=2)
                
                overlay_string = f"[{detected_name.upper()}] {box_confidence * 100:.1f}%"
                draw.text((x_start, y_start - 15), overlay_string, fill=(0, 255, 255))

        ml_system_verdict = run_decision_tree_classifier(current_frame_inventory)
        draw.rectangle([15, 15, 650, 60], fill=(0, 0, 0))
        draw.rectangle([15, 15, 650, 60], outline=(0, 255, 255), width=1)
        draw.text((25, 30), ml_system_verdict, fill=(0, 255, 0))
        
        return frame.from_ndarray(np.array(pil_img), format="rgb24")

webrtc_streamer(
    key="isometric-3d-pil-decision-module",
    video_processor_factory=Isometric3DPILProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
