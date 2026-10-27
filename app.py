# =========================================================================
# FINAL RUNTIME MATRIX: CV2-FREE STABLE DESIGN WITH PIL & NUMPY RENDERING
# =========================================================================
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import urllib.request

# Interface setup configuration layout optimization
st.set_page_config(page_title="Ultra 3D Classifier", layout="wide")
st.title("🤖 Ultra Pro 3D Mesh Engine & Decision Tree Classifier")
st.write("Live webcam object tracking and recognition with simulated 3D vector space graphics (Stable PIL Version).")

@st.cache_resource
def load_vision_weights():
    """
    Checks if model weights file exists, otherwise dynamically pulls it from server clusters.
    """
    weights_filename = "yolov8n.pt"
    if not os.path.exists(weights_filename):
        with st.spinner("Downloading base anchor model weights from secure server..."):
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
        # Convert streamlit stream frame to numpy array format safely
        img_rgb = frame.to_ndarray(format="rgb24")
        
        # Run inference using the loaded model weights directly on RGB array
        prediction_results = vision_model(img_rgb, conf=0.40, stream=True)
        current_frame_inventory = {}
        
        # Convert numpy array to PIL Image for rendering shapes without cv2 dependency
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
                
                # --- ULTRA PRO 3D MODEL ISOMETRIC VECTOR GRAPHICS ENGINE (PIL VERSION) ---
                box_width = x_end - x_start
                box_height = y_end - y_start
                depth_offset = int(min(box_width, box_height) * 0.25)
                
                # Coordinates calculation for front and back faces
                x_start_b, y_start_b = x_start + depth_offset, y_start - depth_offset
                x_end_b, y_end_b = x_end + depth_offset, y_end - depth_offset
                
                # 1. Draw Back Plane Face Geometry (Dark Orange Line Shift)
                draw.rectangle([x_start_b, y_start_b, x_end_b, y_end_b], outline=(180, 105, 50), width=1)
                
                # 2. Draw Connecting Depth Lines (3D Mesh wireframe simulation)
                draw.line([(x_start, y_start), (x_start_b, y_start_b)], fill=(0, 255, 255), width=2)
                draw.line([(x_end, y_start), (x_end + depth_offset, y_start_b)], fill=(0, 255, 255), width=2)
                draw.line([(x_start, y_end), (x_start_b, y_end_b)], fill=(0, 255, 255), width=2)
                draw.line([(x_end, y_end), (x_end_b, y_end_b)], fill=(0, 255, 255), width=2)
                
                # 3. Draw Front Facing Primary Geometry Plane (Neon Cyan/Blue Glow)
                draw.rectangle([x_start, y_start, x_end, y_end], outline=(0, 255, 255), width=2)
                
                # Draw Interactive HUD overlay string labels text
                overlay_string = f"[{detected_name.upper()}] {box_confidence * 100:.1f}%"
                draw.text((x_start, y_start - 15), overlay_string, fill=(0, 255, 255))

        # Trigger processing check on background Decision Tree system routing layer
        ml_system_verdict = run_decision_tree_classifier(current_frame_inventory)
        
        # Render a sleek modern HUD Status Board Dashboard banner at the top of the interface
        draw.rectangle([15, 15, 650, 60], fill=(0, 0, 0))
        draw.rectangle([15, 15, 650, 60], outline=(0, 255, 255), width=1)
        draw.text((25, 30), ml_system_verdict, fill=(0, 255, 0))
        
        # Convert PIL back to standard NumPy frame structure format for client browser
        return frame.from_ndarray(np.array(pil_img), format="rgb24")

# Launch direct Client Camera Portal view frame channel
webrtc_streamer(
    key="isometric-3d-pil-decision-module",
    video_processor_factory=Isometric3DPILProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:://google.com"]}]},
    media_stream_constraints={"video": True, "audio": False}
)
