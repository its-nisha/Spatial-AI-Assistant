import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from gtts import gTTS
import io

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Spatial AI Assistant", layout="wide")

# Custom CSS for a better mobile experience
st.markdown("""
    <style>
    div[data-testid="stCameraInput"] { width: 100% !important; }
    img { width: 100% !important; height: auto !important; border-radius: 10px; }
    .stSecondaryBlock { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. MODELS (CACHED) ---
@st.cache_resource
def load_models():
    # Small model for better balance of accuracy vs speed
    det_model = YOLO("yolov10s.pt") 
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    return det_model, depth_pipe

det_model, depth_pipe = load_models()

# --- 3. SIDEBAR CALIBRATION ---
st.sidebar.title("⚙️ Calibration Engine")
st.sidebar.info("Adjust these values to match your specific camera hardware.")

with st.sidebar.expander("Near Point (30cm - 1m)", expanded=True):
    p1_dist = st.number_input("Near Distance (m)", value=0.5, step=0.1)
    p1_raw = st.number_input("Near Raw Value", value=240)

with st.sidebar.expander("Far Point (1.5m - 4m)", expanded=True):
    p2_dist = st.number_input("Far Distance (m)", value=2.0, step=0.1)
    p2_raw = st.number_input("Far Raw Value", value=120)

# --- 4. CALCULATION LOGIC ---
def get_metric_distance(raw_val):
    if raw_val <= 0 or (p1_raw == p2_raw): return 0
    try:
        # Inverse Linear Regression for non-linear depth sensors
        inv_raw = 1.0 / raw_val
        inv_p1 = 1.0 / p1_raw
        inv_p2 = 1.0 / p2_raw
        
        slope = (p2_dist - p1_dist) / (inv_p2 - inv_p1)
        distance = p1_dist + slope * (inv_raw - inv_p1)
        return max(0.1, distance) 
    except:
        return 0

# --- 5. MAIN INTERFACE ---
st.title("Spatial AI Assistant 🤖")
st.write("A proof-of-concept for accessible indoor navigation.")

img_file_buffer = st.camera_input("Capture your environment")

if img_file_buffer:
    raw_image = Image.open(img_file_buffer)
    img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    
    with st.spinner('Analyzing spatial layout...'):
        results = det_model(raw_image)[0]
        depth_map = np.array(depth_pipe(raw_image)["depth"])

    detections_for_voice = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = det_model.names[int(box.cls[0])]
        
        # Median sampling to ignore outliers (reflections/edges)
        crop_depth = depth_map[y1:y2, x1:x2]
        if crop_depth.size > 0:
            raw_val = np.median(crop_depth)
            dist_m = get_metric_distance(raw_val)
            
            # Filter for realistic indoor range
            if 0.1 < dist_m < 10.0:
                detections_for_voice.append(f"a {label} at {dist_m:.1f} meters")

                # UI Overlay
                color = (0, 255, 0) if dist_m > 1.2 else (0, 0, 255) # Red for close objects
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_cv, f"{label}: {dist_m:.2f}m (Raw: {int(raw_val)})", 
                            (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- 6. VOICE FEEDBACK ---
    if detections_for_voice:
        full_text = "I see " + " and ".join(detections_for_voice)
        st.success(f"**Voice Report:** {full_text}")
        
        tts = gTTS(text=full_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        st.audio(audio_fp.getvalue(), format="audio/mp3", autoplay=True)