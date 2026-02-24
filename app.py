import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from gtts import gTTS
import io

# --- 1. SETTING UP MODELS (CACHED) ---
# We use @st.cache_resource so the models load only once when the app starts
@st.cache_resource
def load_models():
    # Load YOLOv10 Nano (Very fast for web)
    det_model = YOLO("yolov10s.pt")
    # Load Depth Anything V2 (Small for faster inference on Streamlit Cloud)
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
    return det_model, depth_pipe

det_model, depth_pipe = load_models()

# --- 2. CONFIGURATION ---
# Use your calibrated values here!
KNOWN_DISTANCE = 1.5  # meters
KNOWN_RAW_VALUE = 153 
SCALE_FACTOR = KNOWN_DISTANCE * KNOWN_RAW_VALUE

# --- 3. APP UI ---
st.title("Spatial AI Assistant 🤖")
st.subheader("Detect objects and estimate distance in real-time")

# Camera Input Widget (This replaces your drive/wget logic)
img_file_buffer = st.camera_input("Take a photo of your room")

if img_file_buffer is not None:
    # Convert the buffer to a PIL image
    raw_image = Image.open(img_file_buffer)
    img_cv = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    
    # 4. AI INFERENCE
    with st.spinner('Analyzing space...'):
        results = det_model(raw_image)[0]
        depth_output = depth_pipe(raw_image)["depth"]
        depth_map = np.array(depth_output)

    detections_for_voice = []

    # 5. FUSION LOGIC
    for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = det_model.names[int(box.cls[0])]
            
            # 1. Define the area of the object in the depth map
            crop_depth = depth_map[y1:y2, x1:x2]

            if crop_depth.size > 0:
                # 2. Calculate the median raw value FIRST
                raw_val = np.median(crop_depth)
                
                # 3. Calculate distance SECOND
                dist_m = SCALE_FACTOR / raw_val if raw_val > 0 else 0
                
                # 4. Filter and add to voice list
                if 0.1 < dist_m < 15.0:
                    detections_for_voice.append(f"a {label} at {dist_m:.1f} meters")
                
                # 5. Visuals (Now dist_m is guaranteed to exist)
                color = (0, 255, 0) if dist_m > 1.0 else (0, 0, 255)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_cv, f"{label}: {dist_m:.2f}m", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
            # Draw visuals (BGR for OpenCV)
            color = (0, 255, 0) if dist_m > 1.0 else (0, 0, 255)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_cv, f"{label}: {dist_m:.2f}m", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 6. DISPLAY RESULTS
    # Streamlit uses RGB, so we convert back
    st.image(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB), caption="Spatial Analysis Result")

    # 7. VOICE OUTPUT (REPLACES ipd.Audio)
    if detections_for_voice:
        full_text = "I see " + " and ".join(detections_for_voice)
        st.write(f"📢 **Voice Report:** {full_text}")
        
        # Create audio in-memory (fastest for web)
        tts = gTTS(text=full_text, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        
        # Play audio with autoplay
        st.audio(audio_fp.getvalue(), format="audio/mp3", autoplay=True)