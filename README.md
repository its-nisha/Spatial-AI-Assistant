 Spatial AI Assistant: Monocular Depth & Object Detection
 
 Project Overview
 
 This project is a Responsible AI prototype designed to assist in indoor navigation and spatial awareness. By fusing Real-Time Object Detection with Monocular Depth Estimation, the app identifies objects in a room and calculates their physical distance from the user using a single 2D camera feed.Built specifically with Accessibility and Trust & Safety in mind, the tool provides both visual bounding boxes and an automated voice report to assist users with visual impairments.
 
 Tech Stack & Architecture
 
 Object Detection: YOLOv10-Nano (Optimized for high-speed, NMS-free inference).
 Depth Estimation: Depth Anything V2-Small (State-of-the-art relative depth).
 Frontend: Streamlit (Web-native mobile responsive UI)
 Voice Synthesis: gTTS (Google Text-to-Speech).Logic: Python/OpenCV/NumPy.
 
Calibration & Metric Accuracy

Since monocular depth models are "scale-ambiguous," I implemented a Metric Calibration system:Reference Point: The system was calibrated using a ground-truth object at 1.5 meters.Calibration Value: Based on a 153 raw depth units reading at that distance.Scaling Factor: Used to convert relative depth into real-world meters ($Distance = \frac{ScaleFactor}{RawValue}$).Note: Accuracy is dependent on the device's focal length. The current calibration is tuned for a standard mobile smartphone camera.


How to Run Locally

Clone the repo:
Bashgit clone https://github.com/YOUR_USERNAME/Spatial-AI-Assistant.git

Install dependencies:
Bashpip install -r requirements.txt

Run the app:
Bashstreamlit run app.py
