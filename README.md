# Emotion Detection Project 

This project is a Python-based application designed to detect and analyze human emotions in real-time or from static images.

## Technical Details & Architecture

This application processes a real-time webcam feed through a highly optimized pipeline to ensure smooth performance without requiring heavy hardware.

**Core Tech Stack:**
- **Computer Vision:** OpenCV (`cv2`) for video capture, frame resizing, color space conversion (BGR to RGB), and UI rendering.
- **Deep Learning Model:** DeepFace for facial recognition and emotion classification.
- **Data Processing:** Native Python libraries (`time` for FPS tracking, `statistics.mode` for data smoothing).

**Key Features & Optimizations:**
- **Performance - Frame Skipping:** To maintain a high and stable frame rate, the heavy DeepFace analysis runs only once every 10 frames (`ANALYSIS_INTERVAL = 10`).
- **Performance - Resolution Scaling:** Frames are dynamically downscaled to a 640px width before passing them to the neural network, drastically reducing computation time, then upscaled for bounding box mapping.
- **Emotion Stabilization:** Uses a rolling history buffer (`EMOTION_HISTORY_SIZE = 15`) and statistical mode to filter out noise and prevent the "flickering" of detected emotions on the main face.
- **SSD Face Detector:** Utilizes the Single Shot MultiBox Detector (`detector_backend='ssd'`) for fast, robust face detection, even accommodating multiple faces in a single frame.
- **Dynamic UI:** Features a live FPS counter, confidence percentage display, and custom color-coded bounding boxes mapped to specific emotions.

