# Deep Learning Finger Counter

A real-time Computer Vision application built with **Streamlit** and **OpenCV** to count fingers on a user's right hand.

## 🚀 Features
- Real-time video processing via WebRTC.
- Contour and Convexity Defects analysis for finger detection.
- Streamlit-based web interface for easy deployment.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/shouvickaich/Deeep_Learning_Finger_Counter.git](https://github.com/shouvickaich/Deeep_Learning_Finger_Counter.git)
   cd Deeep_Learning_Finger_Counter
Install dependencies:

Bash
pip install -r requirements.txt
Run the App:

Bash
streamlit run app.py
📦 Requirements
Ensure you have a requirements.txt file with the following:

Plaintext
streamlit
streamlit-webrtc
opencv-python-headless
numpy
📝 How to Use
Allow the browser to access your camera.

Position your right hand clearly in front of the camera.

For best accuracy, use a plain, solid-colored background with good lighting.

👨‍💻 Credits
Original project logic by Shouvick Aich.


---

### 3. Important Step: `packages.txt`
To run OpenCV on Streamlit Cloud, you also need to create a file named `packages.txt` in your root folder and add this single line:
```text
libgl1# Deeep_Learning_Finger_Counter
