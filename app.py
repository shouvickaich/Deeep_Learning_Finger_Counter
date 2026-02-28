import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import os

# Set page title
st.set_page_config(page_title="Right Hand Finger Counter", layout="centered")
st.title("🖐️ Right Hand Finger Counter")
st.markdown("This app uses Computer Vision to count fingers on your right hand in real-time.")

class FingerCounterTransformer(VideoTransformerBase):
    def __init__(self):
        # You would typically load your trained model here
        # For this example, we use a simplified contour-based approach
        pass

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror effect
        
        # --- Image Processing Logic ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (35, 35), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        try:
            cnt = max(contours, key=lambda x: cv2.contourArea(x))
            hull = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull)
            
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                
                # Calculate sides of triangle to find angle
                a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
                
                if angle <= 90:
                    count_defects += 1
                    cv2.circle(img, far, [0, 0, 255], -1)
            
            total_fingers = count_defects + 1
            cv2.putText(img, f"Count: {total_fingers}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            
        except Exception:
            pass

        return img

# Start Video Stream
webrtc_streamer(key="finger-counter", video_transformer_factory=FingerCounterTransformer)

st.sidebar.info("Note: Place your hand against a plain background for best results.")