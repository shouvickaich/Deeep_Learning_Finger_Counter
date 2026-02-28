"""
╔══════════════════════════════════════════════════════════════╗
║        Deep Learning Finger Counter — Streamlit App          ║
║  Uses OpenCV Contour + Convexity Defects (no MediaPipe)      ║
║  Compatible with: shouvickaich/Deeep_Learning_Finger_Counter ║
╚══════════════════════════════════════════════════════════════╝

Deploy:
    streamlit run app.py

Requirements:
    pip install streamlit streamlit-webrtc opencv-python-headless numpy av
"""

import streamlit as st
import cv2
import numpy as np
import av
import math
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Finger Counter AI",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=DM+Sans:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0a0f1e;
        color: #e0f0ff;
    }
    .main { background-color: #0a0f1e; }
    .block-container { padding-top: 1.5rem; }

    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #00d4ff, #7b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        margin-bottom: 0;
    }

    .finger-display {
        background: linear-gradient(135deg, #0f1d2e, #141e30);
        border: 2px solid #00d4ff44;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,212,255,0.1);
    }

    .finger-num {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem;
        color: #00d4ff;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
        line-height: 1;
    }

    .finger-label {
        font-size: 1rem;
        color: #5a7a99;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-top: 8px;
    }

    .info-box {
        background: #0f1d2e;
        border: 1px solid #1a3050;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 12px;
        font-size: 0.85rem;
        color: #8aadcc;
        line-height: 1.7;
    }

    .stButton>button {
        background: linear-gradient(135deg, #00d4ff22, #7b5cf622);
        border: 1px solid #00d4ff55;
        color: #00d4ff;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        padding: 8px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00d4ff44, #7b5cf644);
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0,212,255,0.3);
    }

    .stSlider > div > div > div { background: #00d4ff !important; }
    section[data-testid="stSidebar"] { background: #080d18; border-right: 1px solid #1a3050; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Shared state for finger count
# ─────────────────────────────────────────────
if "finger_count" not in st.session_state:
    st.session_state.finger_count = 0


# ─────────────────────────────────────────────
# Core Finger Counting Logic (Contour + Convexity Defects)
# ─────────────────────────────────────────────
def count_fingers_contour(frame: np.ndarray, settings: dict) -> tuple[np.ndarray, int]:
    """
    Detects fingers using:
      1. Skin segmentation (YCrCb + HSV)
      2. Largest contour detection
      3. Convexity defects analysis
    Returns annotated frame and finger count.
    """
    count = 0
    debug_frame = frame.copy()
    h, w = frame.shape[:2]

    # — ROI (Region of Interest): centre of frame —
    roi_top    = int(h * settings["roi_top"])
    roi_bottom = int(h * settings["roi_bottom"])
    roi_left   = int(w * settings["roi_left"])
    roi_right  = int(w * settings["roi_right"])

    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # — Draw ROI rectangle —
    cv2.rectangle(debug_frame, (roi_left, roi_top), (roi_right, roi_bottom),
                  (0, 212, 255), 2)
    cv2.putText(debug_frame, "ROI", (roi_left + 5, roi_top + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)

    # — Skin segmentation —
    blurred = cv2.GaussianBlur(roi, (7, 7), 0)

    # YCrCb skin mask
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # HSV skin mask
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Combine masks
    skin_mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    _, skin_mask = cv2.threshold(skin_mask, 128, 255, cv2.THRESH_BINARY)

    # — Find contours —
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Largest contour = hand
        hand_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(hand_contour)
        roi_area = (roi_bottom - roi_top) * (roi_right - roi_left)

        if area > roi_area * 0.02:   # ignore tiny blobs
            # Draw hand contour on ROI
            roi_draw = debug_frame[roi_top:roi_bottom, roi_left:roi_right]
            cv2.drawContours(roi_draw, [hand_contour], -1, (0, 255, 150), 2)

            # Convex hull
            hull = cv2.convexHull(hand_contour, returnPoints=False)

            # Convexity defects
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(hand_contour, hull)

                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start  = tuple(hand_contour[s][0])
                        end    = tuple(hand_contour[e][0])
                        far    = tuple(hand_contour[f][0])
                        depth  = d / 256.0

                        # Angle between finger vectors using cosine rule
                        a = math.dist(start, end)
                        b = math.dist(far, start)
                        c = math.dist(far, end)

                        if b * c == 0:
                            continue

                        angle = math.degrees(
                            math.acos(
                                max(-1.0, min(1.0,
                                    (b**2 + c**2 - a**2) / (2 * b * c)
                                ))
                            )
                        )

                        # Valid finger gap: angle < threshold & sufficient depth
                        if angle <= settings["angle_thresh"] and depth > settings["depth_thresh"]:
                            count += 1
                            cv2.circle(roi_draw, far, 5, (255, 80, 80), -1)

                    # Finger count = gaps + 1 (if any gaps found)
                    if count > 0:
                        count = min(count + 1, 5)

                    # Draw hull on frame
                    hull_pts = cv2.convexHull(hand_contour)
                    cv2.drawContours(roi_draw, [hull_pts], -1, (123, 92, 246), 2)

    # — HUD Overlay —
    emoji_map = {0: "✊", 1: "☝️", 2: "✌️", 3: "🤟", 4: "🖖", 5: "🖐️"}
    emoji = emoji_map.get(count, "")

    # Background panel for count
    overlay = debug_frame.copy()
    cv2.rectangle(overlay, (10, 10), (160, 90), (10, 25, 50), -1)
    cv2.addWeighted(overlay, 0.7, debug_frame, 0.3, 0, debug_frame)
    cv2.rectangle(debug_frame, (10, 10), (160, 90), (0, 212, 255), 2)

    cv2.putText(debug_frame, "FINGERS", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 170, 200), 1)
    cv2.putText(debug_frame, str(count), (48, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 212, 255), 4, cv2.LINE_AA)

    return debug_frame, count


# ─────────────────────────────────────────────
# WebRTC Video Processor
# ─────────────────────────────────────────────
class FingerCounterProcessor:
    def __init__(self):
        self.finger_count = 0
        self.settings = {
            "roi_top":      0.1,
            "roi_bottom":   0.9,
            "roi_left":     0.1,
            "roi_right":    0.9,
            "angle_thresh": 80,
            "depth_thresh": 15,
        }

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        processed, count = count_fingers_contour(img, self.settings)
        self.finger_count = count
        return av.VideoFrame.from_ndarray(processed, format="bgr24")


# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────
# — Sidebar —
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    st.markdown("**📐 ROI (Region of Interest)**")
    roi_top    = st.slider("Top boundary",    0.0, 0.5, 0.10, 0.05, key="roi_t")
    roi_bottom = st.slider("Bottom boundary", 0.5, 1.0, 0.90, 0.05, key="roi_b")
    roi_left   = st.slider("Left boundary",   0.0, 0.5, 0.10, 0.05, key="roi_l")
    roi_right  = st.slider("Right boundary",  0.5, 1.0, 0.90, 0.05, key="roi_r")

    st.markdown("---")
    st.markdown("**🔬 Detection Parameters**")
    angle_thresh = st.slider("Angle threshold (°)", 40, 100, 80, 5,
                             help="Max angle between fingers for defect to be counted")
    depth_thresh = st.slider("Depth threshold (px)", 5, 40, 15, 1,
                             help="Min convexity defect depth to count as a finger gap")

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Tips for best results:</b><br>
    • Use a <b>plain, solid background</b><br>
    • Ensure <b>good lighting</b><br>
    • Position your <b>right hand</b> clearly in the ROI box<br>
    • Hold hand <b>palm facing camera</b><br>
    • Fingers should be <b>spread apart</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
    <b>📦 Method:</b> OpenCV Skin Segmentation → Contour → Convexity Defects
    </div>
    """, unsafe_allow_html=True)

# — Header —
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown("<h1>✋ Finger Counter AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#5a7a99; margin-top:0; font-size:0.85rem;'>Real-time finger detection using OpenCV Contour & Convexity Defect Analysis</p>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square)", unsafe_allow_html=True)

st.markdown("---")

# — Main layout —
col_video, col_info = st.columns([3, 1])

with col_video:
    st.markdown("#### 📷 Live Camera Feed")

    # WebRTC config (STUN servers for remote deployment)
    RTC_CONFIG = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })

    ctx = webrtc_streamer(
        key="finger-counter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=FingerCounterProcessor,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False,
        },
        async_processing=True,
    )

    # Update settings dynamically
    if ctx.video_processor:
        ctx.video_processor.settings = {
            "roi_top":      roi_top,
            "roi_bottom":   roi_bottom,
            "roi_left":     roi_left,
            "roi_right":    roi_right,
            "angle_thresh": angle_thresh,
            "depth_thresh": depth_thresh,
        }

with col_info:
    st.markdown("#### 🔢 Live Count")

    # Finger count display
    finger_display = st.empty()
    emoji_hand = st.empty()
    history_display = st.empty()

    # Display count (reads from processor if active)
    current_count = 0
    if ctx.video_processor:
        current_count = ctx.video_processor.finger_count

    EMOJI_MAP = {0: "✊", 1: "☝️", 2: "✌️", 3: "🤟", 4: "🖖", 5: "🖐️"}
    WORD_MAP  = {0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE"}

    finger_display.markdown(f"""
    <div class='finger-display'>
        <div class='finger-num'>{current_count}</div>
        <div class='finger-label'>{WORD_MAP.get(current_count, "")}</div>
        <div style='font-size:3rem; margin-top:12px;'>{EMOJI_MAP.get(current_count, "")}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Guide
    st.markdown("#### 📖 Gesture Guide")
    for num, (emoji, word) in enumerate(zip(EMOJI_MAP.values(), WORD_MAP.values())):
        color = "#00d4ff" if num == current_count else "#2a4060"
        st.markdown(
            f"<div style='padding:6px 12px; margin:4px 0; border-radius:6px; "
            f"background:{'rgba(0,212,255,0.1)' if num == current_count else '#0f1d2e'}; "
            f"border:1px solid {color}; font-size:0.85rem;'>"
            f"<span style='font-size:1.2rem'>{emoji}</span> "
            f"<span style='color:{color}; font-weight:600;'>{num} — {word}</span></div>",
            unsafe_allow_html=True
        )

# — How it works —
st.markdown("---")
with st.expander("⚙️ How does it work?", expanded=False):
    st.markdown("""
    This app uses **classical computer vision** (no neural network required at runtime):

    1. **Skin Segmentation** — The frame is converted to YCrCb and HSV color spaces. Pixels matching skin-tone ranges are extracted to form a binary skin mask.
    2. **Morphological Cleanup** — Noise is removed using erosion/dilation (open/close operations) to produce a clean hand silhouette.
    3. **Contour Detection** — `cv2.findContours` finds the outline of the largest skin region (your hand).
    4. **Convex Hull** — A tight-fitting convex polygon is computed around the hand contour.
    5. **Convexity Defects** — Gaps between the hull and the contour are identified. Each finger gap = one defect.
    6. **Angle + Depth Filter** — A defect is only counted as a "finger gap" if:
       - The angle between adjacent finger vectors is **less than the angle threshold**
       - The defect depth exceeds the **depth threshold**
    7. **Final Count** — `fingers = valid_defects + 1` (capped at 5)

    This matches the **Contour + Convexity Defects** approach used in the original repository by Shouvick Aich.
    """)

# — Footer —
st.markdown("""
<div style='text-align:center; padding:20px; color:#2a4060; font-size:0.75rem; margin-top:20px;
border-top: 1px solid #1a3050;'>
    Deep Learning Finger Counter · Streamlit Deployment · Based on
    <a href='https://github.com/shouvickaich/Deeep_Learning_Finger_Counter'
    style='color:#00d4ff; text-decoration:none;'>shouvickaich/Deeep_Learning_Finger_Counter</a>
    <br>Built with OpenCV · streamlit-webrtc · Python
</div>
""", unsafe_allow_html=True)
