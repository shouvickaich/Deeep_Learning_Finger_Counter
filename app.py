"""
╔══════════════════════════════════════════════════════════════╗
║        Deep Learning Finger Counter — Streamlit App          ║
║  Uses OpenCV Contour + Convexity Defects (no MediaPipe)      ║
║  Compatible with: shouvickaich/Deeep_Learning_Finger_Counter ║
║  streamlit-webrtc v0.64.5 · streamlit v1.42.0                ║
╚══════════════════════════════════════════════════════════════╝

Deploy:
    streamlit run app.py
"""

import math
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

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
        font-size: 2.2rem;
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
    section[data-testid="stSidebar"] { background: #080d18; border-right: 1px solid #1a3050; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Thread-safe shared state
# ─────────────────────────────────────────────
_lock = threading.Lock()
_shared = {
    "roi_top":      0.10,
    "roi_bottom":   0.90,
    "roi_left":     0.10,
    "roi_right":    0.90,
    "angle_thresh": 80,
    "depth_thresh": 15,
    "count":        0,
}


# ─────────────────────────────────────────────
# Core Finger Counting (Contour + Convexity Defects)
# ─────────────────────────────────────────────
def count_fingers_contour(frame: np.ndarray, settings: dict):
    count = 0
    debug = frame.copy()
    h, w  = frame.shape[:2]

    roi_t = int(h * settings["roi_top"])
    roi_b = int(h * settings["roi_bottom"])
    roi_l = int(w * settings["roi_left"])
    roi_r = int(w * settings["roi_right"])

    roi = frame[roi_t:roi_b, roi_l:roi_r]

    cv2.rectangle(debug, (roi_l, roi_t), (roi_r, roi_b), (0, 212, 255), 2)
    cv2.putText(debug, "ROI", (roi_l + 6, roi_t + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)

    blurred = cv2.GaussianBlur(roi, (7, 7), 0)

    # YCrCb skin mask
    ycrcb  = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    mask_y = cv2.inRange(ycrcb,
                         np.array([0,   133, 77],  dtype=np.uint8),
                         np.array([255, 173, 127], dtype=np.uint8))

    # HSV skin mask
    hsv    = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask_h = cv2.inRange(hsv,
                         np.array([0,  20,  70],  dtype=np.uint8),
                         np.array([20, 255, 255], dtype=np.uint8))

    skin = cv2.bitwise_or(mask_y, mask_h)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  kernel, iterations=1)
    skin = cv2.GaussianBlur(skin, (5, 5), 0)
    _, skin = cv2.threshold(skin, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand     = max(contours, key=cv2.contourArea)
        roi_area = (roi_b - roi_t) * (roi_r - roi_l)

        if cv2.contourArea(hand) > roi_area * 0.02:
            roi_draw = debug[roi_t:roi_b, roi_l:roi_r]
            cv2.drawContours(roi_draw, [hand], -1, (0, 255, 150), 2)

            hull = cv2.convexHull(hand, returnPoints=False)
            if hull is not None and len(hull) > 3:
                defects = cv2.convexityDefects(hand, hull)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand[s][0])
                        far   = tuple(hand[f][0])
                        end   = tuple(hand[e][0])
                        depth = d / 256.0

                        b = math.dist(far, start)
                        c = math.dist(far, end)
                        a = math.dist(start, end)

                        if b * c == 0:
                            continue

                        cos_a = (b**2 + c**2 - a**2) / (2 * b * c)
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))

                        if angle <= settings["angle_thresh"] and depth > settings["depth_thresh"]:
                            count += 1
                            cv2.circle(roi_draw, far, 5, (255, 80, 80), -1)

                    if count > 0:
                        count = min(count + 1, 5)

                    cv2.drawContours(roi_draw, [cv2.convexHull(hand)], -1, (123, 92, 246), 2)

    # HUD
    overlay = debug.copy()
    cv2.rectangle(overlay, (10, 10), (160, 90), (10, 25, 50), -1)
    cv2.addWeighted(overlay, 0.7, debug, 0.3, 0, debug)
    cv2.rectangle(debug, (10, 10), (160, 90), (0, 212, 255), 2)
    cv2.putText(debug, "FINGERS", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 170, 200), 1)
    cv2.putText(debug, str(count), (48, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 212, 255), 4, cv2.LINE_AA)

    return debug, count


# ─────────────────────────────────────────────
# WebRTC callback (v0.64+ API)
# ─────────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    with _lock:
        settings = dict(_shared)

    processed, count = count_fingers_contour(img, settings)

    with _lock:
        _shared["count"] = count

    return av.VideoFrame.from_ndarray(processed, format="bgr24")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    st.markdown("**📐 ROI (Region of Interest)**")
    roi_top    = st.slider("Top boundary",    0.0, 0.5, 0.10, 0.05)
    roi_bottom = st.slider("Bottom boundary", 0.5, 1.0, 0.90, 0.05)
    roi_left   = st.slider("Left boundary",   0.0, 0.5, 0.10, 0.05)
    roi_right  = st.slider("Right boundary",  0.5, 1.0, 0.90, 0.05)
    st.markdown("---")
    st.markdown("**🔬 Detection Parameters**")
    angle_thresh = st.slider("Angle threshold (°)", 40, 100, 80, 5,
                             help="Max inter-finger angle to count as defect")
    depth_thresh = st.slider("Depth threshold (px)", 5, 40, 15, 1,
                             help="Min defect depth to count as finger gap")

    # Push to shared dict
    with _lock:
        _shared.update({
            "roi_top":      roi_top,
            "roi_bottom":   roi_bottom,
            "roi_left":     roi_left,
            "roi_right":    roi_right,
            "angle_thresh": angle_thresh,
            "depth_thresh": depth_thresh,
        })

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Tips for best results:</b><br>
    • Use a <b>plain solid background</b><br>
    • Ensure <b>good lighting</b><br>
    • Show your <b>right hand, palm facing camera</b><br>
    • Keep fingers <b>spread apart</b>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
    <b>📦 Method:</b><br>
    YCrCb+HSV Skin Mask → Contour → Convexity Defects
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────
st.markdown("<h1>✋ Finger Counter AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#5a7a99;margin-top:0;font-size:0.85rem;'>"
    "Real-time finger detection · OpenCV Contour &amp; Convexity Defect Analysis</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

col_video, col_info = st.columns([3, 1])

with col_video:
    st.markdown("#### 📷 Live Camera Feed")
    webrtc_streamer(
        key="finger-counter",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        },
        async_processing=True,
    )

with col_info:
    st.markdown("#### 🔢 Live Count")

    with _lock:
        current_count = _shared["count"]

    EMOJI_MAP = {0: "✊", 1: "☝️", 2: "✌️", 3: "🤟", 4: "🖖", 5: "🖐️"}
    WORD_MAP  = {0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE"}

    st.markdown(f"""
    <div class='finger-display'>
        <div class='finger-num'>{current_count}</div>
        <div class='finger-label'>{WORD_MAP.get(current_count, '')}</div>
        <div style='font-size:3rem;margin-top:12px;'>{EMOJI_MAP.get(current_count, '')}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📖 Gesture Guide")
    for num in range(6):
        active = (num == current_count)
        color  = "#00d4ff" if active else "#2a4060"
        bg     = "rgba(0,212,255,0.1)" if active else "#0f1d2e"
        st.markdown(
            f"<div style='padding:6px 12px;margin:4px 0;border-radius:6px;"
            f"background:{bg};border:1px solid {color};font-size:0.85rem;'>"
            f"<span style='font-size:1.2rem'>{EMOJI_MAP[num]}</span> "
            f"<span style='color:{color};font-weight:600;'>{num} — {WORD_MAP[num]}</span></div>",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Explainer
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("⚙️ How does it work?"):
    st.markdown("""
    1. **Skin Segmentation** — YCrCb + HSV masks isolate skin pixels.
    2. **Morphological Cleanup** — Open/close removes noise.
    3. **Contour Detection** — `cv2.findContours` outlines the hand.
    4. **Convex Hull** — Tight polygon around the hand.
    5. **Convexity Defects** — Gaps between hull and contour = finger valleys.
    6. **Angle + Depth Filter** — Only valid gaps (angle < threshold, depth > threshold) are counted.
    7. **Count** — `fingers = valid_gaps + 1`, capped at 5.
    """)

st.markdown("""
<div style='text-align:center;padding:16px;color:#2a4060;font-size:0.75rem;
border-top:1px solid #1a3050;margin-top:12px;'>
    Based on <a href='https://github.com/shouvickaich/Deeep_Learning_Finger_Counter'
    style='color:#00d4ff;text-decoration:none;'>shouvickaich/Deeep_Learning_Finger_Counter</a>
    · OpenCV · streamlit-webrtc 0.64.5
</div>""", unsafe_allow_html=True)
