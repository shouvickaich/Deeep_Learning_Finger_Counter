"""
Deep Learning Finger Counter — Streamlit App
Based on: github.com/shouvickaich/Deeep_Learning_Finger_Counter
Method: Skin Segmentation → Contour → Convexity Defects
"""

import math
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Finger Counter AI",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
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
section[data-testid="stSidebar"] {
    background: #080d18;
    border-right: 1px solid #1a3050;
}
</style>
""", unsafe_allow_html=True)


# ── Thread-safe shared state ──────────────────────────────────
_lock = threading.Lock()
_shared: dict = {
    "roi_top":      0.10,
    "roi_bottom":   0.90,
    "roi_left":     0.10,
    "roi_right":    0.90,
    "angle_thresh": 80,
    "depth_thresh": 15,
    "count":        0,
}


# ── Core algorithm ────────────────────────────────────────────
def count_fingers(frame: np.ndarray, cfg: dict):
    """
    Returns (annotated_frame, finger_count)
    using YCrCb+HSV skin mask → contour → convexity defects.
    """
    out = frame.copy()
    h, w = frame.shape[:2]

    t = int(h * cfg["roi_top"])
    b = int(h * cfg["roi_bottom"])
    l = int(w * cfg["roi_left"])
    r = int(w * cfg["roi_right"])

    # Draw ROI
    cv2.rectangle(out, (l, t), (r, b), (0, 212, 255), 2)
    cv2.putText(out, "ROI", (l + 6, t + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)

    roi = frame[t:b, l:r]
    blur = cv2.GaussianBlur(roi, (7, 7), 0)

    # YCrCb mask
    ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(ycc,
                     np.array([0,   133, 77],  np.uint8),
                     np.array([255, 173, 127], np.uint8))

    # HSV mask
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m2 = cv2.inRange(hsv,
                     np.array([0,  20,  70],  np.uint8),
                     np.array([20, 255, 255], np.uint8))

    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    if cnts:
        hand = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(hand) > (b - t) * (r - l) * 0.02:
            sub = out[t:b, l:r]
            cv2.drawContours(sub, [hand], -1, (0, 255, 150), 2)

            hull_idx = cv2.convexHull(hand, returnPoints=False)
            if hull_idx is not None and len(hull_idx) > 3:
                defects = cv2.convexityDefects(hand, hull_idx)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(hand[s][0])
                        end   = tuple(hand[e][0])
                        far   = tuple(hand[f][0])
                        depth = d / 256.0

                        bv = math.dist(far, start)
                        cv = math.dist(far, end)
                        av = math.dist(start, end)

                        if bv * cv == 0:
                            continue

                        cos_a = (bv**2 + cv**2 - av**2) / (2 * bv * cv)
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))

                        if angle <= cfg["angle_thresh"] and depth > cfg["depth_thresh"]:
                            count += 1
                            cv2.circle(sub, far, 6, (255, 80, 80), -1)

                    if count > 0:
                        count = min(count + 1, 5)

                    cv2.drawContours(sub, [cv2.convexHull(hand)], -1, (123, 92, 246), 2)

    # HUD
    ov = out.copy()
    cv2.rectangle(ov, (10, 10), (160, 90), (10, 25, 50), -1)
    cv2.addWeighted(ov, 0.7, out, 0.3, 0, out)
    cv2.rectangle(out, (10, 10), (160, 90), (0, 212, 255), 2)
    cv2.putText(out, "FINGERS", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 170, 200), 1)
    cv2.putText(out, str(count), (48, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 212, 255), 4, cv2.LINE_AA)

    return out, count


# ── WebRTC callback ───────────────────────────────────────────
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    with _lock:
        cfg = dict(_shared)
    processed, cnt = count_fingers(img, cfg)
    with _lock:
        _shared["count"] = cnt
    return av.VideoFrame.from_ndarray(processed, format="bgr24")


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    st.markdown("**📐 ROI Boundaries**")
    roi_top    = st.slider("Top",    0.0, 0.5, 0.10, 0.05)
    roi_bottom = st.slider("Bottom", 0.5, 1.0, 0.90, 0.05)
    roi_left   = st.slider("Left",   0.0, 0.5, 0.10, 0.05)
    roi_right  = st.slider("Right",  0.5, 1.0, 0.90, 0.05)
    st.markdown("---")
    st.markdown("**🔬 Detection**")
    angle_thresh = st.slider("Angle threshold (°)", 40, 100, 80, 5)
    depth_thresh = st.slider("Depth threshold (px)", 5, 40, 15, 1)

    with _lock:
        _shared.update({
            "roi_top": roi_top, "roi_bottom": roi_bottom,
            "roi_left": roi_left, "roi_right": roi_right,
            "angle_thresh": angle_thresh, "depth_thresh": depth_thresh,
        })

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>💡 Tips:</b><br>
    • Plain solid background<br>
    • Good lighting<br>
    • Right hand, palm facing camera<br>
    • Fingers spread apart
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
st.markdown("<h1>✋ Finger Counter AI</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#5a7a99;margin-top:0;font-size:0.85rem;'>"
    "Real-time · OpenCV Skin Mask → Contour → Convexity Defects</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Layout ────────────────────────────────────────────────────
col_cam, col_hud = st.columns([3, 1])

with col_cam:
    st.markdown("#### 📷 Live Camera")
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

with col_hud:
    st.markdown("#### 🔢 Count")
    with _lock:
        cur = _shared["count"]

    EMOJIS = {0: "✊", 1: "☝️", 2: "✌️", 3: "🤟", 4: "🖖", 5: "🖐️"}
    WORDS  = {0: "ZERO", 1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE"}

    st.markdown(f"""
    <div class='finger-display'>
        <div class='finger-num'>{cur}</div>
        <div class='finger-label'>{WORDS.get(cur,'')}</div>
        <div style='font-size:3rem;margin-top:10px'>{EMOJIS.get(cur,'')}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📖 Guide")
    for n in range(6):
        active = n == cur
        c = "#00d4ff" if active else "#2a4060"
        bg = "rgba(0,212,255,0.1)" if active else "#0f1d2e"
        st.markdown(
            f"<div style='padding:6px 12px;margin:4px 0;border-radius:6px;"
            f"background:{bg};border:1px solid {c};font-size:0.85rem;'>"
            f"{EMOJIS[n]} <span style='color:{c};font-weight:600'>{n} — {WORDS[n]}</span></div>",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
with st.expander("⚙️ How it works"):
    st.markdown("""
    1. **Skin Segmentation** — YCrCb + HSV masks isolate hand pixels  
    2. **Morphology** — Open/Close ops clean up noise  
    3. **Contour** — Largest region = hand outline  
    4. **Convex Hull** — Tight polygon around the hand  
    5. **Convexity Defects** — Finger valleys between hull & contour  
    6. **Filter** — Angle < threshold AND depth > threshold = valid gap  
    7. **Count** — `fingers = gaps + 1`, capped at 5  
    """)

st.markdown("""
<div style='text-align:center;padding:12px;color:#2a4060;font-size:0.75rem;'>
Based on <a href='https://github.com/shouvickaich/Deeep_Learning_Finger_Counter'
style='color:#00d4ff;text-decoration:none'>shouvickaich/Deeep_Learning_Finger_Counter</a>
· streamlit-webrtc 0.64.5 · OpenCV
</div>""", unsafe_allow_html=True)
