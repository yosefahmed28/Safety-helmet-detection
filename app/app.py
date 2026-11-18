import os
import tempfile
from collections import Counter

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Safety Helmet Detection",
    page_icon="🦺",
    layout="wide",
)

# =========================
# PATHS
# =========================
# This file lives in <project>/app/app.py. Put the trained weights in <project>/model/best.pt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "best.pt")

IMGSZ = 416


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    return YOLO(MODEL_PATH)


model = load_model()

# =========================
# TITLE
# =========================
st.title("🦺 Safety Helmet Detection using YOLO")
st.write(
    "Detects **people**, and whether each one is wearing a **helmet** or has **no helmet**."
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold", min_value=0.1, max_value=1.0, value=0.4, step=0.05
)

mode = st.sidebar.radio(
    "Choose Detection Mode", ["Image Upload", "Video Upload", "Live Camera"]
)


# =========================
# HELPERS
# =========================
def count_classes(result):
    """Return a Counter like {'helmet': 2, 'no_helmet': 1, 'person': 3}."""
    counts = Counter()
    for box in result.boxes:
        counts[model.names[int(box.cls[0])]] += 1
    return counts


def show_summary(container, counts):
    with container.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Helmet", counts.get("helmet", 0))
        c2.metric("🔴 No helmet", counts.get("no_helmet", 0))
        c3.metric("🧍 Person", counts.get("person", 0))
        if counts.get("no_helmet", 0) > 0:
            st.error(f"⚠️ Safety violation: {counts['no_helmet']} head(s) without a helmet")
        elif counts.get("helmet", 0) > 0:
            st.success("✅ All detected heads are wearing helmets")


def detect(frame_or_image):
    result = model.predict(
        source=frame_or_image, conf=conf_threshold, imgsz=IMGSZ, verbose=False
    )[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return annotated, count_classes(result)


# =========================
# IMAGE UPLOAD MODE
# =========================
if mode == "Image Upload":
    st.header("📷 Image Upload Detection")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        annotated, counts = detect(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detection Result")
            st.image(annotated, use_container_width=True)

        show_summary(st.container(), counts)

# =========================
# VIDEO UPLOAD MODE
# =========================
elif mode == "Video Upload":
    st.header("🎞️ Video Upload Detection")

    frame_skip = st.sidebar.slider("Process every Nth frame", 1, 10, 2)
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        suffix = os.path.splitext(uploaded_video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name

        if st.button("Start detection"):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            progress = st.progress(0.0)
            frame_placeholder = st.empty()
            stats_placeholder = st.empty()

            idx = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % frame_skip == 0:
                    annotated, counts = detect(frame)
                    frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
                    show_summary(stats_placeholder, counts)
                idx += 1
                progress.progress(min(idx / total, 1.0))

            cap.release()
            progress.empty()
            st.success("Video processing finished.")

        # note: the temp file is removed by the OS temp cleanup

# =========================
# LIVE CAMERA MODE
# =========================
elif mode == "Live Camera":
    st.header("🎥 Live Camera Detection")
    st.warning("Make sure your camera is connected and not used by another app.")

    run = st.checkbox("Run camera")
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera.")
        else:
            while run:
                ok, frame = cap.read()
                if not ok:
                    st.error("Failed to read frame from camera.")
                    break
                annotated, counts = detect(frame)
                frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
                show_summary(stats_placeholder, counts)
            cap.release()