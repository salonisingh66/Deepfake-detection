"""DeepShield AI: Deepfake Detection and Forensic Analysis System.

Professional digital forensics and deepfake verification Streamlit application.
Configured for efficient CPU-only execution on 12GB RAM hardware.
"""

from datetime import datetime, timezone
import io
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Application Imports
from config.model_config import AVAILABLE_MODELS, CONFIDENCE_THRESHOLD, DEFAULT_MODEL_ID
from config.settings import settings
from src.database.db import init_db
from src.database.history import (
    clear_all_history,
    delete_record_by_id,
    get_analysis_records,
    get_record_by_id,
    save_analysis_record,
)
from src.detection.face_detector import FaceDetector
from src.detection.image_analyzer import ImageAnalyzer
from src.detection.video_analyzer import VideoAnalyzer
from src.evaluation.evaluate import DatasetEvaluator
from src.forensics.hashing import compute_file_fingerprints
from src.forensics.report_generator import MANDATORY_DISCLAIMER, generate_forensic_pdf
from src.models.model_loader import ModelLoader
from src.utils.file_utils import cleanup_directory, safe_temp_file, sanitize_filename
from src.utils.logging_config import logger

# Initialize Database Schema
init_db()

# Page Configuration
st.set_page_config(
    page_title="DeepShield AI - Deepfake Forensic Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Forensics Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main Background & Cards */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Sleek Forensics Header */
    .forensic-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .forensic-title {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .forensic-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-bottom: 0px;
    }
    
    /* Status Badges */
    .badge-ready {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
        font-size: 12px;
        font-weight: 600;
    }
    .badge-error {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        font-size: 12px;
        font-weight: 600;
    }
    
    /* Result Cards */
    .verdict-box-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    .verdict-box-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(4, 120, 87, 0.08) 100%);
        border: 1.5px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    .verdict-box-inconclusive {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(180, 83, 9, 0.08) 100%);
        border: 1.5px solid #f59e0b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 16px;
    }
    
    .verdict-label {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 1px;
    }
    .verdict-score {
        font-size: 15px;
        color: #cbd5e1;
        margin-top: 4px;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .metric-card-title {
        font-size: 12px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card-value {
        font-size: 22px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 4px;
    }
    
    /* Legal Disclaimer Footer Box */
    .disclaimer-card {
        background-color: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 11px;
        color: #94a3b8;
        line-height: 1.5;
        margin-top: 24px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Cached Analyzers & Model Loader
@st.cache_resource(show_spinner=False)
def get_model_loader() -> ModelLoader:
    return ModelLoader.get_instance()

@st.cache_resource(show_spinner=False)
def get_analyzers():
    loader = get_model_loader()
    face_detector = FaceDetector()
    image_analyzer = ImageAnalyzer(face_detector=face_detector)
    video_analyzer = VideoAnalyzer(face_detector=face_detector)
    return loader, face_detector, image_analyzer, video_analyzer

loader, face_detector, image_analyzer, video_analyzer = get_analyzers()

# Session State Initialization
if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = DEFAULT_MODEL_ID
if "detection_threshold" not in st.session_state:
    st.session_state.detection_threshold = CONFIDENCE_THRESHOLD


# Sidebar Navigation & System Telemetry
with st.sidebar:
    st.markdown("### 🛡️ **DeepShield AI**")
    st.markdown("<p style='font-size: 12px; color: #94a3b8; margin-top: -10px;'>Forensic Deepfake Detection</p>", unsafe_allow_html=True)
    
    # Model Readiness Status
    is_ready, load_err = loader.is_model_ready(st.session_state.selected_model_id)
    if is_ready:
        st.markdown(f"<span class='badge-ready'>● Model Ready (CPU)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-error'>● Model Offline: {load_err}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Image Analysis",
            "Video Analysis",
            "Batch Analysis",
            "Analysis History",
            "Model Evaluation",
            "Model Information",
            "Settings & System"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### **Active Forensic Model**")
    chosen_model = st.selectbox(
        "Model Architecture",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_id),
        format_func=lambda mid: AVAILABLE_MODELS[mid].name,
        key="sidebar_model_select"
    )
    if chosen_model != st.session_state.selected_model_id:
        st.session_state.selected_model_id = chosen_model
        st.rerun()

    st.markdown("---")
    st.markdown(
        f"<div style='font-size: 11px; color: #64748b;'>"
        f"<b>Environment:</b> CPU-only (12GB RAM target)<br>"
        f"<b>Version:</b> {settings.VERSION}<br>"
        f"<b>Device:</b> {settings.DEVICE.upper()}"
        f"</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "Dashboard":
    st.markdown(
        f"""
        <div class="forensic-header">
            <div class="forensic-title">DeepShield AI Forensic Lab</div>
            <div class="forensic-subtitle">
                Automated facial forgery detection, cryptographic provenance hashing, and digital forensic reporting.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Status & Telemetry Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-card-title">System Health</div>
                <div class="metric-card-value" style="color: #10b981;">OPERATIONAL</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c2:
        model_meta = AVAILABLE_MODELS.get(st.session_state.selected_model_id, AVAILABLE_MODELS[DEFAULT_MODEL_ID])
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-card-title">Active Model</div>
                <div class="metric-card-value" style="font-size: 16px;">{model_meta.architecture.split()[0]}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c3:
        records = get_analysis_records(limit=1000)
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-card-title">Audit Log Entries</div>
                <div class="metric-card-value">{len(records)}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-card-title">Compute Target</div>
                <div class="metric-card-value" style="color: #38bdf8;">CPU INFERENCE</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("### **Forensic Inspection Pipeline**")
    st.markdown(
        """
        DeepShield AI executes a verifiable forensic protocol designed to detect facial deepfakes, FaceSwap, and synthetic generation:
        """
    )
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            **1. Cryptographic Provenance**
            - SHA-256 and MD5 digital hashing
            - EXIF timestamp & software tag inspection
            - Container and codec verification
            """
        )
    with col_b:
        st.markdown(
            """
            **2. Facial Geometry & Cropping**
            - Multi-face localization with bounding boxes
            - Boundary margin expansion for blending seams
            - Deterministic normalization & scaling
            """
        )
    with col_c:
        st.markdown(
            """
            **3. Deep Forensic Classification**
            - Vision Transformer (ViT) patch attention
            - Sequential frame temporal aggregation
            - Court-admissible forensic PDF reports
            """
        )

    st.markdown("---")
    st.markdown("### **Quick Analysis Launchpad**")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info("📷 **Analyze Single Image**")
        st.write("Upload JPG, PNG, or WEBP photos for face detection and visual attention explanations.")
    with q2:
        st.info("🎬 **Analyze Video Stream**")
        st.write("Sample MP4, MOV, or AVI video frames with interactive timeline telemetry.")
    with q3:
        st.info("📁 **Batch Image Analysis**")
        st.write("Process up to 20 images simultaneously with structured CSV audit export.")

    # Forensic Disclaimer
    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: IMAGE ANALYSIS
# ==============================================================================
elif page == "Image Analysis":
    st.markdown("## 🔍 **Forensic Image Deepfake Analysis**")
    st.markdown("Upload an image to identify digital manipulation, inspect facial boundaries, and view model sensitivity overlays.")

    # Sample Selector or Upload
    c_source, c_opts = st.columns([2, 1])
    with c_source:
        sample_choice = st.selectbox(
            "Select Input Source",
            ["Upload Image File", "Use Sample: Real Portrait Photo (NASA Astronaut)", "Use Sample: Image Without Face (Rocket)"]
        )

    uploaded_file = None
    image_bytes = None
    input_filename = "upload.jpg"

    if sample_choice == "Upload Image File":
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Maximum size: 20MB"
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            input_filename = sanitize_filename(uploaded_file.name)
    elif "Real Portrait" in sample_choice:
        sample_path = settings.SAMPLES_DIR / "sample_face_real.jpg"
        if sample_path.exists():
            image_bytes = sample_path.read_bytes()
            input_filename = sample_path.name
    else:
        sample_path = settings.SAMPLES_DIR / "sample_no_face.jpg"
        if sample_path.exists():
            image_bytes = sample_path.read_bytes()
            input_filename = sample_path.name

    if image_bytes:
        # Layout: Preview & Parameters
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image_bytes, caption=f"Input: {input_filename}", use_column_width=True)
            run_btn = st.button("🚀 Run Forensic Analysis", type="primary", use_container_width=True)

        with col_info:
            fingerprints = compute_file_fingerprints(image_bytes)
            st.markdown("#### **Cryptographic Provenance**")
            st.markdown(f"**Filename:** `{input_filename}`")
            st.markdown(f"**Size:** `{len(image_bytes):,} bytes`")
            st.markdown(f"**SHA-256:** <br><code style='font-size:11px;'>{fingerprints['sha256']}</code>", unsafe_allow_html=True)
            st.markdown(f"**MD5:** <br><code style='font-size:11px;'>{fingerprints['md5']}</code>", unsafe_allow_html=True)

        if run_btn:
            with st.spinner("Executing forensic pipeline (Face detection, ViT inference, Attention map)..."):
                res = image_analyzer.analyze_image(
                    image_bytes=image_bytes,
                    filename=input_filename,
                    model_id=st.session_state.selected_model_id,
                    threshold=st.session_state.detection_threshold,
                    generate_explanation=True
                )

            st.markdown("---")

            # Check Status
            if not res.is_valid:
                st.error(f"❌ **Validation Failed:** {res.error_message}")
            elif res.status == "NO_FACES_DETECTED":
                st.warning(f"⚠️ **Forensic Notice:** {res.error_message}")
                save_analysis_record(
                    filename=input_filename,
                    file_type="image",
                    sha256=res.sha256,
                    model_name=res.model_name,
                    prediction="No Face",
                    confidence_score=0.0,
                    fake_score=0.0,
                    faces_detected=0,
                    execution_time_ms=res.execution_time_ms,
                    status="NO_FACES_DETECTED",
                    metadata_summary=res.metadata
                )
            elif res.status == "SUCCESS":
                # Save Record to Database
                save_analysis_record(
                    filename=input_filename,
                    file_type="image",
                    sha256=res.sha256,
                    model_name=res.model_name,
                    prediction=res.prediction,
                    confidence_score=res.confidence,
                    fake_score=res.fake_score,
                    faces_detected=res.faces_detected_count,
                    execution_time_ms=res.execution_time_ms,
                    status="SUCCESS",
                    metadata_summary=res.metadata
                )

                # Verdict Display
                v_box_class = "verdict-box-fake" if res.prediction == "Fake" else "verdict-box-real"
                v_color = "#ef4444" if res.prediction == "Fake" else "#10b981"
                st.markdown(
                    f"""
                    <div class="{v_box_class}">
                        <div class="verdict-label" style="color: {v_color};">{res.prediction.upper()}</div>
                        <div class="verdict-score">
                            Model Confidence: <b>{res.confidence*100:.1f}%</b> | 
                            Manipulated Probability: <b>{res.fake_score*100:.1f}%</b> | 
                            Inference Time: <b>{res.execution_time_ms:.1f}ms</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Visual Results Tabs
                tab1, tab2, tab3 = st.tabs(["Detected Faces", "Visual Attention Explanation", "Technical Metadata"])
                
                with tab1:
                    c_det1, c_det2 = st.columns([2, 1])
                    with c_det1:
                        if res.annotated_image_rgb is not None:
                            st.image(res.annotated_image_rgb, caption="Localized Facial Bounding Boxes", use_column_width=True)
                    with c_det2:
                        st.markdown(f"**Faces Localized:** {res.faces_detected_count}")
                        for p in res.face_predictions:
                            st.markdown(f"**Face #{p.face_index + 1}:** {p.prediction} ({p.fake_score*100:.1f}% fake)")

                with tab2:
                    if res.explanation_overlay_rgb is not None:
                        c_ex1, c_ex2 = st.columns(2)
                        with c_ex1:
                            if res.face_predictions:
                                primary_crop = res.face_predictions[0]
                                st.image(res.explanation_overlay_rgb, caption="Attention Sensitivity Overlay", use_column_width=True)
                        with c_ex2:
                            st.markdown("#### **Attention Map Interpretation**")
                            st.markdown(
                                """
                                The overlay depicts self-attention weights from the final Vision Transformer layer.
                                - **Warm colors (Red, Orange, Yellow):** Facial regions that most strongly influenced the classification decision.
                                - **Cool colors (Blue, Green):** Background or low-influence features.
                                
                                *Note: Sensitivity maps represent computational feature correlation, not confirmed forensic evidence.*
                                """
                            )
                    else:
                        st.info("Visual explanation overlay is supported for Vision Transformer backbones.")

                with tab3:
                    st.markdown("#### **Extracted Metadata Forensics**")
                    meta_df = pd.DataFrame(
                        [{"Field": k, "Value": str(v)} for k, v in res.metadata.items() if k != "exif_tags" and k != "forensic_observations"]
                    )
                    st.dataframe(meta_df, use_container_width=True, hide_index=True)
                    
                    if res.metadata.get("forensic_observations"):
                        st.markdown("**Forensic Observations:**")
                        for obs in res.metadata["forensic_observations"]:
                            st.markdown(f"- {obs}")

                # PDF Export Button
                st.markdown("---")
                pdf_report_data = {
                    "filename": input_filename,
                    "media_type": "IMAGE",
                    "prediction": res.prediction,
                    "confidence": res.confidence,
                    "fake_score": res.fake_score,
                    "model_name": res.model_name,
                    "sha256": res.sha256,
                    "md5": res.md5,
                    "size_bytes": res.size_bytes,
                    "metadata_observations": res.metadata.get("forensic_observations", []),
                    "limitations": AVAILABLE_MODELS[res.model_id].known_limitations
                }
                
                # Save temp overlay image for PDF if available
                temp_overlay_path = None
                if res.explanation_overlay_rgb is not None:
                    temp_overlay_path = str(settings.TEMP_DIR / f"overlay_{int(time.time())}.jpg")
                    Image.fromarray(res.explanation_overlay_rgb).save(temp_overlay_path)

                pdf_bytes = generate_forensic_pdf(pdf_report_data, explanation_image_path=temp_overlay_path)
                
                st.download_button(
                    label="📄 Download Forensic PDF Report",
                    data=pdf_bytes,
                    file_name=f"DeepShield_Report_{input_filename.split('.')[0]}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: VIDEO ANALYSIS
# ==============================================================================
elif page == "Video Analysis":
    st.markdown("## 🎬 **Forensic Video Deepfake Analysis**")
    st.markdown("Inspect video streams frame-by-frame with bounded memory sampling, statistical aggregation, and Plotly timeline telemetry.")

    c_vsource, c_vopts = st.columns([2, 1])
    with c_vsource:
        v_choice = st.selectbox(
            "Select Video Source",
            ["Upload Video File", "Use Sample: Video with Face (NASA Astronaut)", "Use Sample: Video Without Face (Rocket)"]
        )

    vid_path_target = None
    v_display_name = "sample_video.mp4"

    if v_choice == "Upload Video File":
        up_vid = st.file_uploader("Choose a video file (MP4, MOV, AVI)", type=["mp4", "mov", "avi"], help="Max: 100MB")
        if up_vid is not None:
            v_display_name = sanitize_filename(up_vid.name)
            # Save safely to temp file for OpenCV reading
            with safe_temp_file(suffix=Path(v_display_name).suffix) as tf:
                tf.write_bytes(up_vid.read())
                vid_path_target = tf
    elif "Video with Face" in v_choice:
        p = settings.SAMPLES_DIR / "sample_video_with_face.mp4"
        if p.exists():
            vid_path_target = p
            v_display_name = p.name
    else:
        p = settings.SAMPLES_DIR / "sample_video_no_face.mp4"
        if p.exists():
            vid_path_target = p
            v_display_name = p.name

    if vid_path_target and Path(vid_path_target).exists():
        st.video(str(vid_path_target))

        # Sampling configuration
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sample_interval = st.slider(
                "Sample Every Nth Frame",
                min_value=settings.MIN_FRAME_SAMPLE_INTERVAL,
                max_value=settings.MAX_FRAME_SAMPLE_INTERVAL,
                value=settings.DEFAULT_FRAME_SAMPLE_INTERVAL,
                help="Higher interval reduces inference time on CPU."
            )
        with col_s2:
            max_frames = st.slider(
                "Max Frames to Sample",
                min_value=5,
                max_value=settings.MAX_FRAMES_TO_ANALYZE,
                value=30,
                help="Limits compute and memory bounds."
            )

        if st.button("🚀 Analyze Video Stream", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(frac, current, total):
                progress_bar.progress(frac)
                status_text.text(f"Sampling & analyzing frame {current}/{total} on CPU...")

            res = video_analyzer.analyze_video(
                video_path=vid_path_target,
                filename=v_display_name,
                model_id=st.session_state.selected_model_id,
                sample_interval=sample_interval,
                max_frames=max_frames,
                threshold=st.session_state.detection_threshold,
                progress_callback=update_progress
            )

            progress_bar.progress(1.0)
            status_text.empty()

            st.markdown("---")

            if not res.is_valid:
                st.error(f"❌ **Validation Failed:** {res.error_message}")
            else:
                summary = res.summary
                
                # Save audit log entry
                save_analysis_record(
                    filename=v_display_name,
                    file_type="video",
                    sha256=res.sha256,
                    model_name=res.model_name,
                    prediction=res.prediction,
                    confidence_score=res.confidence,
                    fake_score=summary.mean_score if summary else 0.0,
                    faces_detected=summary.frames_with_faces if summary else 0,
                    execution_time_ms=res.execution_time_ms,
                    status=res.status,
                    metadata_summary=res.metadata
                )

                # Verdict Box
                v_box_class = (
                    "verdict-box-fake" if res.prediction == "Fake"
                    else "verdict-box-real" if res.prediction == "Real"
                    else "verdict-box-inconclusive"
                )
                v_color = "#ef4444" if res.prediction == "Fake" else "#10b981" if res.prediction == "Real" else "#f59e0b"

                st.markdown(
                    f"""
                    <div class="{v_box_class}">
                        <div class="verdict-label" style="color: {v_color};">{res.prediction.upper()}</div>
                        <div class="verdict-score">
                            Analyzed: <b>{summary.frames_with_faces}/{summary.frames_sampled} frames</b> | 
                            Mean Fake Score: <b>{summary.mean_score:.3f}</b> | 
                            Median: <b>{summary.median_score:.3f}</b> | 
                            Processing Time: <b>{res.execution_time_ms/1000.0:.2f}s</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Statistical Metrics Cards
                sm1, sm2, sm3, sm4 = st.columns(4)
                with sm1:
                    st.metric("Sampled Frames", f"{summary.frames_sampled}")
                with sm2:
                    st.metric("Frames With Faces", f"{summary.frames_with_faces}")
                with sm3:
                    st.metric("Score Variation (Std)", f"{summary.score_std:.3f}")
                with sm4:
                    st.metric("Fake Frame Ratio", f"{summary.fake_frame_ratio*100:.1f}%")

                # Interactive Plotly Timeline Chart
                st.markdown("### 📈 **Frame-Level Forensic Timeline**")
                timeline_data = [
                    {
                        "Frame #": f.frame_index,
                        "Timestamp (s)": f.timestamp,
                        "Fake Probability": f.fake_score if f.fake_score is not None else 0.0,
                        "Status": "Fake" if f.prediction == "Fake" else "Real" if f.prediction == "Real" else "No Face Detected",
                        "Face Detected": "Yes" if f.face_detected else "No"
                    }
                    for f in res.frame_results
                ]
                df_time = pd.DataFrame(timeline_data)

                fig = px.scatter(
                    df_time,
                    x="Timestamp (s)",
                    y="Fake Probability",
                    color="Status",
                    color_discrete_map={"Fake": "#ef4444", "Real": "#10b981", "No Face Detected": "#64748b"},
                    hover_data=["Frame #", "Face Detected"],
                    title="Temporal Manipulation Distribution Across Video Frames"
                )
                fig.add_hline(y=st.session_state.detection_threshold, line_dash="dash", line_color="#f59e0b", annotation_text="Threshold (0.50)")
                fig.update_layout(
                    paper_bgcolor="#0b0f19",
                    plot_bgcolor="#1e293b",
                    font=dict(color="#f1f5f9"),
                    yaxis=dict(range=[-0.05, 1.05], gridcolor="#334155"),
                    xaxis=dict(gridcolor="#334155")
                )
                st.plotly_chart(fig, use_container_width=True)

                # Sampled Frame Gallery / Inspector
                st.markdown("### 🎞️ **Analyzed Frame Gallery**")
                frames_with_img = [f for f in res.frame_results if f.annotated_frame_rgb is not None]
                if frames_with_img:
                    cols = st.columns(min(4, len(frames_with_img)))
                    for idx, f in enumerate(frames_with_img[:8]):
                        col_idx = idx % min(4, len(frames_with_img))
                        with cols[col_idx]:
                            st.image(
                                f.annotated_frame_rgb,
                                caption=f"Frame #{f.frame_index} ({f.timestamp:.1f}s)\n{f.prediction} ({f.fake_score*100:.1f}%)",
                                use_column_width=True
                            )

                # Aggregation Notes
                if summary.aggregation_notes:
                    st.markdown("#### **Forensic Notes & Caveats**")
                    for n in summary.aggregation_notes:
                        st.markdown(f"- {n}")

                # PDF Export
                st.markdown("---")
                pdf_report_data = {
                    "filename": v_display_name,
                    "media_type": "VIDEO",
                    "prediction": res.prediction,
                    "confidence": res.confidence,
                    "fake_score": summary.mean_score,
                    "model_name": res.model_name,
                    "sha256": res.sha256,
                    "md5": res.md5,
                    "size_bytes": res.size_bytes,
                    "metadata_observations": [f"Duration: {res.duration_seconds}s, FPS: {res.fps}"] + summary.aggregation_notes,
                    "video_stats": {
                        "frames_sampled": summary.frames_sampled,
                        "frames_with_faces": summary.frames_with_faces,
                        "mean_score": summary.mean_score,
                        "median_score": summary.median_score,
                        "score_std": summary.score_std,
                        "fake_frame_ratio": summary.fake_frame_ratio
                    },
                    "frame_predictions": [
                        {
                            "frame_index": f.frame_index,
                            "timestamp": f.timestamp,
                            "face_detected": f.face_detected,
                            "prediction": f.prediction,
                            "fake_score": f.fake_score or 0.0
                        }
                        for f in res.frame_results
                    ],
                    "limitations": AVAILABLE_MODELS[res.model_id].known_limitations
                }
                pdf_bytes = generate_forensic_pdf(pdf_report_data)
                
                st.download_button(
                    label="📄 Download Forensic PDF Video Report",
                    data=pdf_bytes,
                    file_name=f"DeepShield_Video_Report_{v_display_name.split('.')[0]}.pdf",
                    mime="application/pdf",
                    type="primary"
                )

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: BATCH ANALYSIS
# ==============================================================================
elif page == "Batch Analysis":
    st.markdown("## 📁 **Batch Image Deepfake Inspection**")
    st.markdown(f"Upload multiple images (up to {settings.MAX_BATCH_IMAGE_COUNT}) to run automated forensic batch analysis with CSV export.")

    uploaded_files = st.file_uploader(
        f"Select Images (Max {settings.MAX_BATCH_IMAGE_COUNT})",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > settings.MAX_BATCH_IMAGE_COUNT:
            st.warning(f"Batch size limited to {settings.MAX_BATCH_IMAGE_COUNT} files to protect CPU and memory.")
            uploaded_files = uploaded_files[:settings.MAX_BATCH_IMAGE_COUNT]

        st.markdown(f"**Loaded Files:** {len(uploaded_files)} images")
        if st.button("🚀 Process Batch", type="primary"):
            pbar = st.progress(0.0)
            status_text = st.empty()

            batch_results = []
            for idx, uf in enumerate(uploaded_files):
                fname = sanitize_filename(uf.name)
                content = uf.read()
                status_text.text(f"Analyzing {fname} ({idx+1}/{len(uploaded_files)})...")

                res = image_analyzer.analyze_image(
                    image_bytes=content,
                    filename=fname,
                    model_id=st.session_state.selected_model_id,
                    threshold=st.session_state.detection_threshold,
                    generate_explanation=False
                )

                batch_results.append({
                    "Filename": fname,
                    "Status": res.status,
                    "Prediction": res.prediction,
                    "Confidence": f"{res.confidence*100:.1f}%",
                    "Fake Score": res.fake_score,
                    "Faces Detected": res.faces_detected_count,
                    "SHA-256": res.sha256[:16] + "...",
                    "Time (ms)": res.execution_time_ms
                })

                # Record in history
                if res.is_valid:
                    save_analysis_record(
                        filename=fname,
                        file_type="image",
                        sha256=res.sha256,
                        model_name=res.model_name,
                        prediction=res.prediction,
                        confidence_score=res.confidence,
                        fake_score=res.fake_score,
                        faces_detected=res.faces_detected_count,
                        execution_time_ms=res.execution_time_ms,
                        status=res.status
                    )

                pbar.progress((idx + 1) / len(uploaded_files))

            pbar.progress(1.0)
            status_text.empty()

            df_batch = pd.DataFrame(batch_results)
            st.markdown("### **Batch Forensic Summary**")
            
            # Summary Metrics
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Total Processed", len(df_batch))
            with b2:
                fake_cnt = sum(1 for r in batch_results if r["Prediction"] == "Fake")
                st.metric("Flagged Fake", fake_cnt)
            with b3:
                real_cnt = sum(1 for r in batch_results if r["Prediction"] == "Real")
                st.metric("Authentic", real_cnt)
            with b4:
                noface_cnt = sum(1 for r in batch_results if r["Prediction"] == "No Face Detected")
                st.metric("No Face", noface_cnt)

            st.dataframe(df_batch, use_container_width=True, hide_index=True)

            # Export to CSV
            csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Batch Results to CSV",
                data=csv_bytes,
                file_name="DeepShield_Batch_Results.csv",
                mime="text/csv"
            )

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 5: ANALYSIS HISTORY
# ==============================================================================
elif page == "Analysis History":
    st.markdown("## 📜 **Forensic Audit Log & Analysis History**")
    st.markdown("Review past forensic predictions stored in SQLite. Raw media payloads and facial vectors are never retained.")

    c_filt1, c_filt2, c_filt3 = st.columns([2, 1, 1])
    with c_filt1:
        search_query = st.text_input("🔍 Search by Filename, Hash, or ID", "")
    with c_filt2:
        pred_filter = st.selectbox("Verdict Filter", ["ALL", "REAL", "FAKE"])
    with c_filt3:
        type_filter = st.selectbox("Media Type", ["ALL", "IMAGE", "VIDEO"])

    records = get_analysis_records(
        search_query=search_query,
        prediction_filter=pred_filter,
        file_type_filter=type_filter,
        limit=100
    )

    if not records:
        st.info("No audit records found matching the current query.")
    else:
        df_history = pd.DataFrame(records)
        display_cols = ["id", "timestamp", "filename", "file_type", "prediction", "confidence_score", "fake_score", "faces_detected", "execution_time_ms"]
        st.dataframe(df_history[display_cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### **Record Inspector & Actions**")
        rec_ids = [r["id"] for r in records]
        selected_id = st.selectbox("Select Audit Record to Inspect", rec_ids)

        if selected_id:
            rec = get_record_by_id(selected_id)
            if rec:
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown(f"**Audit Record ID:** `{rec['id']}`")
                    st.markdown(f"**Timestamp:** `{rec['timestamp']}`")
                    st.markdown(f"**File Name:** `{rec['filename']}` ({rec['file_type']})")
                    st.markdown(f"**SHA-256:** <br><code style='font-size:11px;'>{rec['sha256']}</code>", unsafe_allow_html=True)
                with col_d2:
                    st.markdown(f"**Verdict:** `{rec['prediction']}`")
                    st.markdown(f"**Confidence:** `{rec['confidence_score']*100:.1f}%`")
                    st.markdown(f"**Model:** `{rec['model_name']}`")
                    st.markdown(f"**Execution Time:** `{rec['execution_time_ms']:.1f}ms`")

                # Action Buttons
                c_del, c_pdf = st.columns(2)
                with c_del:
                    if st.button("🗑️ Delete This Record", type="secondary"):
                        delete_record_by_id(selected_id)
                        st.success(f"Deleted record {selected_id}")
                        st.rerun()
                with c_pdf:
                    pdf_data = {
                        "report_id": rec["id"],
                        "timestamp": rec["timestamp"],
                        "filename": rec["filename"],
                        "media_type": rec["file_type"].upper(),
                        "prediction": rec["prediction"],
                        "confidence": rec["confidence_score"],
                        "fake_score": rec["fake_score"],
                        "model_name": rec["model_name"],
                        "sha256": rec["sha256"],
                        "md5": "Preserved in Audit Log",
                        "size_bytes": 0,
                        "metadata_observations": ["Retrieved from persistent SQLite audit database."],
                        "limitations": AVAILABLE_MODELS[DEFAULT_MODEL_ID].known_limitations
                    }
                    pdf_bytes = generate_forensic_pdf(pdf_data)
                    st.download_button(
                        "📄 Download Forensic PDF",
                        data=pdf_bytes,
                        file_name=f"Report_{selected_id}.pdf",
                        mime="application/pdf"
                    )

        st.markdown("---")
        with st.expander("⚠️ Danger Zone: Clear Audit Database"):
            st.write("Erasing audit history removes all logging records permanently. Downloaded PDFs remain unaffected.")
            confirm_clear = st.checkbox("I confirm I want to erase all analysis history.")
            if st.button("Clear All History", disabled=not confirm_clear, type="primary"):
                count = clear_all_history()
                st.success(f"Cleared {count} audit records.")
                st.rerun()

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 6: MODEL EVALUATION
# ==============================================================================
elif page == "Model Evaluation":
    st.markdown("## 📊 **Model Benchmark & Evaluation Suite**")
    st.markdown("Run automated evaluation on labeled datasets with genuine metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.")

    eval_data_dir = settings.EVAL_DATA_DIR
    evaluator = DatasetEvaluator(analyzer=image_analyzer)
    samples = evaluator.discover_dataset_files(eval_data_dir)

    c_e1, c_e2 = st.columns([2, 1])
    with c_e1:
        st.markdown(f"**Evaluation Directory:** `{eval_data_dir}`")
        if samples:
            real_count = sum(1 for _, lbl in samples if lbl == 0)
            fake_count = sum(1 for _, lbl in samples if lbl == 1)
            st.markdown(f"**Labeled Samples Discovered:** {len(samples)} total ({real_count} Real, {fake_count} Fake)")
        else:
            st.info("No labeled dataset found in `data/evaluation/`. Add `real/` and `fake/` subdirectories with images to evaluate.")

    with c_e2:
        max_eval = st.number_input("Max Samples to Test", min_value=1, max_value=500, value=min(20, len(samples)) if samples else 10)

    if samples and st.button("🚀 Run Dataset Evaluation", type="primary"):
        pbar = st.progress(0.0)
        status = st.empty()

        def eval_progress(frac, curr, total):
            pbar.progress(frac)
            status.text(f"Evaluating sample {curr}/{total} on CPU...")

        metrics, df_eval = evaluator.evaluate_directory(
            dataset_dir=eval_data_dir,
            model_id=st.session_state.selected_model_id,
            threshold=st.session_state.detection_threshold,
            max_samples=max_eval,
            progress_callback=eval_progress
        )

        pbar.progress(1.0)
        status.empty()

        if metrics is not None:
            st.markdown("### **Evaluation Performance Metrics**")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Accuracy", f"{metrics.accuracy*100:.1f}%")
            with m2:
                st.metric("Precision", f"{metrics.precision*100:.1f}%")
            with m3:
                st.metric("Recall", f"{metrics.recall*100:.1f}%")
            with m4:
                st.metric("F1-Score", f"{metrics.f1*100:.1f}%")
            with m5:
                auc_str = f"{metrics.roc_auc:.3f}" if metrics.roc_auc is not None else "N/A"
                st.metric("ROC-AUC", auc_str)

            # Confusion Matrix & ROC Plots
            c_cm, c_roc = st.columns(2)
            with c_cm:
                st.markdown("#### **Confusion Matrix**")
                cm_data = np.array(metrics.confusion_matrix)
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predicted Class", y="True Ground Truth", color="Count"),
                    x=["Real (0)", "Fake (1)"],
                    y=["Real (0)", "Fake (1)"],
                    text_auto=True,
                    color_continuous_scale="Blues"
                )
                fig_cm.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#1e293b", font=dict(color="#f1f5f9"))
                st.plotly_chart(fig_cm, use_container_width=True)

            with c_roc:
                st.markdown("#### **ROC Curve**")
                if metrics.roc_fpr and metrics.roc_tpr:
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=metrics.roc_fpr, y=metrics.roc_tpr, mode='lines', name=f'AUC = {metrics.roc_auc:.3f}', line=dict(color='#38bdf8', width=2.5)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash', color='#64748b')))
                    fig_roc.update_layout(
                        paper_bgcolor="#0b0f19",
                        plot_bgcolor="#1e293b",
                        font=dict(color="#f1f5f9"),
                        xaxis=dict(title="False Positive Rate", gridcolor="#334155"),
                        yaxis=dict(title="True Positive Rate", gridcolor="#334155")
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    st.info("ROC Curve requires multi-class variability in sample predictions.")

            # Sample Details Table
            st.markdown("#### **Sample Predictions Table**")
            st.dataframe(df_eval, use_container_width=True, hide_index=True)

            # Export Button
            csv_eval = df_eval.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Evaluation CSV",
                data=csv_eval,
                file_name="DeepShield_Evaluation_Results.csv",
                mime="text/csv"
            )

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 7: MODEL INFORMATION
# ==============================================================================
elif page == "Model Information":
    st.markdown("## ℹ️ **Forensic Model Cards & Documentation**")
    st.markdown("Transparent technical documentation detailing architecture, training provenance, weight licensing, and known operational limitations.")

    for m_id, meta in AVAILABLE_MODELS.items():
        with st.expander(f"📌 {meta.name} ({meta.id})", expanded=(m_id == st.session_state.selected_model_id)):
            st.markdown(f"**Architecture:** `{meta.architecture}`")
            st.markdown(f"**Model Source:** `{meta.source}`")
            st.markdown(f"**License:** `{meta.license}`")
            st.markdown(f"**Input Dimensions:** `{meta.input_size[0]}x{meta.input_size[1]} px`")
            st.markdown(f"**Class Mapping:** `{meta.label_map}` (Index `{meta.fake_class_index}` = Manipulated/Fake)")
            st.markdown("---")
            st.markdown(f"**Overview:**\n{meta.description}")
            st.markdown(f"**Training & Provenance Data:**\n{meta.training_data_info}")
            st.markdown("**Documented Operational Limitations:**")
            for lim in meta.known_limitations:
                st.markdown(f"- ⚠️ {lim}")

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 8: SETTINGS & SYSTEM
# ==============================================================================
elif page == "Settings & System":
    st.markdown("## ⚙️ **System Configuration & Operational Diagnostics**")

    c_set1, c_set2 = st.columns(2)
    with c_set1:
        st.markdown("### **Forensic Detection Sensitivity**")
        new_thresh = st.slider(
            "Classification Decision Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(st.session_state.detection_threshold),
            step=0.05,
            help="Threshold on fake probability score to classify as Fake."
        )
        if new_thresh != st.session_state.detection_threshold:
            st.session_state.detection_threshold = new_thresh
            st.success(f"Threshold updated to {new_thresh:.2f}")

    with c_set2:
        st.markdown("### **Storage & Memory Management**")
        if st.button("🧹 Clear Model Memory Cache"):
            loader.clear_cache()
            st.success("Model memory cache cleared.")
        
        if st.button("🗑️ Clean Temporary Upload Directory"):
            removed = cleanup_directory(settings.TEMP_DIR)
            st.success(f"Removed {removed} temporary files.")

    st.markdown("---")
    st.markdown("### **System Environment Telemetry**")
    import platform
    import sys
    import torch

    sys_info = {
        "DeepShield Version": settings.VERSION,
        "Python Runtime": sys.version.split()[0],
        "PyTorch Version": torch.__version__,
        "Target Compute Device": settings.DEVICE.upper(),
        "Operating System": f"{platform.system()} {platform.release()}",
        "Hardware Target": "12 GB RAM, CPU-Only",
        "Database Path": str(settings.DATABASE_PATH),
        "Temp Directory": str(settings.TEMP_DIR)
    }
    st.dataframe(pd.DataFrame(list(sys_info.items()), columns=["Parameter", "Value"]), use_container_width=True, hide_index=True)

    st.markdown(f"<div class='disclaimer-card'><b>FORENSIC NOTICE:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)
