"""DeepShield AI: Deepfake Detection and Forensic Analysis System.

Unified digital forensics interface supporting both image and video media
with a minimal, high-utility cybersecurity dashboard aesthetic.
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

# Application Core Imports
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
    page_title="DeepShield AI - Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Cyber-Forensics Dark Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main Background */
    .stApp {
        background-color: #080c14;
        color: #f1f5f9;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(59, 130, 246, 0.12) !important;
    }
    
    /* Brand Header in Sidebar */
    .brand-container {
        padding: 6px 0px 16px 0px;
        margin-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .brand-title {
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #60a5fa 0%, #38bdf8 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-tag {
        font-size: 11px;
        color: #64748b;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    
    /* Pulse Online Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 9999px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #10b981;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        margin-right: 7px;
        box-shadow: 0 0 8px #10b981;
        animation: pulseAnimation 2s infinite;
    }
    @keyframes pulseAnimation {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.15); opacity: 1; box-shadow: 0 0 12px #10b981; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }
    
    /* Sidebar Navigation Menu Tiles */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        gap: 6px;
    }
    div[data-testid="stRadio"] label {
        background: rgba(30, 41, 59, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin-bottom: 2px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }
    div[data-testid="stRadio"] label:hover {
        background: rgba(59, 130, 246, 0.12) !important;
        border-color: rgba(59, 130, 246, 0.35) !important;
        transform: translateX(2px) !important;
    }
    div[data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.22) 0%, rgba(30, 41, 59, 0.5) 100%) !important;
        border: 1px solid #3b82f6 !important;
        border-left: 3.5px solid #38bdf8 !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.12) !important;
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
    }
    
    /* Sleek Top Banner */
    .top-banner {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 20px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(59, 130, 246, 0.16);
        border-radius: 12px;
        margin-bottom: 24px;
        backdrop-filter: blur(8px);
    }
    .top-title {
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.4px;
        color: #f8fafc;
    }
    .top-sub {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 2px;
    }
    
    /* Unified Upload Dropzone Card */
    .upload-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.6) 0%, rgba(11, 15, 25, 0.8) 100%);
        border: 1.5px dashed rgba(59, 130, 246, 0.35);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 20px;
        transition: border-color 0.2s ease;
    }
    .upload-card:hover {
        border-color: #38bdf8;
    }
    
    /* Result Verdict Banners */
    .verdict-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 16px 0px;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.15);
    }
    .verdict-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(4, 120, 87, 0.08) 100%);
        border: 1.5px solid #10b981;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 16px 0px;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
    }
    .verdict-inconclusive {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(180, 83, 9, 0.08) 100%);
        border: 1.5px solid #f59e0b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 16px 0px;
    }
    .verdict-head {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .verdict-meta {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 4px;
    }
    
    /* Forensic Stat Tiles */
    .stat-tile {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
    }
    .stat-title {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-val {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 2px;
    }
    
    /* Subtle Legal Footer */
    .forensic-disclaimer {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 11px;
        color: #64748b;
        line-height: 1.4;
        margin-top: 32px;
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


# Minimal Sidebar
with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">🛡️ DeepShield AI</div>
            <div class="brand-tag">Deepfake Forensics Lab</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    is_ready, load_err = loader.is_model_ready(st.session_state.selected_model_id)
    if is_ready:
        st.markdown(
            """
            <div class="status-badge">
                <span class="pulse-dot"></span> System Ready • CPU
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"<div style='color:#ef4444; font-size:11px;'>● Offline: {load_err}</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)
    
    # Minimal 3-Item Menu
    nav_selection = st.radio(
        "Navigation",
        ["Upload", "History", "Settings and System"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        f"<div style='font-size: 11px; color: #64748b; line-height: 1.6;'>"
        f"<b>Model:</b> {AVAILABLE_MODELS[st.session_state.selected_model_id].name.split()[0]}<br>"
        f"<b>Target:</b> 12GB RAM (CPU-Only)<br>"
        f"<b>Version:</b> {settings.VERSION}"
        f"</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# MENU 1: UPLOAD (ONE UNIFIED UPLOAD FOR BOTH PHOTO & VIDEO)
# ==============================================================================
if nav_selection == "Upload":
    st.markdown(
        """
        <div class="top-banner">
            <div>
                <div class="top-title">Forensic Detection & Verification</div>
                <div class="top-sub">Upload an image or video to analyze facial manipulations, view attention regions, and export PDF reports.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ONE UNIFIED UPLOADER FOR BOTH PHOTOS AND VIDEOS
    uploaded_file = st.file_uploader(
        "Choose an image or video file",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"],
        help="Supported formats: JPG, PNG, WEBP, MP4, MOV, AVI. Maximum size: 20MB (Image) / 100MB (Video)",
        label_visibility="visible"
    )

    if not uploaded_file:
        st.markdown(
            """
            <div class="upload-card">
                <div style="font-size: 32px; margin-bottom: 8px;">📂</div>
                <div style="font-size: 15px; font-weight: 600; color: #cbd5e1;">Drag and drop any photo or video above</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    Supported: JPG, JPEG, PNG, WEBP, MP4, MOV, AVI
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        filename = sanitize_filename(uploaded_file.name)
        ext = Path(filename).suffix.lower()
        file_bytes = uploaded_file.read()

        is_image = ext in settings.ALLOWED_IMAGE_EXTENSIONS
        is_video = ext in settings.ALLOWED_VIDEO_EXTENSIONS

        # ----------------------------------------------------------------------
        # MEDIA TYPE 1: IMAGE PROCESSING PIPELINE
        # ----------------------------------------------------------------------
        if is_image:
            col_preview, col_provenance = st.columns([1, 1])
            with col_preview:
                st.image(file_bytes, caption=f"Target Image: {filename}", use_column_width=True)
                run_analysis_img = st.button("🚀 Analyze Image", type="primary", use_container_width=True)

            with col_provenance:
                fingerprints = compute_file_fingerprints(file_bytes)
                st.markdown("##### **File Provenance**")
                st.markdown(f"**Filename:** `{filename}`")
                st.markdown(f"**Media Type:** `Image ({ext.upper()})`")
                st.markdown(f"**Size:** `{len(file_bytes):,} bytes`")
                st.markdown(f"**SHA-256:** <br><code style='font-size:11px;'>{fingerprints['sha256']}</code>", unsafe_allow_html=True)
                st.markdown(f"**MD5:** <br><code style='font-size:11px;'>{fingerprints['md5']}</code>", unsafe_allow_html=True)

            if run_analysis_img:
                with st.spinner("Analyzing facial features & computing Vision Transformer attention map..."):
                    res = image_analyzer.analyze_image(
                        image_bytes=file_bytes,
                        filename=filename,
                        model_id=st.session_state.selected_model_id,
                        threshold=st.session_state.detection_threshold,
                        generate_explanation=True
                    )

                st.markdown("---")
                if not res.is_valid:
                    st.error(f"❌ **Validation Failed:** {res.error_message}")
                elif res.status == "NO_FACES_DETECTED":
                    st.warning(f"⚠️ **Forensic Notice:** {res.error_message}")
                    save_analysis_record(
                        filename=filename, file_type="image", sha256=res.sha256,
                        model_name=res.model_name, prediction="No Face", confidence_score=0.0,
                        fake_score=0.0, faces_detected=0, execution_time_ms=res.execution_time_ms,
                        status="NO_FACES_DETECTED", metadata_summary=res.metadata
                    )
                elif res.status == "SUCCESS":
                    save_analysis_record(
                        filename=filename, file_type="image", sha256=res.sha256,
                        model_name=res.model_name, prediction=res.prediction, confidence_score=res.confidence,
                        fake_score=res.fake_score, faces_detected=res.faces_detected_count,
                        execution_time_ms=res.execution_time_ms, status="SUCCESS", metadata_summary=res.metadata
                    )

                    box_css = "verdict-fake" if res.prediction == "Fake" else "verdict-real"
                    box_color = "#ef4444" if res.prediction == "Fake" else "#10b981"
                    st.markdown(
                        f"""
                        <div class="{box_css}">
                            <div class="verdict-head" style="color: {box_color};">{res.prediction.upper()}</div>
                            <div class="verdict-meta">
                                Model Confidence: <b>{res.confidence*100:.1f}%</b> &nbsp;|&nbsp; 
                                Manipulated Probability: <b>{res.fake_score*100:.1f}%</b> &nbsp;|&nbsp; 
                                Inference Latency: <b>{res.execution_time_ms:.1f}ms</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    c_view1, c_view2 = st.columns(2)
                    with c_view1:
                        if res.annotated_image_rgb is not None:
                            st.image(res.annotated_image_rgb, caption=f"Localized Faces ({res.faces_detected_count})", use_column_width=True)
                    with c_view2:
                        if res.explanation_overlay_rgb is not None:
                            st.image(res.explanation_overlay_rgb, caption="ViT Attention Map (Model Sensitivity)", use_column_width=True)
                        else:
                            st.info("Attention overlays are generated for Vision Transformer backbones.")

                    with st.expander("🔍 Technical & EXIF Observations"):
                        meta_df = pd.DataFrame(
                            [{"Parameter": k, "Value": str(v)} for k, v in res.metadata.items() if k not in ("exif_tags", "forensic_observations")]
                        )
                        st.dataframe(meta_df, use_container_width=True, hide_index=True)
                        if res.metadata.get("forensic_observations"):
                            for obs in res.metadata["forensic_observations"]:
                                st.write(f"• {obs}")

                    # One-Click PDF Report Export
                    pdf_data = {
                        "filename": filename, "media_type": "IMAGE", "prediction": res.prediction,
                        "confidence": res.confidence, "fake_score": res.fake_score, "model_name": res.model_name,
                        "sha256": res.sha256, "md5": res.md5, "size_bytes": res.size_bytes,
                        "metadata_observations": res.metadata.get("forensic_observations", []),
                        "limitations": AVAILABLE_MODELS[res.model_id].known_limitations
                    }
                    temp_overlay_p = None
                    if res.explanation_overlay_rgb is not None:
                        temp_overlay_p = str(settings.TEMP_DIR / f"overlay_{int(time.time())}.jpg")
                        Image.fromarray(res.explanation_overlay_rgb).save(temp_overlay_p)

                    st.download_button(
                        label="📄 Download Forensic PDF Report",
                        data=generate_forensic_pdf(pdf_data, explanation_image_path=temp_overlay_p),
                        file_name=f"DeepShield_{filename.split('.')[0]}_Report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

        # ----------------------------------------------------------------------
        # MEDIA TYPE 2: VIDEO PROCESSING PIPELINE
        # ----------------------------------------------------------------------
        elif is_video:
            with safe_temp_file(suffix=ext) as temp_video_file:
                temp_video_file.write_bytes(file_bytes)
                
                col_vprev, col_vparams = st.columns([1, 1])
                with col_vprev:
                    st.video(str(temp_video_file))
                    run_analysis_vid = st.button("🚀 Analyze Video Stream", type="primary", use_container_width=True)

                with col_vparams:
                    st.markdown("##### **Video Sampling Parameters**")
                    v_interval = st.slider("Sample Interval (Frames)", 5, 45, 15, help="Sample every Nth frame")
                    v_max = st.slider("Max Frames to Inspect", 10, 60, 25, help="Protects CPU and memory bounds")
                    
                    v_fps = compute_file_fingerprints(file_bytes)
                    st.markdown(f"**Filename:** `{filename}`")
                    st.markdown(f"**Size:** `{len(file_bytes):,} bytes`")
                    st.markdown(f"**SHA-256:** <br><code style='font-size:11px;'>{v_fps['sha256']}</code>", unsafe_allow_html=True)

                if run_analysis_vid:
                    pbar = st.progress(0.0)
                    status_lbl = st.empty()

                    def on_vid_progress(frac, curr, total):
                        pbar.progress(frac)
                        status_lbl.text(f"Streaming and analyzing frame {curr}/{total} on CPU...")

                    res_v = video_analyzer.analyze_video(
                        video_path=temp_video_file,
                        filename=filename,
                        model_id=st.session_state.selected_model_id,
                        sample_interval=v_interval,
                        max_frames=v_max,
                        threshold=st.session_state.detection_threshold,
                        progress_callback=on_vid_progress
                    )
                    pbar.progress(1.0)
                    status_lbl.empty()

                    st.markdown("---")
                    if not res_v.is_valid:
                        st.error(f"❌ **Validation Failed:** {res_v.error_message}")
                    else:
                        summary = res_v.summary
                        save_analysis_record(
                            filename=filename, file_type="video", sha256=res_v.sha256,
                            model_name=res_v.model_name, prediction=res_v.prediction, confidence_score=res_v.confidence,
                            fake_score=summary.mean_score if summary else 0.0,
                            faces_detected=summary.frames_with_faces if summary else 0,
                            execution_time_ms=res_v.execution_time_ms, status=res_v.status, metadata_summary=res_v.metadata
                        )

                        v_css = (
                            "verdict-fake" if res_v.prediction == "Fake"
                            else "verdict-real" if res_v.prediction == "Real"
                            else "verdict-inconclusive"
                        )
                        v_col = "#ef4444" if res_v.prediction == "Fake" else "#10b981" if res_v.prediction == "Real" else "#f59e0b"

                        st.markdown(
                            f"""
                            <div class="{v_css}">
                                <div class="verdict-head" style="color: {v_col};">{res_v.prediction.upper()}</div>
                                <div class="verdict-meta">
                                    Sampled: <b>{summary.frames_with_faces}/{summary.frames_sampled} frames with faces</b> &nbsp;|&nbsp; 
                                    Mean Score: <b>{summary.mean_score:.3f}</b> &nbsp;|&nbsp; 
                                    Median: <b>{summary.median_score:.3f}</b> &nbsp;|&nbsp; 
                                    Process Time: <b>{res_v.execution_time_ms/1000.0:.2f}s</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # 4 Stat Tiles
                        s1, s2, s3, s4 = st.columns(4)
                        with s1:
                            st.metric("Sampled Frames", f"{summary.frames_sampled}")
                        with s2:
                            st.metric("Frames With Faces", f"{summary.frames_with_faces}")
                        with s3:
                            st.metric("Score Variation (Std)", f"{summary.score_std:.3f}")
                        with s4:
                            st.metric("Fake Frame Ratio", f"{summary.fake_frame_ratio*100:.1f}%")

                        # Timeline Chart
                        st.markdown("##### **Temporal Manipulation Timeline**")
                        tl_rows = [
                            {
                                "Timestamp (s)": f.timestamp,
                                "Fake Probability": f.fake_score or 0.0,
                                "Verdict": f.prediction,
                                "Face Detected": "Yes" if f.face_detected else "No"
                            }
                            for f in res_v.frame_results
                        ]
                        fig = px.scatter(
                            pd.DataFrame(tl_rows),
                            x="Timestamp (s)",
                            y="Fake Probability",
                            color="Verdict",
                            color_discrete_map={"Fake": "#ef4444", "Real": "#10b981", "No Face": "#64748b"}
                        )
                        fig.add_hline(y=st.session_state.detection_threshold, line_dash="dash", line_color="#f59e0b", annotation_text="Threshold")
                        fig.update_layout(paper_bgcolor="#080c14", plot_bgcolor="#111827", font=dict(color="#f1f5f9"), height=300)
                        st.plotly_chart(fig, use_container_width=True)

                        # Sampled Frame Gallery
                        f_imgs = [f for f in res_v.frame_results if f.annotated_frame_rgb is not None]
                        if f_imgs:
                            st.markdown("##### **Analyzed Frame Crops**")
                            cols = st.columns(min(4, len(f_imgs)))
                            for idx, f in enumerate(f_imgs[:8]):
                                with cols[idx % min(4, len(f_imgs))]:
                                    st.image(f.annotated_frame_rgb, caption=f"t={f.timestamp:.1f}s | {f.prediction}", use_column_width=True)

                        # PDF Video Report Export
                        pdf_v_data = {
                            "filename": filename, "media_type": "VIDEO", "prediction": res_v.prediction,
                            "confidence": res_v.confidence, "fake_score": summary.mean_score, "model_name": res_v.model_name,
                            "sha256": res_v.sha256, "md5": res_v.md5, "size_bytes": res_v.size_bytes,
                            "metadata_observations": [f"Duration: {res_v.duration_seconds}s, FPS: {res_v.fps}"] + summary.aggregation_notes,
                            "video_stats": {
                                "frames_sampled": summary.frames_sampled, "frames_with_faces": summary.frames_with_faces,
                                "mean_score": summary.mean_score, "median_score": summary.median_score,
                                "score_std": summary.score_std, "fake_frame_ratio": summary.fake_frame_ratio
                            },
                            "frame_predictions": [
                                {"frame_index": f.frame_index, "timestamp": f.timestamp, "face_detected": f.face_detected, "prediction": f.prediction, "fake_score": f.fake_score or 0.0}
                                for f in res_v.frame_results
                            ],
                            "limitations": AVAILABLE_MODELS[res_v.model_id].known_limitations
                        }
                        st.download_button(
                            label="📄 Download Forensic PDF Report",
                            data=generate_forensic_pdf(pdf_v_data),
                            file_name=f"DeepShield_{filename.split('.')[0]}_Report.pdf",
                            mime="application/pdf",
                            type="primary"
                        )
        else:
            st.error(f"Unsupported file format '{ext}'. Please upload an image (JPG, PNG, WEBP) or video (MP4, MOV, AVI).")

    st.markdown(f"<div class='forensic-disclaimer'><b>LEGAL & FORENSIC DISCLAIMER:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# MENU 2: HISTORY (Audit Trail, Search, PDF Export, Deletion)
# ==============================================================================
elif nav_selection == "History":
    st.markdown(
        """
        <div class="top-banner">
            <div>
                <div class="top-title">Forensic Audit Log</div>
                <div class="top-sub">Persistent SQLite history of past verifications. Biometric vectors and media are never retained.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c_h1, c_h2, c_h3 = st.columns([2, 1, 1])
    with c_h1:
        q_search = st.text_input("🔍 Search by Filename, Hash, or Record ID", "")
    with c_h2:
        f_verdict = st.selectbox("Verdict", ["ALL", "REAL", "FAKE"])
    with c_h3:
        f_type = st.selectbox("Media Type", ["ALL", "IMAGE", "VIDEO"])

    hist_records = get_analysis_records(
        search_query=q_search,
        prediction_filter=f_verdict,
        file_type_filter=f_type,
        limit=100
    )

    if not hist_records:
        st.info("No audit records found matching criteria.")
    else:
        df_hist = pd.DataFrame(hist_records)
        display_fields = ["id", "timestamp", "filename", "file_type", "prediction", "confidence_score", "fake_score", "faces_detected", "execution_time_ms"]
        st.dataframe(df_hist[display_fields], use_container_width=True, hide_index=True)

        st.markdown("---")
        rec_ids = [r["id"] for r in hist_records]
        selected_rec_id = st.selectbox("Select Record to Inspect / Export", rec_ids)

        if selected_rec_id:
            sel_rec = get_record_by_id(selected_rec_id)
            if sel_rec:
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.write(f"**Record ID:** `{sel_rec['id']}`")
                    st.write(f"**Timestamp:** `{sel_rec['timestamp']}`")
                    st.write(f"**Target Media:** `{sel_rec['filename']}` ({sel_rec['file_type']})")
                    st.write(f"**SHA-256:** `{sel_rec['sha256']}`")
                with col_i2:
                    st.write(f"**Verdict:** `{sel_rec['prediction']}`")
                    st.write(f"**Confidence:** `{sel_rec['confidence_score']*100:.1f}%`")
                    st.write(f"**Model:** `{sel_rec['model_name']}`")
                    st.write(f"**Processing Time:** `{sel_rec['execution_time_ms']:.1f}ms`")

                c_btn1, c_btn2 = st.columns(2)
                with c_btn1:
                    pdf_h_data = {
                        "report_id": sel_rec["id"], "timestamp": sel_rec["timestamp"],
                        "filename": sel_rec["filename"], "media_type": sel_rec["file_type"].upper(),
                        "prediction": sel_rec["prediction"], "confidence": sel_rec["confidence_score"],
                        "fake_score": sel_rec["fake_score"], "model_name": sel_rec["model_name"],
                        "sha256": sel_rec["sha256"], "md5": "Preserved in Audit Log",
                        "size_bytes": 0, "metadata_observations": ["Retrieved from persistent SQLite audit database."],
                        "limitations": AVAILABLE_MODELS[DEFAULT_MODEL_ID].known_limitations
                    }
                    st.download_button(
                        "📄 Download Forensic PDF",
                        data=generate_forensic_pdf(pdf_h_data),
                        file_name=f"Report_{sel_rec['id']}.pdf",
                        mime="application/pdf",
                        key="btn_hist_pdf"
                    )
                with c_btn2:
                    if st.button("🗑️ Delete Record", type="secondary", key="btn_hist_del"):
                        delete_record_by_id(selected_rec_id)
                        st.success(f"Deleted record {selected_rec_id}")
                        st.rerun()

        st.markdown("---")
        with st.expander("⚠️ Danger Zone: Clear Audit Database"):
            st.write("Permanently clears all audit logs. Downloaded PDF files remain intact.")
            confirm = st.checkbox("Confirm permanent deletion of all records.")
            if st.button("Clear All History", disabled=not confirm, type="primary"):
                c = clear_all_history()
                st.success(f"Erased {c} records.")
                st.rerun()

    st.markdown(f"<div class='forensic-disclaimer'><b>LEGAL & FORENSIC DISCLAIMER:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# MENU 3: SETTINGS AND SYSTEM (Diagnostics, Telemetry, Benchmarks, Docs)
# ==============================================================================
elif nav_selection == "Settings and System":
    st.markdown(
        """
        <div class="top-banner">
            <div>
                <div class="top-title">Settings & System Diagnostics</div>
                <div class="top-sub">Adjust detection sensitivity, view hardware telemetry, run evaluation benchmarks, and inspect model cards.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        st.markdown("##### **Forensic Model Architecture**")
        st.markdown(
            f"**Model:** Vision Transformer (`vit-base-patch16-224`)<br>"
            f"**Weights:** `dima806/deepfake_vs_real_image_detection`<br>"
            f"**Training:** 140,000 real & manipulated faces (FaceForensics++, Celeb-DF v2)<br>"
            f"**Status:** Verified Genuine Pretrained Weights (SafeTensors)",
            unsafe_allow_html=True
        )
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        new_t = st.slider(
            "Manipulation Decision Threshold",
            min_value=0.10, max_value=0.90, value=float(st.session_state.detection_threshold), step=0.05,
            help="Threshold on fake probability score to classify an image or frame as Fake."
        )
        if new_t != st.session_state.detection_threshold:
            st.session_state.detection_threshold = new_t

    with c_cfg2:
        st.markdown("##### **Maintenance Controls**")
        if st.button("🧹 Clear Model Memory Cache", use_container_width=True):
            loader.clear_cache()
            st.success("Model memory cache cleared.")
        
        if st.button("🗑️ Clean Temporary Storage", use_container_width=True):
            n_removed = cleanup_directory(settings.TEMP_DIR)
            st.success(f"Removed {n_removed} temporary files.")

    # Telemetry Table
    st.markdown("---")
    st.markdown("##### **Hardware & Runtime Telemetry**")
    import platform
    import sys
    import torch

    telemetry = {
        "Application Version": settings.VERSION,
        "Python Runtime": sys.version.split()[0],
        "PyTorch Version": torch.__version__,
        "Compute Architecture": settings.DEVICE.upper(),
        "Host Platform": f"{platform.system()} {platform.release()}",
        "RAM Target": "12 GB RAM (CPU-Only)",
        "Database Path": str(settings.DATABASE_PATH),
        "Temp Directory": str(settings.TEMP_DIR)
    }
    st.dataframe(pd.DataFrame(list(telemetry.items()), columns=["Parameter", "Status"]), use_container_width=True, hide_index=True)

    # Benchmark Suite Expander
    st.markdown("---")
    with st.expander("📊 Dataset Benchmark & Model Evaluation Suite", expanded=False):
        st.markdown("Evaluate detector accuracy, precision, recall, and ROC-AUC on labeled test sets in `data/evaluation/`.")
        evaluator = DatasetEvaluator(analyzer=image_analyzer)
        found_samples = evaluator.discover_dataset_files(settings.EVAL_DATA_DIR)
        
        if not found_samples:
            st.info("No benchmark dataset located in `data/evaluation/`. Add `real/` and `fake/` subfolders to run evaluation.")
        else:
            st.write(f"Located **{len(found_samples)}** labeled samples in `{settings.EVAL_DATA_DIR}`.")
            if st.button("🚀 Run Dataset Benchmark", type="primary"):
                p_eval = st.progress(0.0)
                status_eval = st.empty()

                def on_eval_prog(frac, cur, tot):
                    p_eval.progress(frac)
                    status_eval.text(f"Evaluating sample {cur}/{tot}...")

                metrics, df_eval_res = evaluator.evaluate_directory(
                    dataset_dir=settings.EVAL_DATA_DIR,
                    model_id=st.session_state.selected_model_id,
                    threshold=st.session_state.detection_threshold,
                    progress_callback=on_eval_prog
                )
                p_eval.progress(1.0)
                status_eval.empty()

                if metrics:
                    em1, em2, em3, em4, em5 = st.columns(5)
                    with em1:
                        st.metric("Accuracy", f"{metrics.accuracy*100:.1f}%")
                    with em2:
                        st.metric("Precision", f"{metrics.precision*100:.1f}%")
                    with em3:
                        st.metric("Recall", f"{metrics.recall*100:.1f}%")
                    with em4:
                        st.metric("F1-Score", f"{metrics.f1*100:.1f}%")
                    with em5:
                        auc_val = f"{metrics.roc_auc:.3f}" if metrics.roc_auc is not None else "N/A"
                        st.metric("ROC-AUC", auc_val)

                    st.dataframe(df_eval_res, use_container_width=True, hide_index=True)

    # Model Cards Expander
    with st.expander("ℹ️ Model Cards & Architecture Specifications", expanded=False):
        for mid, mcard in AVAILABLE_MODELS.items():
            st.markdown(f"**{mcard.name}** (`{mid}`)")
            st.markdown(f"- Architecture: `{mcard.architecture}`")
            st.markdown(f"- Source: `{mcard.source}`")
            st.markdown(f"- License: `{mcard.license}`")
            st.markdown(f"- Inputs: `{mcard.input_size[0]}x{mcard.input_size[1]}` px")
            st.markdown(f"- Description: {mcard.description}")
            st.markdown(f"- Known Limitations: {'; '.join(mcard.known_limitations)}")
            st.markdown("---")

    st.markdown(f"<div class='forensic-disclaimer'><b>LEGAL & FORENSIC DISCLAIMER:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)
