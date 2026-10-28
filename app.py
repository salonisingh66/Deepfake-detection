"""DeepShield AI: Deepfake Detection and Forensic Analysis System.

Minimal, modern digital forensics interface for CPU-only deepfake verification.
Focused on utility, speed, and clean forensic inspection.
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
    page_title="DeepShield AI - Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal Forensics Theme CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Minimal Header */
    .minimal-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0px 20px 0px;
        border-bottom: 1px solid rgba(59, 130, 246, 0.15);
        margin-bottom: 24px;
    }
    .minimal-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #60a5fa 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .minimal-subtitle {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 2px;
    }
    
    /* Status Badges */
    .badge-ready {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        background-color: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        font-size: 11px;
        font-weight: 600;
    }
    .badge-error {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        background-color: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        font-size: 11px;
        font-weight: 600;
    }
    
    /* Verdict Cards */
    .verdict-box-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.08) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin-bottom: 18px;
    }
    .verdict-box-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.08) 100%);
        border: 1.5px solid #10b981;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin-bottom: 18px;
    }
    .verdict-box-inconclusive {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(180, 83, 9, 0.08) 100%);
        border: 1.5px solid #f59e0b;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin-bottom: 18px;
    }
    
    .verdict-label {
        font-size: 24px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .verdict-sub {
        font-size: 14px;
        color: #cbd5e1;
        margin-top: 4px;
    }
    
    /* Clean Metric Box */
    .stat-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 12px;
    }
    .stat-box-title {
        font-size: 11px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-box-val {
        font-size: 20px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 2px;
    }
    
    /* Minimal Disclaimer */
    .minimal-disclaimer {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 11px;
        color: #64748b;
        line-height: 1.4;
        margin-top: 28px;
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


# Minimal Sidebar: Upload, History, Settings and System ONLY
with st.sidebar:
    st.markdown("### 🛡️ **DeepShield AI**")
    
    is_ready, load_err = loader.is_model_ready(st.session_state.selected_model_id)
    if is_ready:
        st.markdown("<span class='badge-ready'>● Ready (CPU)</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span class='badge-error'>● Offline: {load_err}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Strictly Minimal 3-Item Navigation Menu
    nav_selection = st.radio(
        "Menu",
        ["Upload", "History", "Settings and System"],
        index=0
    )
    
    st.markdown("---")
    st.markdown(
        f"<div style='font-size: 11px; color: #64748b;'>"
        f"<b>Model:</b> {AVAILABLE_MODELS[st.session_state.selected_model_id].name.split()[0]}<br>"
        f"<b>Target:</b> CPU-only (12GB RAM)<br>"
        f"<b>Version:</b> {settings.VERSION}"
        f"</div>",
        unsafe_allow_html=True
    )


# ==============================================================================
# MENU 1: UPLOAD (Unified Minimal Interface for Images, Videos, Batch)
# ==============================================================================
if nav_selection == "Upload":
    st.markdown(
        """
        <div class="minimal-header">
            <div>
                <div class="minimal-title">Forensic Detection & Analysis</div>
                <div class="minimal-subtitle">Upload media to identify facial manipulation, inspect attention regions, and export forensic reports.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    upload_tab, video_tab, batch_tab = st.tabs(["📷 Image", "🎬 Video", "📁 Batch Images"])

    # --------------------------------------------------------------------------
    # TAB 1: SINGLE IMAGE
    # --------------------------------------------------------------------------
    with upload_tab:
        c_src1, c_src2 = st.columns([3, 1])
        with c_src1:
            uploaded_img = st.file_uploader(
                "Upload Image (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="img_uploader",
                help="Maximum file size: 20MB"
            )
        with c_src2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            use_sample_img = st.button("🧪 Use Sample Face", use_container_width=True)

        img_bytes = None
        img_filename = "upload.jpg"

        if uploaded_img is not None:
            img_bytes = uploaded_img.read()
            img_filename = sanitize_filename(uploaded_img.name)
        elif use_sample_img or st.session_state.get("load_sample_face_state"):
            sample_p = settings.SAMPLES_DIR / "sample_face_real.jpg"
            if sample_p.exists():
                img_bytes = sample_p.read_bytes()
                img_filename = sample_p.name
                st.session_state["load_sample_face_state"] = True

        if img_bytes:
            c_prev, c_meta = st.columns([1, 1])
            with c_prev:
                st.image(img_bytes, caption=f"Target: {img_filename}", use_column_width=True)
                run_img = st.button("🚀 Analyze Image", type="primary", use_container_width=True, key="btn_run_img")
            
            with c_meta:
                fps = compute_file_fingerprints(img_bytes)
                st.markdown("##### **File Provenance**")
                st.markdown(f"**Filename:** `{img_filename}`")
                st.markdown(f"**Size:** `{len(img_bytes):,} bytes`")
                st.markdown(f"**SHA-256:** <br><code style='font-size:11px;'>{fps['sha256']}</code>", unsafe_allow_html=True)
                st.markdown(f"**MD5:** <br><code style='font-size:11px;'>{fps['md5']}</code>", unsafe_allow_html=True)

            if run_img:
                with st.spinner("Analyzing facial characteristics & generating attention map..."):
                    res = image_analyzer.analyze_image(
                        image_bytes=img_bytes,
                        filename=img_filename,
                        model_id=st.session_state.selected_model_id,
                        threshold=st.session_state.detection_threshold,
                        generate_explanation=True
                    )

                st.markdown("---")
                if not res.is_valid:
                    st.error(f"❌ **Validation Error:** {res.error_message}")
                elif res.status == "NO_FACES_DETECTED":
                    st.warning(f"⚠️ **Forensic Notice:** {res.error_message}")
                    save_analysis_record(
                        filename=img_filename, file_type="image", sha256=res.sha256,
                        model_name=res.model_name, prediction="No Face", confidence_score=0.0,
                        fake_score=0.0, faces_detected=0, execution_time_ms=res.execution_time_ms,
                        status="NO_FACES_DETECTED", metadata_summary=res.metadata
                    )
                elif res.status == "SUCCESS":
                    save_analysis_record(
                        filename=img_filename, file_type="image", sha256=res.sha256,
                        model_name=res.model_name, prediction=res.prediction, confidence_score=res.confidence,
                        fake_score=res.fake_score, faces_detected=res.faces_detected_count,
                        execution_time_ms=res.execution_time_ms, status="SUCCESS", metadata_summary=res.metadata
                    )

                    box_class = "verdict-box-fake" if res.prediction == "Fake" else "verdict-box-real"
                    color = "#ef4444" if res.prediction == "Fake" else "#10b981"
                    st.markdown(
                        f"""
                        <div class="{box_class}">
                            <div class="verdict-label" style="color: {color};">{res.prediction.upper()}</div>
                            <div class="verdict-sub">
                                Model Confidence: <b>{res.confidence*100:.1f}%</b> | 
                                Manipulated Probability: <b>{res.fake_score*100:.1f}%</b> | 
                                Latency: <b>{res.execution_time_ms:.1f}ms</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Inspection views
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        if res.annotated_image_rgb is not None:
                            st.image(res.annotated_image_rgb, caption=f"Localized Faces ({res.faces_detected_count})", use_column_width=True)
                    with v_col2:
                        if res.explanation_overlay_rgb is not None:
                            st.image(res.explanation_overlay_rgb, caption="ViT Attention Map (Sensitivity)", use_column_width=True)
                        else:
                            st.info("Visual sensitivity overlays are available for Vision Transformer backbones.")

                    with st.expander("🔍 Detailed Technical & Metadata Observations"):
                        meta_df = pd.DataFrame(
                            [{"Parameter": k, "Value": str(v)} for k, v in res.metadata.items() if k not in ("exif_tags", "forensic_observations")]
                        )
                        st.dataframe(meta_df, use_container_width=True, hide_index=True)
                        if res.metadata.get("forensic_observations"):
                            for obs in res.metadata["forensic_observations"]:
                                st.write(f"• {obs}")

                    # One-Click PDF Report Download
                    pdf_data = {
                        "filename": img_filename, "media_type": "IMAGE", "prediction": res.prediction,
                        "confidence": res.confidence, "fake_score": res.fake_score, "model_name": res.model_name,
                        "sha256": res.sha256, "md5": res.md5, "size_bytes": res.size_bytes,
                        "metadata_observations": res.metadata.get("forensic_observations", []),
                        "limitations": AVAILABLE_MODELS[res.model_id].known_limitations
                    }
                    temp_overlay = None
                    if res.explanation_overlay_rgb is not None:
                        temp_overlay = str(settings.TEMP_DIR / f"overlay_{int(time.time())}.jpg")
                        Image.fromarray(res.explanation_overlay_rgb).save(temp_overlay)

                    pdf_bytes = generate_forensic_pdf(pdf_data, explanation_image_path=temp_overlay)
                    st.download_button(
                        label="📄 Download Forensic PDF Report",
                        data=pdf_bytes,
                        file_name=f"DeepShield_{img_filename.split('.')[0]}_Report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

    # --------------------------------------------------------------------------
    # TAB 2: VIDEO ANALYSIS
    # --------------------------------------------------------------------------
    with video_tab:
        c_v1, c_v2 = st.columns([3, 1])
        with c_v1:
            uploaded_vid = st.file_uploader(
                "Upload Video File (MP4, MOV, AVI)",
                type=["mp4", "mov", "avi"],
                key="vid_uploader",
                help="Max size: 100MB"
            )
        with c_v2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            use_sample_vid = st.button("🧪 Use Sample Video", use_container_width=True)

        vid_path_actual = None
        vid_name_disp = "video.mp4"

        if uploaded_vid is not None:
            vid_name_disp = sanitize_filename(uploaded_vid.name)
            with safe_temp_file(suffix=Path(vid_name_disp).suffix) as tf:
                tf.write_bytes(uploaded_vid.read())
                vid_path_actual = tf
        elif use_sample_vid or st.session_state.get("load_sample_vid_state"):
            sample_vp = settings.SAMPLES_DIR / "sample_video_with_face.mp4"
            if sample_vp.exists():
                vid_path_actual = sample_vp
                vid_name_disp = sample_vp.name
                st.session_state["load_sample_vid_state"] = True

        if vid_path_actual and Path(vid_path_actual).exists():
            st.video(str(vid_path_actual))
            
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                v_interval = st.slider("Frame Sample Interval", 5, 45, 15, help="Sample every Nth frame")
            with c_p2:
                v_max = st.slider("Max Frames", 10, 60, 25, help="Upper bound on sampled frames")

            if st.button("🚀 Analyze Video Stream", type="primary", use_container_width=True, key="btn_run_vid"):
                pbar = st.progress(0.0)
                status = st.empty()

                def on_video_prog(frac, cur, tot):
                    pbar.progress(frac)
                    status.text(f"Processing frame {cur}/{tot} on CPU...")

                res = video_analyzer.analyze_video(
                    video_path=vid_path_actual,
                    filename=vid_name_disp,
                    model_id=st.session_state.selected_model_id,
                    sample_interval=v_interval,
                    max_frames=v_max,
                    threshold=st.session_state.detection_threshold,
                    progress_callback=on_video_prog
                )
                pbar.progress(1.0)
                status.empty()

                st.markdown("---")
                if not res.is_valid:
                    st.error(f"❌ **Validation Failed:** {res.error_message}")
                else:
                    summary = res.summary
                    save_analysis_record(
                        filename=vid_name_disp, file_type="video", sha256=res.sha256,
                        model_name=res.model_name, prediction=res.prediction, confidence_score=res.confidence,
                        fake_score=summary.mean_score if summary else 0.0,
                        faces_detected=summary.frames_with_faces if summary else 0,
                        execution_time_ms=res.execution_time_ms, status=res.status, metadata_summary=res.metadata
                    )

                    v_box = (
                        "verdict-box-fake" if res.prediction == "Fake"
                        else "verdict-box-real" if res.prediction == "Real"
                        else "verdict-box-inconclusive"
                    )
                    v_col = "#ef4444" if res.prediction == "Fake" else "#10b981" if res.prediction == "Real" else "#f59e0b"

                    st.markdown(
                        f"""
                        <div class="{v_box}">
                            <div class="verdict-label" style="color: {v_col};">{res.prediction.upper()}</div>
                            <div class="verdict-sub">
                                Analyzed: <b>{summary.frames_with_faces}/{summary.frames_sampled} frames</b> | 
                                Mean Score: <b>{summary.mean_score:.3f}</b> | 
                                Median: <b>{summary.median_score:.3f}</b> | 
                                Process Time: <b>{res.execution_time_ms/1000.0:.2f}s</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Quick 4-stat row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Sampled Frames", f"{summary.frames_sampled}")
                    with m2:
                        st.metric("Frames With Faces", f"{summary.frames_with_faces}")
                    with m3:
                        st.metric("Score Variation (Std)", f"{summary.score_std:.3f}")
                    with m4:
                        st.metric("Fake Frame Ratio", f"{summary.fake_frame_ratio*100:.1f}%")

                    # Timeline Chart
                    timeline_rows = [
                        {
                            "Timestamp (s)": f.timestamp,
                            "Fake Probability": f.fake_score or 0.0,
                            "Verdict": f.prediction,
                            "Face Detected": "Yes" if f.face_detected else "No"
                        }
                        for f in res.frame_results
                    ]
                    fig_timeline = px.scatter(
                        pd.DataFrame(timeline_rows),
                        x="Timestamp (s)",
                        y="Fake Probability",
                        color="Verdict",
                        color_discrete_map={"Fake": "#ef4444", "Real": "#10b981", "No Face": "#64748b"},
                        title="Temporal Manipulation Distribution"
                    )
                    fig_timeline.add_hline(y=st.session_state.detection_threshold, line_dash="dash", line_color="#f59e0b", annotation_text="Threshold")
                    fig_timeline.update_layout(paper_bgcolor="#0b0f19", plot_bgcolor="#1e293b", font=dict(color="#f1f5f9"))
                    st.plotly_chart(fig_timeline, use_container_width=True)

                    # Sampled Frame Thumbnails
                    frames_ready = [f for f in res.frame_results if f.annotated_frame_rgb is not None]
                    if frames_ready:
                        st.markdown("##### **Sampled Frame Inspection**")
                        cols = st.columns(min(4, len(frames_ready)))
                        for idx, f in enumerate(frames_ready[:8]):
                            with cols[idx % min(4, len(frames_ready))]:
                                st.image(f.annotated_frame_rgb, caption=f"t={f.timestamp:.1f}s | {f.prediction}", use_column_width=True)

                    # PDF Export
                    pdf_v_data = {
                        "filename": vid_name_disp, "media_type": "VIDEO", "prediction": res.prediction,
                        "confidence": res.confidence, "fake_score": summary.mean_score, "model_name": res.model_name,
                        "sha256": res.sha256, "md5": res.md5, "size_bytes": res.size_bytes,
                        "metadata_observations": [f"Duration: {res.duration_seconds}s, FPS: {res.fps}"] + summary.aggregation_notes,
                        "video_stats": {
                            "frames_sampled": summary.frames_sampled, "frames_with_faces": summary.frames_with_faces,
                            "mean_score": summary.mean_score, "median_score": summary.median_score,
                            "score_std": summary.score_std, "fake_frame_ratio": summary.fake_frame_ratio
                        },
                        "frame_predictions": [
                            {"frame_index": f.frame_index, "timestamp": f.timestamp, "face_detected": f.face_detected, "prediction": f.prediction, "fake_score": f.fake_score or 0.0}
                            for f in res.frame_results
                        ],
                        "limitations": AVAILABLE_MODELS[res.model_id].known_limitations
                    }
                    pdf_v_bytes = generate_forensic_pdf(pdf_v_data)
                    st.download_button(
                        label="📄 Download Forensic PDF Report",
                        data=pdf_v_bytes,
                        file_name=f"DeepShield_{vid_name_disp.split('.')[0]}_Report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )

    # --------------------------------------------------------------------------
    # TAB 3: BATCH IMAGES
    # --------------------------------------------------------------------------
    with batch_tab:
        batch_files = st.file_uploader(
            f"Select Multiple Images (Up to {settings.MAX_BATCH_IMAGE_COUNT})",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="batch_uploader"
        )
        if batch_files:
            if len(batch_files) > settings.MAX_BATCH_IMAGE_COUNT:
                st.warning(f"Batch capped at {settings.MAX_BATCH_IMAGE_COUNT} images to preserve CPU memory.")
                batch_files = batch_files[:settings.MAX_BATCH_IMAGE_COUNT]

            st.write(f"Selected **{len(batch_files)}** files.")
            if st.button("🚀 Process Batch", type="primary", key="btn_run_batch"):
                bpbar = st.progress(0.0)
                bstatus = st.empty()

                rows = []
                for idx, bf in enumerate(batch_files):
                    bstatus.text(f"Processing {bf.name} ({idx+1}/{len(batch_files)})...")
                    b_bytes = bf.read()
                    b_name = sanitize_filename(bf.name)
                    r = image_analyzer.analyze_image(
                        image_bytes=b_bytes, filename=b_name,
                        model_id=st.session_state.selected_model_id,
                        threshold=st.session_state.detection_threshold,
                        generate_explanation=False
                    )
                    rows.append({
                        "Filename": b_name,
                        "Prediction": r.prediction,
                        "Confidence": f"{r.confidence*100:.1f}%",
                        "Fake Score": r.fake_score,
                        "Faces": r.faces_detected_count,
                        "SHA-256": r.sha256[:12] + "...",
                        "Time (ms)": r.execution_time_ms
                    })
                    if r.is_valid:
                        save_analysis_record(
                            filename=b_name, file_type="image", sha256=r.sha256,
                            model_name=r.model_name, prediction=r.prediction,
                            confidence_score=r.confidence, fake_score=r.fake_score,
                            faces_detected=r.faces_detected_count,
                            execution_time_ms=r.execution_time_ms, status=r.status
                        )
                    bpbar.progress((idx + 1) / len(batch_files))

                bpbar.progress(1.0)
                bstatus.empty()

                df_b = pd.DataFrame(rows)
                st.markdown("##### **Batch Results Summary**")
                st.dataframe(df_b, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Export Results as CSV",
                    data=df_b.to_csv(index=False).encode("utf-8"),
                    file_name="DeepShield_Batch_Results.csv",
                    mime="text/csv"
                )

    st.markdown(f"<div class='minimal-disclaimer'><b>LEGAL & FORENSIC DISCLAIMER:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# MENU 2: HISTORY (Audit Trail, Search, PDF Export, Deletion)
# ==============================================================================
elif nav_selection == "History":
    st.markdown(
        """
        <div class="minimal-header">
            <div>
                <div class="minimal-title">Forensic Audit Log</div>
                <div class="minimal-subtitle">Persistent SQLite history of past verifications. Biometric data and raw media are never stored.</div>
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

    st.markdown(f"<div class='minimal-disclaimer'><b>LEGAL & FORENSIC DISCLAIMER:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)


# ==============================================================================
# MENU 3: SETTINGS AND SYSTEM (Diagnostics, Telemetry, Benchmarks, Docs)
# ==============================================================================
elif nav_selection == "Settings and System":
    st.markdown(
        """
        <div class="minimal-header">
            <div>
                <div class="minimal-title">Settings & System Diagnostics</div>
                <div class="minimal-subtitle">Adjust detection sensitivity, view hardware telemetry, run evaluation benchmarks, and inspect model cards.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c_cfg1, c_cfg2 = st.columns(2)
    with c_cfg1:
        st.markdown("##### **Model Configuration**")
        chosen_model_id = st.selectbox(
            "Active Forensic Backbone",
            options=list(AVAILABLE_MODELS.keys()),
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model_id),
            format_func=lambda m: AVAILABLE_MODELS[m].name
        )
        if chosen_model_id != st.session_state.selected_model_id:
            st.session_state.selected_model_id = chosen_model_id
            st.success(f"Switched model to {AVAILABLE_MODELS[chosen_model_id].name}")
            st.rerun()

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

    st.markdown(f"<div class='minimal-disclaimer'><b>LEGAL & FORENSIC DISCLAIMER:</b> {MANDATORY_DISCLAIMER}</div>", unsafe_allow_html=True)
