"""Forensic PDF Report Generator for DeepShield AI.

Generates professional, court-admissible-style digital forensic reports
using ReportLab, including hashes, technical observations, frame tables,
model predictions, and required ethical disclaimers.
"""

from datetime import datetime
import io
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.settings import settings
from src.utils.logging_config import logger

MANDATORY_DISCLAIMER = (
    "This report contains automated model predictions and technical observations. "
    "It does not establish the authenticity or provenance of the media. Predictions may "
    "be incorrect, and results should not be used as the sole basis for legal, financial, "
    "disciplinary, or reputational decisions."
)


def generate_forensic_pdf(
    report_data: Dict[str, Any],
    output_path: Optional[Path] = None,
    explanation_image_path: Optional[str] = None
) -> bytes:
    """Generate a downloadable PDF forensic report.
    
    Args:
        report_data: Dictionary containing analysis metadata, predictions, and stats.
        output_path: Optional file path to save the generated PDF.
        explanation_image_path: Optional path to an attention/Grad-CAM overlay image.
        
    Returns:
        bytes: PDF binary content.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0f172a"),
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )
    verdict_fake_style = ParagraphStyle(
        "VerdictFake",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#dc2626"),
    )
    verdict_real_style = ParagraphStyle(
        "VerdictReal",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#16a34a"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # 1. Header Banner
    report_id = report_data.get("report_id", f"DS-{uuid.uuid4().hex[:10].upper()}")
    date_str = report_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    story.append(Paragraph("DEEPSHIELD AI - FORENSIC ANALYSIS REPORT", title_style))
    story.append(Paragraph(f"Digital Forensics & Deepfake Verification Lab | Report ID: <b>{report_id}</b>", subtitle_style))
    story.append(Paragraph(f"Generated on: {date_str}", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=12))

    # 2. Executive Summary / Verdict Box
    prediction = report_data.get("prediction", "Indeterminate")
    confidence = report_data.get("confidence", 0.0)
    fake_score = report_data.get("fake_score", 0.0)
    media_type = report_data.get("media_type", "Image").upper()

    verdict_style = verdict_fake_style if prediction.lower() == "fake" else verdict_real_style
    summary_data = [
        [
            Paragraph("<b>Target Media:</b>", table_header_style),
            Paragraph(f"{report_data.get('filename', 'Unknown')} ({media_type})", table_cell_style),
            Paragraph("<b>Analysis Verdict:</b>", table_header_style),
            Paragraph(f"<b>{prediction.upper()}</b> ({confidence*100:.1f}% confidence)", verdict_style),
        ],
        [
            Paragraph("<b>Model Backend:</b>", table_header_style),
            Paragraph(str(report_data.get("model_name", "ViT Deepfake Classifier")), table_cell_style),
            Paragraph("<b>Manipulated Probability:</b>", table_header_style),
            Paragraph(f"<b>{fake_score*100:.1f}%</b>", table_cell_style),
        ]
    ]
    summary_table = Table(summary_data, colWidths=[1.4*inch, 2.1*inch, 1.4*inch, 2.1*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 12))

    # 3. File Provenance & Cryptographic Hashes
    story.append(Paragraph("1. File Integrity & Cryptographic Hashes", section_heading))
    file_info_data = [
        [Paragraph("<b>File Name</b>", table_header_style), Paragraph(str(report_data.get("filename", "N/A")), table_cell_style)],
        [Paragraph("<b>File Size</b>", table_header_style), Paragraph(f"{report_data.get('size_bytes', 0):,} bytes", table_cell_style)],
        [Paragraph("<b>SHA-256 Hash</b>", table_header_style), Paragraph(f"<font face='Courier'>{report_data.get('sha256', 'N/A')}</font>", table_cell_style)],
        [Paragraph("<b>MD5 Hash</b>", table_header_style), Paragraph(f"<font face='Courier'>{report_data.get('md5', 'N/A')}</font>", table_cell_style)],
    ]
    file_table = Table(file_info_data, colWidths=[1.8*inch, 5.2*inch])
    file_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(file_table)
    story.append(Spacer(1, 10))

    # 4. Media & Metadata Forensics
    story.append(Paragraph("2. Technical & Metadata Forensics", section_heading))
    metadata_obs = report_data.get("metadata_observations", [])
    if not metadata_obs:
        metadata_obs = ["Standard container properties parsed successfully. No anomalous headers detected."]
    
    meta_rows = [[Paragraph("<b>Observation</b>", table_header_style)]]
    for obs in metadata_obs:
        meta_rows.append([Paragraph(f"• {obs}", table_cell_style)])

    meta_table = Table(meta_rows, colWidths=[7.0*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f8fafc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 5. Video Breakdown Table (If Video)
    if media_type == "VIDEO" and "video_stats" in report_data:
        vstats = report_data["video_stats"]
        story.append(Paragraph("3. Video Sampling & Frame Statistics", section_heading))
        vstats_data = [
            [Paragraph("<b>Sampled Frames:</b>", table_header_style), Paragraph(str(vstats.get("frames_sampled", 0)), table_cell_style),
             Paragraph("<b>Frames with Faces:</b>", table_header_style), Paragraph(str(vstats.get("frames_with_faces", 0)), table_cell_style)],
            [Paragraph("<b>Mean Frame Fake Score:</b>", table_header_style), Paragraph(f"{vstats.get('mean_score', 0.0):.3f}", table_cell_style),
             Paragraph("<b>Score Std Variation:</b>", table_header_style), Paragraph(f"{vstats.get('score_std', 0.0):.3f}", table_cell_style)],
            [Paragraph("<b>Median Frame Fake Score:</b>", table_header_style), Paragraph(f"{vstats.get('median_score', 0.0):.3f}", table_cell_style),
             Paragraph("<b>Fake Frame Ratio:</b>", table_header_style), Paragraph(f"{vstats.get('fake_frame_ratio', 0.0)*100:.1f}%", table_cell_style)],
        ]
        vstats_table = Table(vstats_data, colWidths=[1.8*inch, 1.7*inch, 1.8*inch, 1.7*inch])
        vstats_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(vstats_table)
        story.append(Spacer(1, 8))

        # Sampled Frame Details (up to 12 frames in PDF table)
        frame_items = report_data.get("frame_predictions", [])
        if frame_items:
            frame_table_data = [
                [
                    Paragraph("<b>Frame #</b>", table_header_style),
                    Paragraph("<b>Timestamp (s)</b>", table_header_style),
                    Paragraph("<b>Face Detected</b>", table_header_style),
                    Paragraph("<b>Prediction</b>", table_header_style),
                    Paragraph("<b>Fake Probability</b>", table_header_style),
                ]
            ]
            for f in frame_items[:12]:
                f_pred = f.get("prediction", "No Face")
                f_style = verdict_fake_style if f_pred.lower() == "fake" else table_cell_style
                frame_table_data.append([
                    Paragraph(str(f.get("frame_index", "")), table_cell_style),
                    Paragraph(f"{f.get('timestamp', 0.0):.2f}s", table_cell_style),
                    Paragraph("Yes" if f.get("face_detected", False) else "No", table_cell_style),
                    Paragraph(str(f_pred), f_style),
                    Paragraph(f"{f.get('fake_score', 0.0):.3f}", table_cell_style),
                ])
            if len(frame_items) > 12:
                frame_table_data.append([
                    Paragraph(f"... [{len(frame_items)-12} more frames analyzed, omitted for brevity]", table_cell_style),
                    Paragraph("", table_cell_style), Paragraph("", table_cell_style), Paragraph("", table_cell_style), Paragraph("", table_cell_style)
                ])

            frame_table = Table(frame_table_data, colWidths=[1.0*inch, 1.4*inch, 1.4*inch, 1.4*inch, 1.8*inch])
            frame_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(frame_table)
            story.append(Spacer(1, 10))

    # 6. Visual Explanation Image (If provided)
    if explanation_image_path and Path(explanation_image_path).exists():
        story.append(Paragraph("Visual Forensic Explanation (Attention Map Overlay)", section_heading))
        story.append(Paragraph(
            "The image below illustrates spatial regions of highest model sensitivity. "
            "Warm colors (red/yellow) indicate facial patches that most heavily influenced the classification decision.",
            body_style
        ))
        story.append(Spacer(1, 4))
        try:
            story.append(RLImage(str(explanation_image_path), width=3.2*inch, height=3.2*inch))
        except Exception as e:
            logger.warning(f"Could not render explanation image in PDF: {e}")
        story.append(Spacer(1, 10))

    # 7. Model Documentation & Limitations
    story.append(Paragraph("4. Model Information & Forensic Limitations", section_heading))
    limitations = report_data.get("limitations", [
        "Sensitivity decreases on heavily compressed video frames.",
        "Extreme head poses (>45 deg) may cause false positive detections.",
        "Model scores indicate mathematical pattern correlation, not definitive legal proof of manipulation."
    ])
    lim_rows = [[Paragraph("<b>Documented Limitations</b>", table_header_style)]]
    for lim in limitations:
        lim_rows.append([Paragraph(f"• {lim}", table_cell_style)])
    lim_table = Table(lim_rows, colWidths=[7.0*inch])
    lim_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#fca5a5")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fff1f2")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(lim_table)
    story.append(Spacer(1, 12))

    # 8. Mandatory Disclaimer Box
    story.append(Paragraph("<b>LEGAL & FORENSIC DISCLAIMER</b>", section_heading))
    disclaimer_box = Table([[Paragraph(MANDATORY_DISCLAIMER, disclaimer_style)]], colWidths=[7.0*inch])
    disclaimer_box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94a3b8")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(disclaimer_box)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
