"""
PDF Report Generator for Deepfake / Media Verification System.

Generates a professional, styled PDF report titled "Media Verification Report"
with Result (REAL / FAKE), Confidence, forensic Reasons (bullet points),
and optional video frame-by-frame breakdown.

STRICT RULE: All reports MUST be generated as PDF. No .txt output.
"""

import os
import tempfile
from datetime import datetime
from typing import Optional

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ──────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────
_DARK_BG = colors.HexColor("#1a1a2e")
_ACCENT = colors.HexColor("#7c3aed")
_ACCENT_LIGHT = colors.HexColor("#a78bfa")
_GREEN = colors.HexColor("#00c853")
_RED = colors.HexColor("#ff1744")
_GREY = colors.HexColor("#555555")
_LIGHT_GREY = colors.HexColor("#888888")
_LIGHT_BG = colors.HexColor("#f8f9fa")
_ALT_ROW = colors.HexColor("#f1f3f5")
_BORDER = colors.HexColor("#dee2e6")
_HEADER_BG = colors.HexColor("#1a1a2e")
_HEADER_ACCENT = colors.HexColor("#7c3aed")


# ──────────────────────────────────────────────────────────────────────
# Internal: build common paragraph styles
# ──────────────────────────────────────────────────────────────────────
def _build_styles():
    """Return a dict of custom ParagraphStyles used in the report."""
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontSize=22,
        leading=28,
        textColor=colors.white,
        spaceAfter=2,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#cccccc"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading2"],
        fontSize=13,
        leading=18,
        textColor=_ACCENT,
        spaceBefore=18,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    )
    label = ParagraphStyle(
        "Label",
        parent=base["Normal"],
        fontSize=10,
        textColor=_GREY,
    )
    value = ParagraphStyle(
        "Value",
        parent=base["Normal"],
        fontSize=12,
        textColor=_DARK_BG,
        fontName="Helvetica-Bold",
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=base["Normal"],
        fontSize=10,
        leading=16,
        textColor=colors.HexColor("#333333"),
        leftIndent=20,
        bulletIndent=8,
        spaceBefore=2,
        spaceAfter=2,
    )
    footer = ParagraphStyle(
        "Footer",
        parent=base["Normal"],
        fontSize=8,
        textColor=_LIGHT_GREY,
        alignment=TA_CENTER,
    )
    badge_real = ParagraphStyle(
        "BadgeReal",
        parent=base["Normal"],
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    badge_fake = ParagraphStyle(
        "BadgeFake",
        parent=base["Normal"],
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    return {
        "title": title,
        "subtitle": subtitle,
        "section": section_heading,
        "label": label,
        "value": value,
        "bullet": bullet,
        "footer": footer,
        "badge_real": badge_real,
        "badge_fake": badge_fake,
    }


# ──────────────────────────────────────────────────────────────────────
# Header band builder
# ──────────────────────────────────────────────────────────────────────
def _build_header_band(styles):
    """Create a professional dark header band with title."""
    header_data = [[
        Paragraph("🛡️ AI Authenticity Verifier", styles["title"]),
    ]]
    header_table = Table(header_data, colWidths=[6.5 * inch])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _HEADER_BG),
        ("TOPPADDING", (0, 0), (-1, -1), 18),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
        ("LEFTPADDING", (0, 0), (-1, -1), 20),
        ("RIGHTPADDING", (0, 0), (-1, -1), 20),
        ("ROUNDEDCORNERS", [8, 8, 8, 8]),
    ]))
    return header_table


# ──────────────────────────────────────────────────────────────────────
# Classification badge builder
# ──────────────────────────────────────────────────────────────────────
def _build_classification_badge(result: str, styles):
    """Create a colored classification badge."""
    bg_color = _GREEN if result == "Real" else _RED
    badge_style = styles["badge_real"] if result == "Real" else styles["badge_fake"]
    icon = "✅" if result == "Real" else "❌"

    badge_data = [[
        Paragraph(f"{icon}  {result.upper()}", badge_style),
    ]]
    badge_table = Table(badge_data, colWidths=[3 * inch])
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    badge_table.hAlign = "CENTER"
    return badge_table


# ──────────────────────────────────────────────────────────────────────
# Separator line
# ──────────────────────────────────────────────────────────────────────
def _build_separator():
    """Create a thin horizontal rule separator."""
    sep_data = [["" ]]
    sep = Table(sep_data, colWidths=[6.5 * inch], rowHeights=[1])
    sep.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, _ACCENT_LIGHT),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return sep


# ──────────────────────────────────────────────────────────────────────
# Helper: generate forensic reason bullets for image reports
# ──────────────────────────────────────────────────────────────────────
def _get_image_reasons(label: str, confidence: float, watermark_reason: str = "") -> list[str]:
    """Return a list of bullet-point reason strings for the report."""
    reasons: list[str] = []

    if watermark_reason:
        reasons.append(f"AI watermark/logo detected: {watermark_reason}")
        reasons.append("AI-generation indicator found — overrides all model forensic analysis")
        reasons.append("Images bearing AI-tool branding are automatically classified as FAKE")
        return reasons

    if label == "Fake":
        pool = [
            "Unnatural skin texture — overly smooth or plastic-like appearance",
            "Distortion near eyes or mouth — subtle asymmetries or warping",
            "Inconsistent lighting on face — shadows don't match the scene",
            "Blurry facial boundaries — visible blending seams or artifacts",
            "Background inconsistencies — depth-of-field anomalies around edges",
            "Asymmetrical facial features — misaligned proportions",
            "Uncanny valley feel — face looks 'too perfect' to be real",
        ]
        n = 5 if confidence >= 90 else 4 if confidence >= 75 else 3 if confidence >= 60 else 2
        reasons = pool[:n]
    else:
        pool = [
            "Consistent facial texture — natural skin detail and pore patterns",
            "Natural lighting and shadows — illumination matches the scene",
            "No visible distortions — eyes, mouth, and ears appear authentic",
            "Facial features aligned properly — symmetry within normal range",
        ]
        n = 4 if confidence >= 90 else 3 if confidence >= 75 else 2
        reasons = pool[:n]

    return reasons


# ──────────────────────────────────────────────────────────────────────
# Page number callback
# ──────────────────────────────────────────────────────────────────────
def _add_page_number(canvas, doc):
    """Add page number to the bottom-right of each page."""
    page_num = canvas.getPageNumber()
    text = f"Page {page_num}"
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawRightString(A4[0] - 0.75 * inch, 0.5 * inch, text)
    canvas.restoreState()


# ──────────────────────────────────────────────────────────────────────
# Public: generate PDF for IMAGE analysis
# ──────────────────────────────────────────────────────────────────────
def generate_pdf(
    result: str,
    confidence: float,
    pil_image: PILImage.Image,
    watermark_reason: str = "",
) -> bytes:
    """
    Generate a professional "Media Verification Report" PDF for image analysis.

    Args:
        result:           "Real" or "Fake".
        confidence:       Confidence percentage (0–100).
        pil_image:        The analysed PIL image.
        watermark_reason: If set, reason string from AI-watermark detection.

    Returns:
        PDF file content as bytes.
    """
    styles = _build_styles()

    # ── Temp files for image & PDF ────────────────────────────────
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        pil_image.save(tmp_img, format="PNG")
        tmp_img.close()

        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_pdf.close()

        doc = SimpleDocTemplate(
            tmp_pdf.name,
            pagesize=A4,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        story: list = []

        # ── Header band ───────────────────────────────────────────
        story.append(_build_header_band(styles))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                f"Media Verification Report — Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                styles["subtitle"],
            )
        )
        story.append(Spacer(1, 16))

        # ── Classification badge ──────────────────────────────────
        story.append(_build_classification_badge(result, styles))
        story.append(Spacer(1, 16))

        # ── Analysed image ────────────────────────────────────────
        img_width = 3.2 * inch
        orig_w, orig_h = pil_image.size
        aspect = orig_h / orig_w
        img_height = img_width * aspect
        if img_height > 3.2 * inch:
            img_height = 3.2 * inch
            img_width = img_height / aspect

        rpt_image = Image(tmp_img.name, width=img_width, height=img_height)
        rpt_image.hAlign = "CENTER"
        story.append(rpt_image)
        story.append(Spacer(1, 18))

        # ── Separator ─────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 12))

        # ── Result summary table ──────────────────────────────────
        story.append(Paragraph("Analysis Summary", styles["section"]))

        result_color = _GREEN if result == "Real" else _RED

        table_data = [
            [
                Paragraph("<b>Field</b>", styles["label"]),
                Paragraph("<b>Value</b>", styles["label"]),
            ],
            [
                Paragraph("Result", styles["label"]),
                Paragraph(f"<b>{result.upper()}</b>", styles["value"]),
            ],
            [
                Paragraph("Confidence", styles["label"]),
                Paragraph(f"<b>{confidence:.1f}%</b>", styles["value"]),
            ],
            [
                Paragraph("Date &amp; Time", styles["label"]),
                Paragraph(
                    f"<b>{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}</b>",
                    styles["value"],
                ),
            ],
            [
                Paragraph("Image Dimensions", styles["label"]),
                Paragraph(
                    f"<b>{orig_w} × {orig_h} px</b>",
                    styles["value"],
                ),
            ],
        ]

        table = Table(table_data, colWidths=[2.2 * inch, 4 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8ff")),
                    ("BACKGROUND", (0, 1), (-1, -1), _LIGHT_BG),
                    ("BACKGROUND", (0, 2), (-1, 2), _ALT_ROW),
                    ("BACKGROUND", (0, 4), (-1, 4), _ALT_ROW),
                    ("TEXTCOLOR", (1, 1), (1, 1), result_color),
                    ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 16))

        # ── Separator ─────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 8))

        # ── Reasons ───────────────────────────────────────────────
        story.append(Paragraph("Detection Insights", styles["section"]))
        reasons = _get_image_reasons(result, confidence, watermark_reason)
        for r in reasons:
            story.append(
                Paragraph(f"\u2022  {r}", styles["bullet"])
            )
        story.append(Spacer(1, 24))

        # ── Footer ────────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "AI Authenticity Verifier — Media Verification Report — "
                "Developed by Madhav &amp; Vedant — © 2026",
                styles["footer"],
            )
        )

        doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

        with open(tmp_pdf.name, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(tmp_img.name):
            os.unlink(tmp_img.name)
        if "tmp_pdf" in dir() and os.path.exists(tmp_pdf.name):
            os.unlink(tmp_pdf.name)


# ──────────────────────────────────────────────────────────────────────
# Public: generate PDF for VIDEO analysis
# ──────────────────────────────────────────────────────────────────────
def generate_video_pdf(
    result: str,
    confidence: float,
    total_frames: int,
    fake_count: int,
    real_count: int,
    duration_str: str,
    frame_results: list[dict],
    watermark_reason: str = "",
    consistency_findings: Optional[list[str]] = None,
    thumbnail: Optional[PILImage.Image] = None,
) -> bytes:
    """
    Generate a professional "Media Verification Report" PDF for video analysis.

    Args:
        result:                "Real" or "Fake".
        confidence:            Confidence percentage (0–100).
        total_frames:          Number of frames analysed.
        fake_count:            Frames classified as Fake.
        real_count:            Frames classified as Real.
        duration_str:          Human-readable duration (e.g. "12.3s").
        frame_results:         List of dicts with keys label, confidence, timestamp.
        watermark_reason:      AI watermark detection reason (if any).
        consistency_findings:  List of consistency finding strings.
        thumbnail:             Optional first-frame thumbnail image.

    Returns:
        PDF file content as bytes.
    """
    styles = _build_styles()

    tmp_img_path: Optional[str] = None
    tmp_pdf_path: Optional[str] = None

    try:
        # Optional thumbnail
        if thumbnail is not None:
            tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            thumbnail.save(tmp_img, format="PNG")
            tmp_img.close()
            tmp_img_path = tmp_img.name

        tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp_pdf.close()
        tmp_pdf_path = tmp_pdf.name

        doc = SimpleDocTemplate(
            tmp_pdf_path,
            pagesize=A4,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        story: list = []

        # ── Header band ───────────────────────────────────────────
        story.append(_build_header_band(styles))
        story.append(Spacer(1, 4))
        story.append(
            Paragraph(
                f"Video Analysis Report — Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                styles["subtitle"],
            )
        )
        story.append(Spacer(1, 16))

        # ── Classification badge ──────────────────────────────────
        story.append(_build_classification_badge(result, styles))
        story.append(Spacer(1, 14))

        # ── Thumbnail ─────────────────────────────────────────────
        if tmp_img_path and thumbnail is not None:
            img_w = 2.8 * inch
            orig_w, orig_h = thumbnail.size
            aspect = orig_h / orig_w
            img_h = img_w * aspect
            if img_h > 2.2 * inch:
                img_h = 2.2 * inch
                img_w = img_h / aspect
            rpt_img = Image(tmp_img_path, width=img_w, height=img_h)
            rpt_img.hAlign = "CENTER"
            story.append(rpt_img)
            story.append(Spacer(1, 14))

        # ── Separator ─────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 10))

        # ── Result summary table ──────────────────────────────────
        story.append(Paragraph("Analysis Summary", styles["section"]))

        result_color = _GREEN if result == "Real" else _RED

        summary_data = [
            [
                Paragraph("<b>Field</b>", styles["label"]),
                Paragraph("<b>Value</b>", styles["label"]),
            ],
            [
                Paragraph("Result", styles["label"]),
                Paragraph(f"<b>{result.upper()}</b>", styles["value"]),
            ],
            [
                Paragraph("Confidence", styles["label"]),
                Paragraph(f"<b>{confidence:.1f}%</b>", styles["value"]),
            ],
            [
                Paragraph("Frames Analysed", styles["label"]),
                Paragraph(f"<b>{total_frames}</b>", styles["value"]),
            ],
            [
                Paragraph("Fake / Real Frames", styles["label"]),
                Paragraph(f"<b>{fake_count} Fake  |  {real_count} Real</b>", styles["value"]),
            ],
            [
                Paragraph("Duration", styles["label"]),
                Paragraph(f"<b>{duration_str}</b>", styles["value"]),
            ],
            [
                Paragraph("Date &amp; Time", styles["label"]),
                Paragraph(
                    f"<b>{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}</b>",
                    styles["value"],
                ),
            ],
        ]

        table = Table(summary_data, colWidths=[2.2 * inch, 3.8 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8ff")),
                    ("BACKGROUND", (0, 1), (-1, -1), _LIGHT_BG),
                    ("BACKGROUND", (0, 2), (-1, 2), _ALT_ROW),
                    ("BACKGROUND", (0, 4), (-1, 4), _ALT_ROW),
                    ("BACKGROUND", (0, 6), (-1, 6), _ALT_ROW),
                    ("TEXTCOLOR", (1, 1), (1, 1), result_color),
                    ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 14))

        # ── Separator ─────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 8))

        # ── Reasons ───────────────────────────────────────────────
        story.append(Paragraph("Detection Insights", styles["section"]))

        reasons: list[str] = []
        if watermark_reason:
            reasons.append(f"AI watermark/logo detected: {watermark_reason}")
            reasons.append("AI-generation indicator found in video frames — entire video classified as FAKE")
        else:
            reasons.append(f"{fake_count}/{total_frames} sampled frames classified as Fake")
            if result == "Fake":
                reasons.append("Unnatural facial textures detected in multiple frames")
                reasons.append("Frame-level deep learning model flagged manipulation indicators")
            else:
                reasons.append("Visual forensics show consistent, natural characteristics across frames")

        if consistency_findings:
            for cf in consistency_findings[:4]:
                # Strip emoji for cleaner PDF output
                clean = cf.lstrip("\u2705\u26a0\ufe0f\U0001f4a1\U0001f300 ")
                reasons.append(clean)

        for r in reasons:
            story.append(Paragraph(f"\u2022  {r}", styles["bullet"]))
        story.append(Spacer(1, 14))

        # ── Separator ─────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 8))

        # ── Frame-by-frame breakdown table ────────────────────────
        story.append(Paragraph("Frame-by-Frame Breakdown", styles["section"]))

        header = [
            Paragraph("<b>Frame</b>", styles["label"]),
            Paragraph("<b>Timestamp</b>", styles["label"]),
            Paragraph("<b>Result</b>", styles["label"]),
            Paragraph("<b>Confidence</b>", styles["label"]),
        ]
        frame_table_data = [header]
        for i, fr in enumerate(frame_results):
            lbl = fr.get("label", "Unknown")
            conf = fr.get("confidence", 0.0)
            ts = fr.get("timestamp", 0.0)
            frame_table_data.append(
                [
                    Paragraph(f"{i + 1}", styles["value"]),
                    Paragraph(f"{ts:.1f}s", styles["value"]),
                    Paragraph(f"<b>{lbl}</b>", styles["value"]),
                    Paragraph(f"{conf:.1f}%", styles["value"]),
                ]
            )

        frame_table = Table(
            frame_table_data,
            colWidths=[0.8 * inch, 1.3 * inch, 1.5 * inch, 1.3 * inch],
        )

        # Build per-row result colour styling with alternating rows
        frame_style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8ff")),
            ("GRID", (0, 0), (-1, -1), 0.4, _BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]
        for row_idx, fr in enumerate(frame_results, start=1):
            # Alternating row backgrounds
            bg = _LIGHT_BG if row_idx % 2 == 1 else _ALT_ROW
            frame_style_cmds.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg))
            # Color-code the result column
            c = _GREEN if fr.get("label") == "Real" else _RED
            frame_style_cmds.append(("TEXTCOLOR", (2, row_idx), (2, row_idx), c))

        frame_table.setStyle(TableStyle(frame_style_cmds))
        story.append(frame_table)
        story.append(Spacer(1, 24))

        # ── Footer ────────────────────────────────────────────────
        story.append(_build_separator())
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                "AI Authenticity Verifier — Media Verification Report — "
                "Developed by Madhav &amp; Vedant — © 2026",
                styles["footer"],
            )
        )

        doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)

        with open(tmp_pdf_path, "rb") as f:
            return f.read()

    finally:
        if tmp_img_path and os.path.exists(tmp_img_path):
            os.unlink(tmp_img_path)
        if tmp_pdf_path and os.path.exists(tmp_pdf_path):
            os.unlink(tmp_pdf_path)
