import io
import datetime
import os
import uuid
import re
import asyncio
import json
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import qrcode
import pypdf
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from fastapi import FastAPI
from nicegui import app, ui, run

from google import genai
from google.genai import types

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

GEMINI_MODEL = "gemini-3.5-flash-lite"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 32
USABLE_WIDTH = PAGE_WIDTH - (2 * MARGIN)

# =====================================================================================
# STYLING - MODERN & PROFESSIONAL (unchanged)
# =====================================================================================
app.native.window_args = {"resizable": True}

ui.add_head_html('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    ::-webkit-scrollbar { width: 8px !important; background: #031338 !important; }
    ::-webkit-scrollbar-thumb { background: #FF8C00 !important; border-radius: 10px; }

    html, body {
        background: radial-gradient(circle at 10% 20%, #0a1a3a, #031338) !important;
        color: #E9EDF5 !important;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        margin: 0; padding: 0;
        width: 100vw; height: 100vh;
        overflow-x: hidden;
    }

    /* Sidebar - solid dark navy */
    .sidebar-container {
        background: #0b1a3a !important;
        border-right: 2px solid rgba(255, 140, 0, 0.4) !important;
        box-shadow: 8px 0 30px rgba(0,0,0,0.6) !important;
    }
    .sidebar-container .q-field__control {
        background-color: rgba(13, 26, 53, 0.8) !important;
        border: 1px solid #2c3f6b !important;
        border-radius: 10px !important;
    }
    .sidebar-container .q-field__native,
    .sidebar-container .q-field__input,
    .sidebar-container .q-field__label {
        color: #E9EDF5 !important;
    }
    .sidebar-container .q-select .q-field__control {
        background-color: rgba(13, 26, 53, 0.8) !important;
    }

    /* Output no containers */
    .output-card {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        width: 100% !important;
        max-width: none !important;
        box-sizing: border-box;
    }

    /* Input cards - subtle glass */
    .input-card {
        background: rgba(13, 26, 53, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 140, 0, 0.2);
        border-radius: 16px;
        padding: 18px 22px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        width: 100% !important;
        max-width: none !important;
        box-sizing: border-box;
    }

    /* Buttons */
    .primary-btn, .q-btn {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #555 !important;
        font-weight: 600 !important;
        border-radius: 14px !important;
        padding: 10px 28px !important;
        letter-spacing: .4px;
        text-transform: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5) !important;
        transition: all 0.25s ease !important;
        min-height: 44px !important;
    }
    .primary-btn:hover, .q-btn:hover {
        background: linear-gradient(135deg, #2d2d2d 0%, #444444 100%) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0,0,0,0.7) !important;
        border-color: #FF8C00 !important;
    }
    .primary-btn:active, .q-btn:active {
        transform: translateY(0px) !important;
    }

    /* Upload */
    .q-uploader {
        background: rgba(13, 26, 53, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 14px !important;
        border: 2px dashed rgba(255, 140, 0, 0.5) !important;
        color: #FFFFFF !important;
        padding: 8px !important;
    }
    .q-uploader .q-uploader__header {
        background: transparent !important;
        color: #FFFFFF !important;
    }
    .q-uploader .q-uploader__header-content {
        color: #FFFFFF !important;
    }
    .q-uploader .q-uploader__file {
        background: rgba(13, 26, 53, 0.8) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
    }

    /* Inputs */
    input, select, textarea, .q-field__control {
        background-color: rgba(13, 26, 53, 0.7) !important;
        color: #FFFFFF !important;
        border: 1px solid #2c3f6b !important;
        border-radius: 10px !important;
    }
    .q-field__native, .q-field__input, .q-field__label {
        color: #E9EDF5 !important;
    }
    .q-field--highlighted .q-field__label {
        color: #FF8C00 !important;
    }

    /* Dropdown */
    .q-menu, .q-popover, .q-virtual-scroll__content {
        background: rgba(13, 26, 53, 0.95) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid #2c3f6b !important;
        border-radius: 10px !important;
    }
    .q-item {
        color: #FFFFFF !important;
        background: transparent !important;
        border-radius: 8px !important;
    }
    .q-item:hover {
        background: rgba(255, 140, 0, 0.15) !important;
        color: #FF8C00 !important;
    }

    .app-footer {
        width: 100%;
        background: rgba(13, 26, 53, 0.7);
        backdrop-filter: blur(8px);
        border-top: 2px solid rgba(255, 140, 0, 0.5);
        padding: 20px 24px;
        margin-top: 50px;
        text-align: center;
        color: #A9B6D0;
        font-size: 13px;
        box-sizing: border-box;
        border-radius: 16px 16px 0 0;
    }
    .app-footer a {
        color: #4FC3F7;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .app-footer a:hover {
        color: #FF8C00;
        text-decoration: underline;
    }

    .markdown-body {
        font-size: 14px;
        line-height: 1.7;
        color: #E9EDF5;
        background: transparent !important;
        padding: 0 !important;
    }
    .markdown-body h1, .markdown-body h2, .markdown-body h3,
    .markdown-body h4, .markdown-body h5, .markdown-body h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #FF8C00 !important;
        margin: 20px 0 10px 0 !important;
        line-height: 1.35 !important;
    }
    .markdown-body h1 { font-size: 22px !important; border-bottom: 2px solid #FF8C00; padding-bottom: 8px; }
    .markdown-body h2 { font-size: 19px !important; }
    .markdown-body h3 { font-size: 17px !important; color: #4FC3F7 !important; }
    .markdown-body h4, .markdown-body h5, .markdown-body h6 { font-size: 15px !important; color: #4FC3F7 !important; }
    .markdown-body p { margin: 10px 0 !important; }
    .markdown-body strong { color: #FFFFFF; }
    .markdown-body ul, .markdown-body ol { padding-left: 25px !important; margin: 10px 0 !important; }
    .markdown-body li { margin: 5px 0 !important; }
    .markdown-body hr { border-color: #1f3355; margin: 16px 0; }
    .markdown-body code {
        background: rgba(3, 19, 56, 0.8);
        border: 1px solid #1f3355;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 12.5px;
        color: #4FC3F7;
    }
    .markdown-body table {
        border-collapse: collapse !important;
        width: 100% !important;
        margin: 16px 0 !important;
        font-size: 13px !important;
        table-layout: auto !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    }
    .markdown-body th, .markdown-body td {
        border: 1px solid #1f3355 !important;
        padding: 10px 14px !important;
        text-align: left !important;
        word-wrap: break-word !important;
        white-space: normal !important;
    }
    .markdown-body th {
        background: linear-gradient(135deg, #1a1a1a 0%, #333333 100%) !important;
        color: #FF8C00 !important;
        font-weight: 700 !important;
    }
    .markdown-body tr:nth-child(even) td {
        background-color: rgba(10, 26, 58, 0.5);
    }
    .markdown-body tr:hover td {
        background-color: rgba(255, 140, 0, 0.08);
    }

    .stat-chip {
        background: rgba(13, 26, 53, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid #1f3355;
        border-radius: 12px;
        padding: 14px 20px;
        text-align: center;
        min-width: 140px;
        transition: all 0.3s ease;
    }
    .stat-chip:hover {
        border-color: #FF8C00;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(255,140,0,0.15);
    }
    .stat-chip .val {
        font-size: 24px;
        font-weight: 800;
        color: #FF8C00;
    }
    .stat-chip .lbl {
        font-size: 11px;
        color: #A9B6D0;
        text-transform: uppercase;
        letter-spacing: .05em;
        margin-top: 4px;
    }

    /* Tabs - modern, scrollable */
    .q-tabs {
        border-radius: 14px !important;
        overflow: hidden !important;
        background: rgba(13, 26, 53, 0.6) !important;
        backdrop-filter: blur(8px) !important;
        padding: 4px !important;
    }
    .q-tabs__content {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        scrollbar-width: thin;
        scrollbar-color: #FF8C00 transparent;
    }
    .q-tabs__content::-webkit-scrollbar {
        height: 4px;
    }
    .q-tabs__content::-webkit-scrollbar-thumb {
        background: #FF8C00;
        border-radius: 2px;
    }
    .q-tabs__content::-webkit-scrollbar-track {
        background: transparent;
    }
    .q-tab {
        color: #A9B6D0 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        border-radius: 10px !important;
        margin: 2px !important;
        padding: 8px 16px !important;
        white-space: nowrap;
        flex-shrink: 0;
    }
    .q-tab:hover {
        color: #FFFFFF !important;
        background: rgba(255, 140, 0, 0.1) !important;
    }
    .q-tab--active {
        color: #FF8C00 !important;
        background: rgba(255, 140, 0, 0.15) !important;
    }
    .q-tab__indicator {
        background: #FF8C00 !important;
        height: 3px !important;
        border-radius: 2px !important;
    }

    .chat-message {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .chat-message:last-child {
        border-bottom: none;
    }
    .chat-message .role-label {
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .chat-message .role-label.assistant {
        color: #FF8C00;
    }
    .chat-message .role-label.user {
        color: #4FC3F7;
    }
    .chat-message .content {
        padding-left: 8px;
    }

    /* Responsive */
    @media (max-width: 768px) {
        .markdown-body table {
            font-size: 11px !important;
        }
        .markdown-body th, .markdown-body td {
            padding: 6px 8px !important;
        }
        .stat-chip {
            min-width: 100px !important;
            padding: 10px 14px !important;
        }
        .stat-chip .val {
            font-size: 18px !important;
        }
        .input-card {
            padding: 12px 14px !important;
        }
        .primary-btn, .q-btn {
            padding: 8px 16px !important;
            font-size: 13px !important;
            min-height: 36px !important;
            border-radius: 10px !important;
        }
        .app-footer {
            font-size: 11px !important;
            padding: 14px 12px !important;
        }
        .q-tabs__content {
            flex-wrap: nowrap !important;
        }
        .q-tab {
            font-size: 12px !important;
            padding: 6px 10px !important;
        }
        .q-uploader {
            font-size: 12px !important;
        }
        .markdown-body {
            overflow-x: auto;
        }
        .markdown-body table {
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        .markdown-body table td, .markdown-body table th {
            white-space: normal !important;
        }
        .markdown-body {
            overflow-x: auto;
        }
    }

    .main-title {
        font-size: 3.8rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: #FFFFFF !important;
        font-weight: 500;
    }
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2rem !important;
        }
        .sub-title {
            font-size: 1rem !important;
        }
    }
</style>
''', shared=True)

# =====================================================================================
# TEXT SANITIZATION (unchanged)
# =====================================================================================
_LATEX_SIMPLE = {
    r'\times': ' x ', r'\cdot': ' . ', r'\div': ' / ',
    r'\geq': ' >= ', r'\ge': ' >= ', r'\leq': ' <= ', r'\le': ' <= ',
    r'\pm': ' +/- ', r'\approx': ' ~= ', r'\neq': ' != ',
    r'\infty': 'infinity', r'\text': '', r'\mathrm': '', r'\mathbf': '',
    r'\left': '', r'\right': '', r'\,': ' ', r'\;': ' ', r'\!': '',
    r'\Delta': 'Delta ', r'\delta': 'delta ', r'\sigma': 'sigma ', r'\Sigma': 'Sigma ',
    r'\phi': 'phi ', r'\gamma': 'gamma ', r'\theta': 'theta ', r'\mu': 'mu ',
    r'\pi': 'pi ', r'\alpha': 'alpha ', r'\beta': 'beta ', r'\rho': 'rho ',
    r'\max': 'Max', r'\min': 'Min', r'\sum': 'Sum', r'\bar': '',
}


def sanitize_ai_markdown(text: str) -> str:
    if not text:
        return ""
    text = str(text)

    for macro, repl in _LATEX_SIMPLE.items():
        text = text.replace(macro, repl)

    for _ in range(2):
        text = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'(\1 / \2)', text)
        text = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', text)

    text = re.sub(r'_\{([^{}]*)\}', r'_\1', text)
    text = re.sub(r'\^\{([^{}]*)\}', r'^\1', text)

    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)

    text = text.replace('$$', '').replace('$', '')
    text = re.sub(r'(?<!\w)\{([^{}]{0,40})\}(?!\w)', r'\1', text)

    text = re.sub(r'\*{3,}', '**', text)

    text = re.sub(r'([^\n])\n(#{1,6}\s)', r'\1\n\n\2', text)
    text = re.sub(r'([^\n|])\n(\|)', r'\1\n\n\2', text)

    text = re.sub(r'(?<![\w#*`|])[&$%^~?/\\]{2,}(?![\w#*`|])', '', text)

    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def inline_md_to_reportlab(text: str) -> str:
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', text)
    return text


def build_pdf_styles():
    base = getSampleStyleSheet()
    return {
        'h1': ParagraphStyle('PdfH1', parent=base['Heading1'], fontSize=13.5, leading=17,
                              textColor=colors.HexColor('#1B2A4A'), spaceBefore=10, spaceAfter=5,
                              fontName='Helvetica-Bold'),
        'h2': ParagraphStyle('PdfH2', parent=base['Heading2'], fontSize=11.5, leading=15,
                              textColor=colors.HexColor('#B45309'), spaceBefore=8, spaceAfter=4,
                              fontName='Helvetica-Bold'),
        'h3': ParagraphStyle('PdfH3', parent=base['Heading3'], fontSize=10.5, leading=14,
                              textColor=colors.HexColor('#1B2A4A'), spaceBefore=6, spaceAfter=3,
                              fontName='Helvetica-Bold'),
        'body': ParagraphStyle('PdfBody', parent=base['Normal'], fontSize=9.5, leading=13.5,
                                textColor=colors.HexColor('#1E293B'), spaceAfter=4, fontName='Helvetica'),
        'bullet': ParagraphStyle('PdfBullet', parent=base['Normal'], fontSize=9.5, leading=13,
                                  leftIndent=12, textColor=colors.HexColor('#1E293B'), spaceAfter=2),
        'tablecell': ParagraphStyle('PdfCell', parent=base['Normal'], fontSize=8.5, leading=11,
                                     textColor=colors.HexColor('#1E293B')),
        'tablehead': ParagraphStyle('PdfCellHead', parent=base['Normal'], fontSize=8.5, leading=11,
                                     textColor=colors.white, fontName='Helvetica-Bold'),
    }


def markdown_to_pdf_flowables(raw_text: str, styles: dict, avail_width: float = USABLE_WIDTH):
    text = sanitize_ai_markdown(raw_text)
    lines = text.split('\n')
    flowables = []
    para_buffer = []
    i, n = 0, len(lines)

    def flush_para():
        if para_buffer:
            joined = ' '.join(l.strip() for l in para_buffer if l.strip())
            if joined:
                flowables.append(Paragraph(inline_md_to_reportlab(joined), styles['body']))
            para_buffer.clear()

    while i < n:
        raw_line = lines[i]
        stripped = raw_line.strip()

        if not stripped:
            flush_para()
            i += 1
            continue

        h_match = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if h_match:
            flush_para()
            level = len(h_match.group(1))
            content = h_match.group(2).strip('* ').strip()
            key = 'h1' if level <= 2 else ('h2' if level == 3 else 'h3')
            flowables.append(Paragraph(inline_md_to_reportlab(content), styles[key]))
            i += 1
            continue

        if stripped.startswith('|'):
            flush_para()
            table_lines = []
            while i < n and lines[i].strip().startswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                if re.match(r'^\|?[\s:|-]+\|?$', tl):
                    continue
                cells = [c.strip() for c in tl.strip('|').split('|')]
                rows.append(cells)
            if rows:
                ncols = max(len(r) for r in rows)
                rows = [r + [''] * (ncols - len(r)) for r in rows]
                table_data = []
                for ridx, row in enumerate(rows):
                    style_key = 'tablehead' if ridx == 0 else 'tablecell'
                    table_data.append([Paragraph(inline_md_to_reportlab(c), styles[style_key]) for c in row])
                colw = avail_width / ncols
                t = Table(table_data, colWidths=[colw] * ncols, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B2A4A')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                flowables.append(t)
                flowables.append(Spacer(1, 6))
            continue

        b_match = re.match(r'^[-*•]\s+(.*)', stripped)
        n_match = re.match(r'^(\d+)[.)]\s+(.*)', stripped)
        if b_match or n_match:
            flush_para()
            while i < n:
                s2 = lines[i].strip()
                bm = re.match(r'^[-*•]\s+(.*)', s2)
                nm = re.match(r'^(\d+)[.)]\s+(.*)', s2)
                if bm:
                    flowables.append(Paragraph(f"&#8226; {inline_md_to_reportlab(bm.group(1))}", styles['bullet']))
                    i += 1
                elif nm:
                    flowables.append(Paragraph(f"{nm.group(1)}. {inline_md_to_reportlab(nm.group(2))}", styles['bullet']))
                    i += 1
                else:
                    break
            flowables.append(Spacer(1, 4))
            continue

        para_buffer.append(stripped)
        i += 1

    flush_para()
    return flowables


# =====================================================================================
# PDF / EXPORT HELPERS (modified for OCR)
# =====================================================================================
def generate_qr_code(data_str):
    qr = qrcode.QRCode(version=1, box_size=5, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def build_pdf_header(story, styles, doc_title, subtitle, logo_bytes, engineer, project, location, rep_date, ticket_id, unique_hash, show_ticket=True):
    title_style = ParagraphStyle("DocTitle", fontSize=14, textColor=colors.HexColor("#1B2A4A"),
                                  spaceAfter=3, fontName="Helvetica-Bold", leading=17)
    sub_style = ParagraphStyle("DocSub", fontSize=9, textColor=colors.HexColor("#B45309"),
                                spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", fontSize=8, textColor=colors.HexColor("#334155"),
                                 leading=11.5, fontName="Helvetica")

    ticket_part = f"&nbsp;|&nbsp; <b>Batch Ticket ID:</b> {ticket_id}" if show_ticket else ""
    meta_html = f"""
    <b>Project:</b> {project} &nbsp;|&nbsp; <b>Location:</b> {location}<br/>
    <b>Engineer in Charge:</b> {engineer} &nbsp;|&nbsp; <b>Date:</b> {rep_date}{ticket_part}<br/>
    <b>Verification UID:</b> <font color="#CC0000"><b>{unique_hash}</b></font>
    """

    right_cell = ReportLabImage(io.BytesIO(logo_bytes), width=70, height=32) if logo_bytes else ""
    try:
        header_table_data = [
            [Paragraph(f"<b>{doc_title}</b>", title_style), right_cell],
            [Paragraph(subtitle, sub_style), ""],
            [Paragraph(meta_html, meta_style), ""],
        ]
        t_head = Table(header_table_data, colWidths=[USABLE_WIDTH - 100, 100])
        t_head.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_head)
    except Exception:
        story.append(Paragraph(doc_title, title_style))
        story.append(Paragraph(subtitle, sub_style))
        story.append(Paragraph(meta_html, meta_style))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor("#FF8C00"), spaceAfter=8))


def build_pdf_footer_signature_and_qr(story, styles, qr_img_buffer, engineer_name):
    """Custom footer: only Prepared by Engineer with signature line, QR on the right."""
    body_style = ParagraphStyle("SigBody", fontSize=8, textColor=colors.HexColor("#334155"), leading=11)
    # Prepare QR image
    qr_lab_img = ReportLabImage(qr_img_buffer, width=38, height=38)
    # Signature cell: "Prepared by Engineer:" and a blank line for signature
    sign_text = f"<b>Prepared by Engineer:</b><br/>{engineer_name}<br/><br/>_________________<br/>(Signature &amp; Date)"
    sign_cell = Paragraph(sign_text, body_style)

    # Table with two cells: signature on left, QR on right
    w = USABLE_WIDTH
    t = Table([[sign_cell, qr_lab_img]], colWidths=[w * 0.7, w * 0.3])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
    ]))
    story.append(t)


def build_report_pdf(doc_title, subtitle, body_markdown, meta, logo_bytes, extra_flowables_before_body=None, show_ticket=True):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN)
    styles = build_pdf_styles()
    story = []
    unique_uid = meta['uid']
    qr_buf = generate_qr_code(f"UID: {unique_uid} | {doc_title} - {meta['project']}")

    build_pdf_header(story, styles, doc_title, subtitle, logo_bytes, meta['engineer'],
                      meta['project'], meta['location'], meta['date'], meta['ticket'], unique_uid, show_ticket)

    if extra_flowables_before_body:
        story.extend(extra_flowables_before_body)
        story.append(Spacer(1, 6))

    story.extend(markdown_to_pdf_flowables(body_markdown, styles))
    story.append(Spacer(1, 8))

    # Custom footer with only one signature + QR
    build_pdf_footer_signature_and_qr(story, styles, qr_buf, meta['engineer'])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =====================================================================================
# CODE-COMPLIANCE DIRECTIVE (unchanged)
# =====================================================================================
CODE_BASIS_OPTIONS = [
    "Egyptian Codes: ECP 203 / ECP 202 / ECP 104 (Default Core Basis)",
    "ACI 318-25 — Structural Concrete (Primary)",
    "Eurocode 2 — BS EN 1992 + UK Annex (Primary)",
    "AASHTO LRFD Bridge & Pavement Design (Primary)",
    "IBC — International Building Code (Primary)",
]


def get_code_directive(basis: str) -> str:
    if not basis or basis.startswith("Egyptian Codes"):
        return (
            "GOVERNING STANDARD (MANDATORY): Base every clause reference, formula, allowable limit and "
            "pass/fail verdict strictly on the Egyptian Codes — ECP 203 (Reinforced Concrete Structures), "
            "ECP 202 (Soil Mechanics & Foundations) and ECP 104 (Roads, Highways & Airfields), as applicable "
            "to the topic. Do NOT substitute ACI, Eurocode or AASHTO limits. Where relevant, cite the specific "
            "ECP clause, table or article number."
        )
    return (
        f"GOVERNING STANDARD (MANDATORY): The user has selected an alternative primary design basis: "
        f"\"{basis}\". Use that standard as the PRIMARY source for formulas, limits, and clause citations. "
        "Mention the equivalent Egyptian Code (ECP 203 / 202 / 104) clause only as a secondary cross-reference."
    )


NO_LATEX_RULE = (
    "OUTPUT FORMAT (MANDATORY): Write in clean GitHub-flavoured Markdown only. "
    "Never use LaTeX, dollar-sign math delimiters ($ or $$), backslash commands (\\frac, \\times, \\ge ...), "
    "or curly-brace variable syntax. Write formulas in plain readable text, e.g. 'f_cu = 30 N/mm2', "
    "'Standard Deviation = 2.1 N/mm2'. Use real Markdown tables (with a header row and a --- separator row) "
    "for any tabular data — never hand-draw tables with dashes or asterisks. Use ## / ### for section headings, "
    "never #### or deeper. Use single asterisks pairs (**bold**) and never stack more than two."
)


async def call_gemini(contents, system_instruction=None, temperature=0.1, timeout=240):
    cfg_kwargs = {"temperature": temperature}
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    config = types.GenerateContentConfig(**cfg_kwargs)
    try:
        response = await asyncio.wait_for(
            run.io_bound(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            ),
            timeout=timeout
        )
        return sanitize_ai_markdown(response.text)
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds. Please try with a smaller file or simplify your query.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")

async def call_gemini_json(contents, temperature=0.1, timeout=240):
    """Call Gemini and return raw text without sanitization (for JSON)."""
    cfg_kwargs = {"temperature": temperature}
    config = types.GenerateContentConfig(**cfg_kwargs)
    try:
        response = await asyncio.wait_for(
            run.io_bound(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            ),
            timeout=timeout
        )
        return response.text  # raw text, no sanitization
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")


# =====================================================================================
# JOB SCRAPING FUNCTIONS (FIXED)
# =====================================================================================
def scrape_wuzzuf(query: str) -> list:
    """Scrape Wuzzuf job search results with fallback selectors."""
    url = f"https://wuzzuf.net/search/jobs/?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        jobs = []

        # Try multiple selectors to find job cards
        selectors = [
            'div.job-card',
            'div[class*="job-card"]',
            'div[class*="css-"]',  # Wuzzuf uses these
            'div[data-testid="job-card"]',
            'div[class*="job"]',
        ]
        cards = []
        for sel in selectors:
            cards = soup.select(sel)
            if cards:
                break
        if not cards:
            # Fallback: find all <a> with href containing '/jobs/' and take their parent div
            job_links = soup.find_all('a', href=re.compile(r'/jobs/'))
            seen = set()
            for link in job_links:
                parent = link.find_parent('div')
                if parent and parent not in seen:
                    cards.append(parent)
                    seen.add(parent)

        for card in cards:
            try:
                # Title & link
                title_elem = card.find('h2') or card.find('a', class_=re.compile(r'job.*title', re.I)) or card.find('a', href=re.compile(r'/jobs/'))
                if not title_elem:
                    continue
                title = title_elem.text.strip()
                link = title_elem.get('href')
                if link and not link.startswith('http'):
                    link = 'https://wuzzuf.net' + link

                # Company
                company_elem = card.find('div', class_=re.compile(r'company', re.I)) or card.find('a', class_=re.compile(r'company', re.I))
                company = company_elem.text.strip() if company_elem else ''

                # Location
                location_elem = card.find('span', class_=re.compile(r'location', re.I)) or card.find('div', class_=re.compile(r'location', re.I))
                location = location_elem.text.strip() if location_elem else ''

                # Description (excerpt)
                desc_elem = card.find('div', class_=re.compile(r'description', re.I)) or card.find('p', class_=re.compile(r'description', re.I))
                description = desc_elem.text.strip() if desc_elem else ''

                if title and link:
                    jobs.append({
                        'title': title,
                        'company': company,
                        'location': location,
                        'description': description,
                        'url': link
                    })
            except Exception:
                continue
        return jobs
    except Exception:
        return []


def scrape_bayt(query: str) -> list:
    """Scrape Bayt job search results as a fallback."""
    url = f"https://www.bayt.com/en/egypt/jobs/?search={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        jobs = []
        # Bayt uses <li class="has-pointer"> or <div class="job-card">
        cards = soup.find_all('li', class_='has-pointer') or soup.find_all('div', class_='job-card')
        for card in cards:
            try:
                title_elem = card.find('h2') or card.find('a', class_='job-title')
                if not title_elem:
                    continue
                title = title_elem.text.strip()
                link = title_elem.get('href')
                if link and not link.startswith('http'):
                    link = 'https://www.bayt.com' + link

                company_elem = card.find('span', class_='company-name') or card.find('a', class_='company')
                company = company_elem.text.strip() if company_elem else ''

                location_elem = card.find('span', class_='location') or card.find('div', class_='location')
                location = location_elem.text.strip() if location_elem else ''

                desc_elem = card.find('div', class_='description') or card.find('p', class_='description')
                description = desc_elem.text.strip() if desc_elem else ''

                if title and link:
                    jobs.append({
                        'title': title,
                        'company': company,
                        'location': location,
                        'description': description,
                        'url': link
                    })
            except Exception:
                continue
        return jobs
    except Exception:
        return []


def scrape_jobs(query: str) -> list:
    """Try Wuzzuf first; if empty, fallback to Bayt."""
    jobs = scrape_wuzzuf(query)
    if not jobs:
        time.sleep(1)  # avoid rapid requests
        jobs = scrape_bayt(query)
    return jobs


# =====================================================================================
# BOQ CALCULATION ENGINE - AI EXTRACTION (unchanged)
# =====================================================================================

# Global storage for BOQ results per branch, element, and mode
boq_results = {
    'architectural': {},
    'structural': {}
}

UNIT_RATES = {
    "Flooring (Ceramic)": 150,
    "Flooring (Marble)": 500,
    "Flooring (Tiles)": 200,
    "Wall Finishing (Paint)": 30,
    "Wall Finishing (Plaster)": 80,
    "Ceiling (Paint)": 25,
    "Ceiling (Gypsum Board)": 120,
    "Skirting (Ceramic)": 60,
    "Skirting (Marble)": 200,
    "Doors (Wood)": 3000,
    "Windows (Aluminum)": 2000,
    "Partitions (Gypsum)": 150,
    "Concrete (C30/37)": 2500,
    "Concrete (C25/30)": 2200,
    "Concrete (C40/50)": 3000,
    "Rebar (Grade 400)": 15000,
    "Rebar (Grade 600)": 18000,
    "Formwork": 300,
    "Excavation": 200,
    "Backfill": 150,
    "Foundation Concrete": 2800,
}

# Field name to user-friendly label mapping
FIELD_LABELS = {
    'width_mm': 'Width (mm)',
    'depth_mm': 'Depth (mm)',
    'height_mm': 'Height (mm)',
    'length_mm': 'Length (mm)',
    'thickness_mm': 'Thickness (mm)',
    'area_m2': 'Area (m²)',
    'length_m': 'Length (m)',
    'height_m': 'Height (m)',
    'count': 'Count (number of columns/beams)',
    'main_diameter_mm': 'Main Bar Diameter (mm)',
    'stirrup_diameter_mm': 'Stirrup Diameter (mm)',
    'spacing_mm': 'Spacing (mm)',
    'top_diameter_mm': 'Top Bar Diameter (mm)',
    'bottom_diameter_mm': 'Bottom Bar Diameter (mm)',
}

# ---- Element-specific schemas for mass extraction ----
MASS_SCHEMAS = {
    'columns': {
        'required': ['label', 'count', 'width_mm', 'depth_mm', 'height_mm'],
        'field_aliases': {
            'width': 'width_mm',
            'depth': 'depth_mm',
            'height': 'height_mm',
            'width_mm': 'width_mm',
            'depth_mm': 'depth_mm',
            'height_mm': 'height_mm',
            'count': 'count',
        }
    },
    'beams': {
        'required': ['label', 'count', 'width_mm', 'depth_mm', 'length_mm'],
        'field_aliases': {
            'width': 'width_mm',
            'depth': 'depth_mm',
            'length': 'length_mm',
            'width_mm': 'width_mm',
            'depth_mm': 'depth_mm',
            'length_mm': 'length_mm',
            'count': 'count',
        }
    },
    'slabs': {
        'required': ['label', 'thickness_mm', 'area_m2'],
        'field_aliases': {
            'thickness': 'thickness_mm',
            'area': 'area_m2',
            'thickness_mm': 'thickness_mm',
            'area_m2': 'area_m2',
        }
    },
    'footings': {
        'required': ['label', 'count', 'width_mm', 'depth_mm', 'length_mm'],
        'field_aliases': {
            'width': 'width_mm',
            'depth': 'depth_mm',
            'length': 'length_mm',
            'width_mm': 'width_mm',
            'depth_mm': 'depth_mm',
            'length_mm': 'length_mm',
            'count': 'count',
        }
    },
    'walls': {
        'required': ['label', 'count', 'length_m', 'height_m', 'thickness_mm'],
        'field_aliases': {
            'length': 'length_m',
            'height': 'height_m',
            'thickness': 'thickness_mm',
            'length_m': 'length_m',
            'height_m': 'height_m',
            'thickness_mm': 'thickness_mm',
            'count': 'count',
        }
    }
}

# ---- Architectural schemas (NEW) ----
ARCH_SCHEMAS = {
    'flooring': {
        'required': ['total_length_m', 'total_width_m', 'area_m2'],
        'field_aliases': {
            'length': 'total_length_m',
            'width': 'total_width_m',
            'area': 'area_m2',
            'total_length_m': 'total_length_m',
            'total_width_m': 'total_width_m',
            'area_m2': 'area_m2',
        },
        'formula': lambda data: data.get('area_m2') if data.get('area_m2') else (data.get('total_length_m', 0) * data.get('total_width_m', 0))
    },
    'wall_finishing': {
        'required': ['total_area_m2'],
        'field_aliases': {
            'area': 'total_area_m2',
            'total_area_m2': 'total_area_m2',
        },
        'formula': lambda data: data.get('total_area_m2', 0)
    },
    'ceilings': {
        'required': ['total_area_m2'],
        'field_aliases': {
            'area': 'total_area_m2',
            'total_area_m2': 'total_area_m2',
        },
        'formula': lambda data: data.get('total_area_m2', 0)
    },
    'doors_windows': {
        'required': ['door_count', 'window_count'],
        'field_aliases': {
            'doors': 'door_count',
            'windows': 'window_count',
            'door_count': 'door_count',
            'window_count': 'window_count',
        },
        'formula': lambda data: (data.get('door_count', 0), data.get('window_count', 0))
    }
}

def normalize_keys(obj, aliases):
    """Convert dictionary keys using alias mapping."""
    new_obj = {}
    for k, v in obj.items():
        if k in aliases:
            new_obj[aliases[k]] = v
        else:
            new_obj[k] = v
    return new_obj

async def extract_architectural_with_ai(element_type, file_bytes, file_type, user_params, code_basis, retry=True):
    """Extract architectural quantities (area, dimensions, counts) using AI."""
    contents = []
    schema_info = ARCH_SCHEMAS.get(element_type)
    if not schema_info:
        raise ValueError(f"Unsupported architectural element: {element_type}")

    # Build a detailed prompt depending on the element
    if element_type == 'flooring':
        prompt = f"""
You are a Quantity Surveyor. Extract the building dimensions from the architectural plan.
From the drawing, determine:
- total_length_m: the overall length of the building in meters
- total_width_m: the overall width of the building in meters
- area_m2: the total floor area in square meters (if not given, compute from length × width)

If a dimension is not clearly visible, set it to null.
Return ONLY a JSON object with these fields, no extra text.

Example:
{{"total_length_m": 20.0, "total_width_m": 15.0, "area_m2": 300.0}}
"""
    elif element_type == 'wall_finishing':
        prompt = f"""
You are a Quantity Surveyor. Extract the total wall finishing area from the architectural plan.
Determine the total area of walls that need finishing (paint, plaster, etc.) in square meters.
This is often given as a total wall area or can be computed from perimeter and height.
Return ONLY a JSON object with field "total_area_m2", no extra text.
Example: {{"total_area_m2": 250.0}}
"""
    elif element_type == 'ceilings':
        prompt = f"""
You are a Quantity Surveyor. Extract the total ceiling area from the architectural plan.
This is usually the same as the floor area (or given separately).
Return ONLY a JSON object with field "total_area_m2", no extra text.
Example: {{"total_area_m2": 300.0}}
"""
    elif element_type == 'doors_windows':
        prompt = f"""
You are a Quantity Surveyor. Count the number of doors and windows from the architectural plan.
Return ONLY a JSON object with fields "door_count" and "window_count", no extra text.
Example: {{"door_count": 10, "window_count": 15}}
"""
    else:
        raise ValueError(f"Unsupported architectural element: {element_type}")

    contents.append(prompt)

    # Process file – send first page as PNG
    if file_type == 'application/pdf':
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            doc.close()
        except Exception:
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        img_part = types.Part.from_bytes(data=file_bytes, mime_type=file_type)
        contents.append(img_part)

    raw_response = None
    try:
        response_text = await call_gemini_json(contents, temperature=0, timeout=240)
        raw_response = response_text
        json_str = response_text.strip()
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        data = json.loads(json_str)
        # Normalize keys
        aliases = schema_info.get('field_aliases', {})
        norm_data = normalize_keys(data, aliases)
        return norm_data, raw_response
    except Exception as e:
        if retry:
            # Simpler prompt without images
            prompt2 = f"""
Return a JSON object with the fields: {', '.join(schema_info['required'])}.
If unclear, set values to null.
"""
            contents2 = [prompt2]
            try:
                response_text2 = await call_gemini_json(contents2, temperature=0, timeout=240)
                raw_response = response_text2
                json_str2 = response_text2.strip()
                json_str2 = re.sub(r'^```json\s*', '', json_str2)
                json_str2 = re.sub(r'\s*```$', '', json_str2)
                start = json_str2.find('{')
                end = json_str2.rfind('}')
                if start != -1 and end != -1:
                    json_str2 = json_str2[start:end+1]
                data2 = json.loads(json_str2)
                aliases = schema_info.get('field_aliases', {})
                norm_data2 = normalize_keys(data2, aliases)
                return norm_data2, raw_response
            except:
                return {}, raw_response
        else:
            return {}, raw_response

def compute_architectural_quantities(element_type, data, user_params):
    """Compute quantities from architectural AI data."""
    schema_info = ARCH_SCHEMAS.get(element_type)
    if not schema_info:
        return [], 0, []

    required = schema_info['required']
    results = []
    total_quantity = 0
    missing_fields = []

    # Check for missing required fields
    for req in required:
        if req not in data or data[req] is None:
            missing_fields.append(req)

    if missing_fields:
        return results, total_quantity, [{'label': 'General', 'idx': 0, 'missing': missing_fields}]

    # Compute quantity using formula
    if element_type in ['flooring', 'wall_finishing', 'ceilings']:
        qty = schema_info['formula'](data) if callable(schema_info['formula']) else 0
        total_quantity += qty
        results.append({
            'label': element_type.capitalize(),
            'quantity': qty,
            'unit': 'm²'
        })
    elif element_type == 'doors_windows':
        door_count, window_count = schema_info['formula'](data)
        if door_count:
            results.append({
                'label': 'Doors',
                'quantity': door_count,
                'unit': 'nos'
            })
            total_quantity += door_count
        if window_count:
            results.append({
                'label': 'Windows',
                'quantity': window_count,
                'unit': 'nos'
            })
            total_quantity += window_count

    return results, total_quantity, []

def generate_arch_boq_table(results, element_type, wastage):
    """Generate BOQ table for architectural items."""
    rows = []
    for r in results:
        rows.append({
            'Item': f"{element_type.capitalize()} - {r['label']}",
            'Count': 1,  # For architectural, count is the item itself
            'Unit': r['unit'],
            'Quantity (net)': round(r['quantity'], 2),
            'Wastage %': wastage,
            'Quantity (with waste)': round(r['quantity'] * (1 + wastage/100), 2),
            'Unit Rate (EGP)': round(UNIT_RATES.get(r['label'], 0), 2),
            'Total Cost (EGP)': round(r['quantity'] * (1 + wastage/100) * UNIT_RATES.get(r['label'], 0), 2)
        })
    # Add total row
    if rows:
        total_row = {
            'Item': 'TOTAL',
            'Count': '',
            'Unit': '',
            'Quantity (net)': round(sum(r['Quantity (net)'] for r in rows), 2),
            'Wastage %': '',
            'Quantity (with waste)': round(sum(r['Quantity (with waste)'] for r in rows), 2),
            'Unit Rate (EGP)': '',
            'Total Cost (EGP)': round(sum(r['Total Cost (EGP)'] for r in rows), 2)
        }
        rows.append(total_row)
    return pd.DataFrame(rows)


# ---- Structural mass extraction (improved) ----
async def extract_mass_with_ai(element_type, file_bytes, file_type, user_params, code_basis, retry=True):
    """Extract mass quantities using AI with a simple JSON array.
       For columns: only count columns inside the structural grid, ignore schedule/detail sheets.
    """
    contents = []
    schema_info = MASS_SCHEMAS.get(element_type)
    if not schema_info:
        raise ValueError(f"Unsupported element type: {element_type}")

    # Build a prompt with extra instruction for columns
    extra_instruction = ""
    if element_type == 'columns':
        extra_instruction = " IMPORTANT: Only count columns that are part of the structural grid/plan. Ignore any columns shown in a separate schedule, detail sheet, or table. "

    prompt = f"""
You are an expert Quantity Surveyor. Your task is to EXTRACT raw data from the provided drawing(s) and return ONLY a JSON array of objects.

Extract the following fields for each group:
{', '.join(schema_info['required'])}

If a dimension is not clearly visible, set it to null.
{extra_instruction}
Return ONLY the JSON array, no extra text, no explanations, no markdown.

Example for columns:
[{{"label":"C1","count":6,"width_mm":300,"depth_mm":300,"height_mm":3000}}]

Now extract from the drawing.
"""
    contents.append(prompt)

    # Process file – send high-quality image
    if file_type == 'application/pdf':
        try:
            # Extract text from first 3 pages for context
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i in range(min(3, len(reader.pages))):
                try:
                    txt = reader.pages[i].extract_text() or ""
                    pages_text.append(txt)
                except:
                    pass
            full_text = "".join(pages_text)
            if full_text.strip():
                contents.append(f"Extracted text from PDF:\n{full_text[:6000]}")
            # Send first page as high-quality PNG
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)  # higher resolution
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            doc.close()
        except Exception as e:
            # Fallback: send full PDF as binary
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        # Image – we'll send as is (PNG or JPEG)
        img_part = types.Part.from_bytes(data=file_bytes, mime_type=file_type)
        contents.append(img_part)

    try:
        response_text = await call_gemini_json(contents, temperature=0, timeout=240)
        # Try to extract JSON array
        json_str = response_text.strip()
        # Remove markdown fences if present
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        # Find the first '[' and last ']'
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        data = json.loads(json_str)
        return data
    except Exception as e:
        if retry:
            # Simplified retry: prompt without images, only text
            prompt2 = f"""
Return a JSON array of objects with fields: {', '.join(schema_info['required'])}.
{extra_instruction}
If the drawing is unclear, return an empty array [].
"""
            contents2 = [prompt2]
            # Try to extract text again
            if file_type == 'application/pdf':
                try:
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    txt = "".join([p.extract_text() or "" for p in reader.pages[:3]])
                    if txt.strip():
                        contents2.append(f"Extracted text from PDF:\n{txt[:6000]}")
                except:
                    pass
            try:
                response_text2 = await call_gemini_json(contents2, temperature=0, timeout=240)
                json_str2 = response_text2.strip()
                json_str2 = re.sub(r'^```json\s*', '', json_str2)
                json_str2 = re.sub(r'\s*```$', '', json_str2)
                start = json_str2.find('[')
                end = json_str2.rfind(']')
                if start != -1 and end != -1:
                    json_str2 = json_str2[start:end+1]
                data2 = json.loads(json_str2)
                return data2
            except:
                # If still fails, return empty array to trigger manual fallback
                return []
        else:
            return []

def compute_mass_from_ai_data(element_type, data, user_params):
    """Compute quantities from AI-extracted data."""
    schema_info = MASS_SCHEMAS.get(element_type)
    required = schema_info['required']
    results = []
    total_concrete = 0
    floor_height = user_params.get('floor_height_mm', 3000) / 1000

    # Data is a list of groups
    for group in data:
        # Check if all required fields are present (not null)
        all_present = True
        for req in required:
            if req not in group or group[req] is None:
                all_present = False
                break
        if not all_present:
            continue  # skip incomplete groups

        # Compute volume based on element type
        if element_type == 'columns':
            # Use floor height if height is None and allowed
            if group.get('height_mm') is None and user_params.get('use_floor_height', False):
                group['height_mm'] = user_params.get('floor_height_mm', 3000)
            if group.get('height_mm') is None:
                continue
            vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (group['height_mm']/1000) * group.get('count', 1)
        elif element_type == 'beams':
            vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (group['length_mm']/1000) * group.get('count', 1)
        elif element_type == 'slabs':
            vol = group['area_m2'] * (group['thickness_mm']/1000)
        elif element_type == 'footings':
            vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (group['length_mm']/1000) * group.get('count', 1)
        elif element_type == 'walls':
            vol = group['length_m'] * group['height_m'] * (group['thickness_mm']/1000) * group.get('count', 1)
        else:
            vol = 0

        total_concrete += vol
        results.append({
            'label': group.get('label', 'Unknown'),
            'count': group.get('count', 1),
            'concrete_m3': vol,
            'rebar_ton': 0
        })

    total_concrete = round(total_concrete, 2)
    return results, total_concrete, 0


def generate_boq_table(results, branch, element_type, wastage, mode):
    """Create a Pandas DataFrame for display and export."""
    rows = []
    if mode == 'mass':
        for r in results:
            if 'concrete_m3' in r:
                rows.append({
                    'Item': f"{element_type.capitalize()} - {r.get('label', '')}",
                    'Count': r.get('count', 1),
                    'Unit': 'm³',
                    'Quantity (net)': round(r['concrete_m3'], 2),
                    'Wastage %': wastage,
                    'Quantity (with waste)': round(r['concrete_m3'] * (1 + wastage/100), 2),
                    'Unit Rate (EGP)': round(UNIT_RATES.get('Concrete (C30/37)', 2500), 2),
                    'Total Cost (EGP)': round(r['concrete_m3'] * (1 + wastage/100) * UNIT_RATES.get('Concrete (C30/37)', 2500), 2)
                })
    else:  # rebar
        for r in results:
            if 'rebar_ton' in r:
                rows.append({
                    'Item': f"{element_type.capitalize()} - {r.get('label', '')} - Rebar",
                    'Count': r.get('count', 1),
                    'Unit': 'ton',
                    'Quantity (net)': round(r['rebar_ton'], 2),
                    'Wastage %': wastage,
                    'Quantity (with waste)': round(r['rebar_ton'] * (1 + wastage/100), 2),
                    'Unit Rate (EGP)': round(UNIT_RATES.get('Rebar (Grade 400)', 15000), 2),
                    'Total Cost (EGP)': round(r['rebar_ton'] * (1 + wastage/100) * UNIT_RATES.get('Rebar (Grade 400)', 15000), 2)
                })
            if 'concrete_m3' in r:
                rows.append({
                    'Item': f"{element_type.capitalize()} - {r.get('label', '')} - Concrete",
                    'Count': r.get('count', 1),
                    'Unit': 'm³',
                    'Quantity (net)': round(r['concrete_m3'], 2),
                    'Wastage %': wastage,
                    'Quantity (with waste)': round(r['concrete_m3'] * (1 + wastage/100), 2),
                    'Unit Rate (EGP)': round(UNIT_RATES.get('Concrete (C30/37)', 2500), 2),
                    'Total Cost (EGP)': round(r['concrete_m3'] * (1 + wastage/100) * UNIT_RATES.get('Concrete (C30/37)', 2500), 2)
                })
    # Add total row if rows exist
    if rows:
        total_row = {
            'Item': 'TOTAL',
            'Count': '',
            'Unit': '',
            'Quantity (net)': round(sum(r['Quantity (net)'] for r in rows), 2),
            'Wastage %': '',
            'Quantity (with waste)': round(sum(r['Quantity (with waste)'] for r in rows), 2),
            'Unit Rate (EGP)': '',
            'Total Cost (EGP)': round(sum(r['Total Cost (EGP)'] for r in rows), 2)
        }
        rows.append(total_row)
    return pd.DataFrame(rows)


def generate_charts(df, element_type):
    """Generate bar chart for concrete volume and pie chart for cost distribution."""
    df_no_total = df[df['Item'] != 'TOTAL'].copy()
    if df_no_total.empty:
        return None, None

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_no_total['Item'],
        y=df_no_total['Quantity (net)'],
        name='Concrete Volume (m³)',
        marker_color='#FF8C00',
        text=df_no_total['Quantity (net)'],
        textposition='auto',
    ))
    fig_bar.update_layout(
        title=f'{element_type.capitalize()} - Concrete Volume per Group',
        template='plotly_dark',
        paper_bgcolor='#0d1a35',
        plot_bgcolor='#0d1a35',
        font=dict(color='white'),
        margin=dict(t=40, b=20, l=40, r=20),
        height=400,
        xaxis_tickangle=-45,
    )

    fig_pie = go.Figure(data=[go.Pie(
        labels=df_no_total['Item'],
        values=df_no_total['Total Cost (EGP)'],
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Oranges_r),
        textinfo='label+percent',
        textposition='auto',
    )])
    fig_pie.update_layout(
        title=f'{element_type.capitalize()} - Cost Distribution',
        template='plotly_dark',
        paper_bgcolor='#0d1a35',
        plot_bgcolor='#0d1a35',
        font=dict(color='white'),
        margin=dict(t=40, b=20, l=40, r=20),
        height=400,
    )
    return fig_bar, fig_pie


# Helper function for rebar quantities
def compute_rebar_quantities(element_type, data, user_params):
    results = []
    total_concrete = 0
    total_rebar = 0
    for group in data:
        required = ['label', 'count', 'width_mm', 'depth_mm', 'height_mm', 'rebar']
        all_present = True
        for req in required:
            if req not in group or group[req] is None:
                all_present = False
                break
        if not all_present:
            continue
        height = group.get('height_mm') or user_params.get('floor_height_mm', 3000)
        vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (height/1000) * group.get('count', 1)
        total_concrete += vol
        rebar = group.get('rebar', {})
        main_d = rebar.get('main_diameter_mm', 0)
        stirrup_d = rebar.get('stirrup_diameter_mm', 0)
        spacing = rebar.get('spacing_mm', 200)
        count = group.get('count', 1)
        height_m = height / 1000
        main_length = height_m * 4 * count
        perimeter = 2 * ((group['width_mm'] + group['depth_mm']) / 1000)
        num_stirrups = (height_m / (spacing/1000)) + 1
        stirrup_length = perimeter * num_stirrups * count
        main_weight = main_length * ( (3.1416 * (main_d/1000)**2 / 4) * 7850 )
        stirrup_weight = stirrup_length * ( (3.1416 * (stirrup_d/1000)**2 / 4) * 7850 )
        total_rebar += (main_weight + stirrup_weight)
        results.append({
            'label': group.get('label', 'Unknown'),
            'count': count,
            'concrete_m3': vol,
            'rebar_ton': (main_weight + stirrup_weight) / 1000
        })
    total_concrete = round(total_concrete, 2)
    total_rebar = round(total_rebar / 1000, 2)
    return results, total_concrete, total_rebar


# =====================================================================================
# MIME TYPE DETECTION (fixes the PNG vs JPEG bug)
# =====================================================================================
def detect_mime_type(filename: str, data: bytes) -> str:
    """Detect MIME type from filename and magic bytes."""
    # First try by extension
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.png']:
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext in ['.pdf']:
        return 'application/pdf'
    # Fallback to magic bytes
    if data.startswith(b'\x89PNG'):
        return 'image/png'
    if data.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    if data.startswith(b'%PDF'):
        return 'application/pdf'
    return 'image/jpeg'


# =====================================================================================
# MAIN APP LAYOUT
# =====================================================================================
@ui.page('/')
def main_page():
    ui.query('body').style('width: 100vw; height: 100vh; overflow-x: hidden;')

    # ---------------- SIDEBAR (unchanged) ----------------
    sidebar = ui.left_drawer().classes('sidebar-container').style('width: 380px;')
    with sidebar:
        with ui.row().classes('w-full items-center justify-between mb-4 p-2'):
            ui.label('📋 PROJECT METADATA').classes('text-white font-bold text-base tracking-wide')
            ui.button('✕', on_click=sidebar.toggle).classes(
                'bg-transparent text-white text-xl hover:text-[#FF8C00] p-1 min-w-[36px] !shadow-none !rounded-full !bg-transparent'
            ).style('font-size: 20px; line-height: 1;')

        project_name_input = ui.input(label='Project Name', value='Highway Expansion Project').classes('w-full mb-3')
        pour_location_input = ui.input(label='Structural Element / Chainage', value='Highway Section Ch. 12+500').classes('w-full mb-4')

        ui.label('Governing Design Code Basis').classes('text-white font-bold text-sm mb-1')
        ui.markdown('By default every AI output in this app is generated strictly per **ECP 203 / ECP 202 / ECP 104**. Change this to switch the primary basis.').classes('text-xs text-[#A9B6D0] mb-2')
        code_basis_select = ui.select(
            label='Code Type (applies app-wide)',
            options=CODE_BASIS_OPTIONS,
            value=CODE_BASIS_OPTIONS[0],
        ).classes('w-full mb-4')

        fcu_input = ui.number(label='Specified 28-Day Grade f_cu (N/mm2)', value=30.0, step=5.0).classes('w-full mb-4')

        ui.label('Batch Plant & Site Logs').classes('text-white font-bold text-sm mb-2')
        truck_input = ui.input(label='Mixer Truck No.', value='TRK-104').classes('w-full mb-2')
        ticket_input = ui.input(label='Batch Ticket ID', value='BT-99482').classes('w-full mb-4')

        ui.label('Mix Design Parameters').classes('text-white font-bold text-sm mb-2')
        cement_input = ui.input(label='Cement Content (kg/m3)', value='350.0').classes('w-full mb-2')
        water_input = ui.input(label='Free Water Content (kg/m3)', value='150.0').classes('w-full mb-4')

        engineer_input = ui.input(label='Engineer Name', value='Eng. Mohamed Abd Al Aty').classes('w-full mb-2')

        logo_status = ui.label('Logo: Not uploaded').classes('text-xs text-amber-400 mb-1')
        logo_bytes_holder = {'bytes': None}

        async def handle_logo_upload(e):
            try:
                logo_bytes_holder['bytes'] = await e.file.read()
                logo_status.set_text(f'Logo Loaded: {e.file.name}')
                logo_status.classes(replace='text-xs text-emerald-400 mb-1')
                ui.notify('Company logo loaded successfully!', type='positive')
            except Exception as ex:
                ui.notify(f'Error reading logo: {str(ex)}', type='negative')

        ui.upload(label='Upload Company Logo', auto_upload=True, on_upload=handle_logo_upload).props('flat dark').classes('w-full mb-2')

    ui.button('☰', on_click=sidebar.toggle).classes(
        'fixed top-4 left-4 z-50 bg-[#10203f] text-white border border-[#FF8C00] p-3 rounded-full shadow-lg hover:bg-[#1a2a4a]'
    ).style('font-size: 20px; min-width: 48px; min-height: 48px;')

    def current_meta(uid_prefix):
        return {
            'uid': f"{uid_prefix}-{uuid.uuid4().hex[:8].upper()}",
            'project': project_name_input.value,
            'location': pour_location_input.value,
            'engineer': engineer_input.value,
            'date': datetime.date.today().strftime('%Y-%m-%d'),
            'ticket': ticket_input.value,
        }

    # ---------------- MAIN COLUMN ----------------
    with ui.column().classes('w-full min-h-screen p-4 bg-[#031338]'):
        # Title block (unchanged)
        with ui.column().classes('w-full bg-[#0d1a35] px-6 py-4 rounded-xl border border-[#FF8C00] shadow-lg mb-4'):
            ui.label('SMART EGY-CIVIL AI AUDITOR').classes('main-title text-white')
            ui.label('Intelligent General Civil, Geotechnical & Structural Compliance Engine').classes('sub-title text-lg font-medium mt-1')
            ui.label('Lead Technical Auditor: Eng. Mohamed Abd Al Aty').classes('text-base text-[#A9B6D0] font-semibold mt-1')
            ui.label('Next-generation automated civil engineering and quality intelligence, precision-calibrated for the Egyptian Code of Practice.').classes('text-sm text-[#A9B6D0] mt-1 italic')

        ui.add_head_html('''
        <style>@keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }</style>
        ''')
        ui.html('''
        <div style="width: 100%; overflow: hidden; white-space: nowrap; background-color: rgba(13,26,53,0.6); backdrop-filter: blur(8px); color: #FFFFFF; padding: 10px 0; font-weight: 600; font-size: 13px; margin-bottom: 15px; border-radius: 8px; border: 1px solid rgba(255,140,0,0.3);">
          <div style="display: inline-block; padding-left: 100%; animation: marquee 28s linear infinite;">
            <span style="color: #FF8C00;">[CORE ACTIVE]</span> ECP 203 &middot; ECP 202 &middot; ECP 104 &middot; ASTM &middot; AASHTO &middot; BS EN &middot; ISO
            &nbsp;&nbsp;|&nbsp;&nbsp; Advanced Geotechnical & Concrete Calculation Sheet &nbsp;&nbsp;|&nbsp;&nbsp; Active Site Inspection Portal
          </div>
        </div>
        ''')

        # Tabs (now with Job Board)
        with ui.tabs().classes('w-full text-white bg-[#0d1a35] rounded-lg') as tabs:
            t_dash = ui.tab('Concrete Cube Verifier').classes('text-white font-bold')
            t_audit = ui.tab('AI Multi-Standard Auditor').classes('text-white font-bold')
            t_defect = ui.tab('Defect Diagnostic').classes('text-white font-bold')
            t_chat = ui.tab('AI Chatbot').classes('text-white font-bold')
            t_handwriting = ui.tab('Handwriting OCR').classes('text-white font-bold')
            t_jobs = ui.tab('Job Board').classes('text-white font-bold')

        with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent mt-4'):

            # =========================================================================
            # TAB 1: CONCRETE CUBE VERIFIER (unchanged)
            # =========================================================================
            with ui.tab_panel(t_dash):
                ui.label('Concrete Cube Calculation Sheet & Statistical Verifier').classes('text-2xl font-bold text-white mb-4')

                with ui.row().classes('w-full gap-4 mb-4'):
                    with ui.column().classes('input-card flex-1'):
                        ui.label('7-Day Cubes (comma separated, N/mm2)').classes('font-bold text-white text-sm')
                        c7_input = ui.input(value='21.0, 22.5, 20.5').classes('w-full')
                    with ui.column().classes('input-card flex-1'):
                        ui.label('14-Day Cubes (comma separated, N/mm2)').classes('font-bold text-white text-sm')
                        c14_input = ui.input(value='26.0, 27.2, 25.8').classes('w-full')
                    with ui.column().classes('input-card flex-1'):
                        ui.label('28-Day Cubes (comma separated, N/mm2)').classes('font-bold text-white text-sm')
                        c28_input = ui.input(value='32.5, 34.0, 31.0, 35.5, 29.0, 33.0').classes('w-full')

                ai_cube_result_holder = {'text': ''}

                def parse_vals(txt):
                    try:
                        return [float(x.strip()) for x in txt.split(',') if x.strip() != '']
                    except Exception:
                        return []

                def compute_stats(values):
                    if not values:
                        return None
                    arr = np.array(values, dtype=float)
                    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                    mean = float(arr.mean())
                    return {
                        'n': len(arr), 'mean': mean, 'std': std,
                        'min': float(arr.min()), 'max': float(arr.max()),
                        'cov': (std / mean * 100.0) if mean > 0 else 0.0,
                    }

                def get_selected_stages(stage_filter):
                    all_stages = [
                        ('7-Day', c7_input, parse_vals(c7_input.value)),
                        ('14-Day', c14_input, parse_vals(c14_input.value)),
                        ('28-Day', c28_input, parse_vals(c28_input.value)),
                    ]
                    mapping = {'7-Day Stage': [0], '14-Day Stage': [1], '28-Day Stage': [2]}
                    if stage_filter in mapping:
                        idxs = mapping[stage_filter]
                        return [all_stages[i] for i in idxs]
                    return all_stages

                async def run_verification():
                    result_output_area.clear()
                    export_buttons_area.clear()
                    chart_area.clear()
                    stats_area.clear()

                    if not client:
                        ui.notify('GEMINI_API_KEY missing in .env!', type='negative')
                        return

                    with result_output_area:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Running AI statistical evaluation & code compliance verification...').classes('self-center text-sm')

                    try:
                        stage_filter = stage_selector.value
                        stages = get_selected_stages(stage_filter)
                        target_fcu = float(fcu_input.value) if fcu_input.value else 30.0
                        basis = code_basis_select.value

                        stage_stats = []
                        for label, _inp, values in stages:
                            s = compute_stats(values)
                            stage_stats.append((label, values, s))

                        stats_area.clear()
                        with stats_area:
                            with ui.row().classes('w-full gap-4 flex-wrap mb-2'):
                                for label, values, s in stage_stats:
                                    if not s:
                                        continue
                                    with ui.column().classes('stat-chip'):
                                        ui.label(f"{s['mean']:.2f}").classes('val')
                                        ui.label(f'{label} Mean (N/mm2)').classes('lbl')
                                    with ui.column().classes('stat-chip'):
                                        ui.label(f"{s['std']:.2f}").classes('val')
                                        ui.label(f'{label} Std Dev').classes('lbl')
                                    with ui.column().classes('stat-chip'):
                                        ui.label(f"{s['min']:.1f} / {s['max']:.1f}").classes('val')
                                        ui.label(f'{label} Min / Max').classes('lbl')

                        stage_data_text = "\n".join(
                            f"- {label} Crushing Values (N/mm2): {', '.join(str(v) for v in values) if values else 'No data provided'} "
                            f"(n={s['n'] if s else 0}, mean={s['mean']:.2f} if s else 'n/a')"
                            for label, values, s in stage_stats
                        )

                        prompt = f"""
You are an elite Senior Concrete Quality Assurance and Structural Engineering Expert.
Perform a complete, professional statistical evaluation and code-compliance verification
for the concrete cube test results below. Only evaluate the stage(s) actually provided.

{get_code_directive(basis)}

{NO_LATEX_RULE}

DISPLAY FILTER SELECTED BY USER: {stage_filter}
(Only discuss the stage(s) listed below in detail; do not invent data for stages not listed.)

PROJECT PARAMETERS:
- Specified 28-Day Characteristic Compressive Strength (f_cu): {target_fcu} N/mm2
{stage_data_text}
- Mix Details: Cement = {cement_input.value} kg/m3, Water = {water_input.value} kg/m3
- Truck No: {truck_input.value} | Ticket ID: {ticket_input.value}

REQUIRED REPORT STRUCTURE:
1. A Markdown table per stage: Specimen ID, Crushing Strength, Deviation from Mean, Individual Limit Check.
2. A short statistical commentary (mean, standard deviation, coefficient of variation) referencing the numbers above.
3. A clear final compliance verdict (PASS / FAIL) with the specific ECP 203 (or selected code) clause used to judge it.
"""

                        res_text = await call_gemini(prompt)
                        ai_cube_result_holder['text'] = res_text

                        result_output_area.clear()
                        with result_output_area:
                            with ui.column().classes('output-card w-full'):
                                ui.label('AI Statistical Evaluation & Compliance Verdict').classes('text-xl font-bold text-white mb-2')
                                ui.markdown(res_text).classes('markdown-body')

                        with chart_area:
                            labels = [label for label, _v, _s in stage_stats] + ['Target Grade']
                            means = [(s['mean'] if s else 0) for _l, _v, s in stage_stats] + [target_fcu]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=labels, y=means, mode='lines+markers+text',
                                text=[f"{v:.1f}" for v in means], textposition="top center",
                                line=dict(color='#4FC3F7', width=3), marker=dict(size=10, color='#FF8C00'),
                            ))
                            fig.add_hline(y=target_fcu, line_dash="dash", line_color="#22C55E",
                                          annotation_text=f"Target f_cu ({target_fcu} N/mm2)", annotation_position="bottom right")
                            fig.update_layout(
                                title=f'Compressive Strength — {stage_filter}',
                                template='plotly_dark', paper_bgcolor='#0d1a35', plot_bgcolor='#0d1a35',
                                margin=dict(t=40, b=20, l=40, r=20), height=340,
                            )
                            ui.plotly(fig).classes('w-full mt-2')

                        with export_buttons_area:
                            def download_pdf_report():
                                try:
                                    meta = current_meta('ECP-AI')
                                    styles = build_pdf_styles()
                                    stat_rows = [["Stage", "n", "Mean (N/mm2)", "Std Dev", "Min", "Max", "COV %"]]
                                    for label, values, s in stage_stats:
                                        if s:
                                            stat_rows.append([label, str(s['n']), f"{s['mean']:.2f}", f"{s['std']:.2f}",
                                                               f"{s['min']:.1f}", f"{s['max']:.1f}", f"{s['cov']:.1f}"])
                                    colw = USABLE_WIDTH / len(stat_rows[0])
                                    stat_table_data = [[Paragraph(c, styles['tablehead'] if r == 0 else styles['tablecell'])
                                                         for c in row] for r, row in enumerate(stat_rows)]
                                    stat_table = Table(stat_table_data, colWidths=[colw] * len(stat_rows[0]))
                                    stat_table.setStyle(TableStyle([
                                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B2A4A')),
                                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
                                        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
                                        ('TOPPADDING', (0, 0), (-1, -1), 4), ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                                    ]))
                                    pdf_bytes = build_report_pdf(
                                        "AI CONCRETE CUBE CALCULATION & VERIFICATION REPORT",
                                        f"Governing Standard: {basis} | Filter: {stage_filter}",
                                        ai_cube_result_holder['text'], meta, logo_bytes_holder['bytes'],
                                        extra_flowables_before_body=[
                                            Paragraph("Deterministic Statistics", styles['h2']), stat_table,
                                        ],
                                    )
                                    ui.download(pdf_bytes, filename=f"AI_Concrete_Calculation_Sheet_{ticket_input.value}.pdf")
                                    ui.notify('Calculation Sheet PDF downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Generation Error: {str(ex)}', type='negative')

                            def download_csv_export():
                                rows = {"Field": [], "Value": []}
                                rows["Field"] += ["Project Name", "Location", "Specified f_cu", "Code Basis", "Stage Filter", "Truck No", "Batch Ticket"]
                                rows["Value"] += [project_name_input.value, pour_location_input.value, str(fcu_input.value),
                                                   basis, stage_filter, truck_input.value, ticket_input.value]
                                for label, values, s in stage_stats:
                                    rows["Field"].append(f"{label} Mean / Std Dev")
                                    rows["Value"].append(f"{s['mean']:.2f} / {s['std']:.2f}" if s else "No data")
                                df = pd.DataFrame(rows)
                                ui.download(df.to_csv(index=False).encode('utf-8'), filename=f"AI_Concrete_Calculation_{ticket_input.value}.csv")
                                ui.notify('CSV downloaded!', type='positive')

                            ui.button('Download Calculation PDF', on_click=download_pdf_report).classes('primary-btn flex-1')
                            ui.button('Export CSV', on_click=download_csv_export).classes('primary-btn flex-1')

                    except Exception as ex:
                        result_output_area.clear()
                        with result_output_area:
                            ui.notify(f'Calculation Error: {str(ex)}', type='negative')

                stage_selector = ui.select(
                    label='Select Stage Display Filter',
                    options=['All Stages', '7-Day Stage', '14-Day Stage', '28-Day Stage'],
                    value='All Stages',
                    on_change=run_verification,
                ).classes('w-full md:w-1/3 mb-4')

                stats_area = ui.column().classes('w-full')
                result_output_area = ui.column().classes('w-full')
                chart_area = ui.column().classes('w-full')
                export_buttons_area = ui.row().classes('w-full gap-4 mt-4')

                ui.button('Run AI Statistical Calculation & Verification', on_click=run_verification).classes('primary-btn q-my-md')
                with result_output_area:
                    ui.markdown('*Click "Run AI Statistical Calculation & Verification" to generate the report.*').classes('text-sm text-[#A9B6D0]')

            # =========================================================================
            # TAB 2: AI MULTI-STANDARD AUDITOR (unchanged)
            # =========================================================================
            with ui.tab_panel(t_audit):
                ui.label('AI Multi-Standard Engineering Auditor').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload a specification, mix design, or site report to audit against the selected code basis.').classes('markdown-body mb-2')

                audit_focus = ui.select(
                    label='Audit Focus',
                    options=[
                        "Multi-Standard Structural & Geotechnical Compliance",
                        "Roads, Pavements & Subgrade Materials (ECP 104 & AASHTO)",
                        "Soil Mechanics & Foundations (ECP 202 & ASTM / ISO)",
                        "Reinforced Concrete Structures (ECP 203 & ACI / BS EN)",
                    ],
                    value="Multi-Standard Structural & Geotechnical Compliance",
                ).classes('w-full mb-4')

                audit_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
                uploaded_file_data = {'bytes': None, 'name': None, 'type': None}

                async def handle_audit_upload(e):
                    try:
                        data = await e.file.read()
                        uploaded_file_data['bytes'] = data
                        uploaded_file_data['name'] = e.file.name
                        uploaded_file_data['type'] = detect_mime_type(e.file.name, data)
                        audit_status_label.set_text(f'File Ready: {e.file.name}')
                        audit_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                        ui.notify(f'Successfully loaded: {e.file.name}', type='positive')
                    except Exception as ex:
                        ui.notify(f'Error reading file: {str(ex)}', type='negative')

                ui.upload(label='Select PDF or Image File', auto_upload=True, on_upload=handle_audit_upload).props('flat dark').classes('w-full mb-4')

                audit_output_container = ui.column().classes('w-full')
                audit_export_container = ui.row().classes('w-full gap-4 mt-4')
                audit_result_text_holder = {'text': ''}

                async def run_ai_audit():
                    if not client:
                        ui.notify('GEMINI_API_KEY missing in .env!', type='negative')
                        return
                    if not uploaded_file_data['bytes']:
                        ui.notify('Please upload a file first!', type='warning')
                        return

                    audit_output_container.clear()
                    audit_export_container.clear()
                    with audit_output_container:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Executing multi-standard engineering audit...').classes('self-center text-sm')

                    try:
                        basis = code_basis_select.value
                        prompt = f"""
You are a Principal Civil, Geotechnical and Highway Engineering Consultant and Lead Auditor.
Audit Focus: {audit_focus}

{get_code_directive(basis)}

{NO_LATEX_RULE}

Perform a comprehensive technical audit of the provided document or image. Structure your
report with clear ## section headings and real Markdown tables for any comparative data.
"""
                        contents = [prompt]
                        if uploaded_file_data['type'] == 'application/pdf':
                            reader = pypdf.PdfReader(io.BytesIO(uploaded_file_data['bytes']))
                            text = "".join([p.extract_text() or "" for p in reader.pages[:10]])
                            if len(text) > 10000:
                                text = text[:10000] + "\n... (truncated)"
                            contents.append(f"Extracted PDF Text:\n{text}")
                        else:
                            img_part = types.Part.from_bytes(data=uploaded_file_data['bytes'], mime_type=uploaded_file_data['type'])
                            contents.append(img_part)

                        audit_result_text = await call_gemini(contents, timeout=240)
                        audit_result_text_holder['text'] = audit_result_text

                        audit_output_container.clear()
                        with audit_output_container:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Engineering Audit Findings & Code Compliance Report').classes('text-xl font-bold text-white mb-2')
                                ui.markdown(audit_result_text).classes('markdown-body')

                        with audit_export_container:
                            def download_audit_pdf():
                                try:
                                    meta = current_meta('AUDIT')
                                    pdf_bytes = build_report_pdf(
                                        "AI MULTI-STANDARD ENGINEERING AUDIT REPORT",
                                        f"Focus: {audit_focus} | Basis: {basis}",
                                        audit_result_text_holder['text'], meta, logo_bytes_holder['bytes'],
                                    )
                                    ui.download(pdf_bytes, filename=f"AI_Audit_Report_{ticket_input.value}.pdf")
                                    ui.notify('Audit PDF downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Export Error: {str(ex)}', type='negative')

                            def download_audit_csv():
                                df = pd.DataFrame({
                                    "Audit Field": ["Project Name", "Focus", "Code Basis", "Source File", "Engineer", "Summary Findings"],
                                    "Value": [project_name_input.value, audit_focus, basis, uploaded_file_data['name'],
                                              engineer_input.value, audit_result_text_holder['text'][:300].replace('\n', ' ')],
                                })
                                ui.download(df.to_csv(index=False).encode('utf-8'), filename=f"AI_Audit_{ticket_input.value}.csv")
                                ui.notify('Audit CSV downloaded!', type='positive')

                            ui.button('Download Audit PDF', on_click=download_audit_pdf).classes('primary-btn flex-1')
                            ui.button('Export Audit CSV', on_click=download_audit_csv).classes('primary-btn flex-1')

                    except Exception as ex:
                        audit_output_container.clear()
                        with audit_output_container:
                            ui.notify(f'Error: {str(ex)}', type='negative')

                ui.button('Execute AI Audit & Compliance Check', on_click=run_ai_audit).classes('primary-btn')

            # =========================================================================
            # TAB 3: DEFECT DIAGNOSTIC (unchanged)
            # =========================================================================
            with ui.tab_panel(t_defect):
                ui.label('AI Engineering Defect Diagnostic & Repair Protocol').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload site defect photos or PDFs for forensic analysis. Describe the issue below for more precise diagnosis.').classes('markdown-body mb-2')

                defect_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
                defect_file_data = {'bytes': None, 'type': None}
                defect_result_holder = {'text': ''}
                defect_user_message = ui.input(label='Describe the defect or additional context (optional)',
                                               placeholder='e.g., "Cracks near column base with spalling concrete"').classes('w-full mb-3')

                async def handle_defect_upload(e):
                    try:
                        data = await e.file.read()
                        defect_file_data['bytes'] = data
                        defect_file_data['type'] = detect_mime_type(e.file.name, data)
                        defect_status_label.set_text(f'File Ready: {e.file.name}')
                        defect_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                        ui.notify(f'Successfully loaded file: {e.file.name}', type='positive')
                    except Exception as ex:
                        ui.notify(f'Error reading file: {str(ex)}', type='negative')

                ui.upload(label='Select Site Defect Photo or PDF', auto_upload=True, on_upload=handle_defect_upload).props('flat dark').classes('w-full mb-4')
                defect_output = ui.column().classes('w-full')
                defect_export_area = ui.row().classes('w-full gap-4 mt-4')

                async def run_defect_diagnosis():
                    if not client or not defect_file_data['bytes']:
                        ui.notify('API key missing or file not uploaded!', type='negative')
                        return
                    defect_output.clear()
                    defect_export_area.clear()
                    with defect_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Analyzing defect and generating repair protocol...').classes('self-center text-sm')

                    try:
                        basis = code_basis_select.value
                        user_desc = defect_user_message.value.strip() or "No additional description provided."
                        contents = []
                        prompt = f"""
You are a Senior Forensic Structural Engineer and Materials Specialist.
Perform a detailed engineering diagnostic of the defect shown. The user has provided the following description:
"{user_desc}"

{get_code_directive(basis)}

{NO_LATEX_RULE}

Based on the visual evidence (and description), provide:
1. A clear identification of the defect type and severity.
2. Root cause analysis with reference to code provisions.
3. A detailed repair protocol with step-by-step instructions.
4. **A professional table of recommended repair products available in the Egyptian market** with columns:
   - Product Name
   - Manufacturer (e.g., Sika, Fosroc, etc.)
   - Application Method
   - Unit Price (EGP) – provide realistic current market prices in Egyptian Pounds.
   - Quantity Required (estimate)
   - Total Cost (EGP)
5. Overall cost summary and recommended contractor qualification.

Ensure all tables are proper Markdown tables with header and separator rows.
"""
                        contents.append(prompt)
                        if defect_file_data['type'] == 'application/pdf':
                            reader = pypdf.PdfReader(io.BytesIO(defect_file_data['bytes']))
                            text = "".join([p.extract_text() or "" for p in reader.pages[:10]])
                            if len(text) > 10000:
                                text = text[:10000] + "\n... (truncated)"
                            contents.append(f"Extracted PDF Text (if any):\n{text}")
                        else:
                            img_part = types.Part.from_bytes(data=defect_file_data['bytes'], mime_type=defect_file_data['type'])
                            contents.append(img_part)

                        res_text = await call_gemini(contents, timeout=240)
                        defect_result_holder['text'] = res_text

                        defect_output.clear()
                        with defect_output:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Forensic Diagnosis & Repair Protocol with Market Prices').classes('text-xl font-bold text-white mb-2')
                                ui.markdown(res_text).classes('markdown-body')

                        with defect_export_area:
                            def download_defect_pdf():
                                try:
                                    meta = current_meta('DEFECT')
                                    pdf_bytes = build_report_pdf(
                                        "AI DEFECT DIAGNOSTIC & REPAIR REPORT",
                                        "Forensic Structural Evaluation with Product Pricing",
                                        defect_result_holder['text'], meta, logo_bytes_holder['bytes'],
                                    )
                                    ui.download(pdf_bytes, filename=f"Defect_Diagnostic_Report_{ticket_input.value}.pdf")
                                    ui.notify('Defect Diagnostic PDF downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Export Error: {str(ex)}', type='negative')

                            def download_defect_csv():
                                try:
                                    df = pd.DataFrame({
                                        "Diagnostic Report": [defect_result_holder['text']]
                                    })
                                    ui.download(df.to_csv(index=False).encode('utf-8'), filename=f"Defect_Report_{ticket_input.value}.csv")
                                    ui.notify('CSV downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'CSV Export Error: {str(ex)}', type='negative')

                            ui.button('Download Defect PDF Report', on_click=download_defect_pdf).classes('primary-btn flex-1')
                            ui.button('Export Report as CSV', on_click=download_defect_csv).classes('primary-btn flex-1')

                    except Exception as ex:
                        defect_output.clear()
                        with defect_output:
                            ui.notify(f'Diagnosis failed: {ex}', type='negative')

                ui.button('Diagnose Defect & Get Repair Protocol', on_click=run_defect_diagnosis).classes('primary-btn')

            # =========================================================================
            # TAB 4: AI CHATBOT (unchanged)
            # =========================================================================
            with ui.tab_panel(t_chat):
                ui.label('Core-Code Intelligent Assistant Chatbot').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Ask any engineering, mix design, geotechnical, or pavement question and get answers based on the Egyptian Codes (ECP 203, ECP 202, ECP 104) and international standards.').classes('markdown-body mb-2')

                chat_container = ui.column().classes('output-card w-full h-[500px] overflow-y-auto mb-4')
                chat_messages = [{"role": "assistant", "content": "Hello! I am your Multi-Standard Engineering Assistant. How can I assist you today?"}]

                def render_chat():
                    chat_container.clear()
                    with chat_container:
                        for msg in chat_messages:
                            is_ai = msg['role'] == 'assistant'
                            with ui.column().classes('chat-message'):
                                role_label = 'Assistant' if is_ai else 'You'
                                label_class = 'assistant' if is_ai else 'user'
                                ui.label(role_label).classes(f'role-label {label_class}')
                                ui.markdown(msg['content']).classes('content markdown-body')

                render_chat()
                user_msg = ui.input(placeholder='Type your engineering question here...').classes('w-full mb-2')
                user_msg.on('keydown.enter', lambda: send_chat())

                async def send_chat():
                    q = user_msg.value
                    if not q or not q.strip():
                        return
                    chat_messages.append({"role": "user", "content": q})
                    user_msg.value = ''
                    render_chat()

                    if not client:
                        chat_messages.append({"role": "assistant", "content": "GEMINI_API_KEY is not configured."})
                        render_chat()
                        return

                    try:
                        basis = code_basis_select.value
                        system_prompt = (
                            "You are an elite Senior Civil, Geotechnical, and Structural Quality Engineering Expert "
                            "acting as a master multi-standard technical assistant.\n\n"
                            f"{get_code_directive(basis)}\n\n{NO_LATEX_RULE}\n\n"
                            "UNIT SYSTEM: Use strictly METRIC (SI) units (mm, cm, m, MPa, kN, kg/m3, C)."
                        )
                        cleaned_response = await call_gemini(q, system_instruction=system_prompt, timeout=240)
                        chat_messages.append({"role": "assistant", "content": cleaned_response})
                    except Exception as e:
                        chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                    render_chat()

                with ui.row().classes('w-full gap-4 mt-2'):
                    ui.button('Send Query', on_click=send_chat).classes('primary-btn flex-1')

                    def download_chat_pdf():
                        try:
                            meta = current_meta('CHAT')
                            styles = build_pdf_styles()
                            flowables = []
                            for m in chat_messages:
                                role_label = "ASSISTANT" if m['role'] == 'assistant' else "USER"
                                flowables.append(Paragraph(role_label, styles['h3']))
                                flowables.extend(markdown_to_pdf_flowables(m['content'], styles))
                                flowables.append(Spacer(1, 4))
                            pdf_bytes = build_report_pdf(
                                "AI ENGINEERING ASSISTANT TRANSCRIPT",
                                "Official Q&A Consultation Record",
                                "", meta, logo_bytes_holder['bytes'],
                                extra_flowables_before_body=flowables,
                            )
                            ui.download(pdf_bytes, filename=f"AI_Chat_Transcript_{ticket_input.value}.pdf")
                            ui.notify('Chat Transcript PDF downloaded!', type='positive')
                        except Exception as ex:
                            ui.notify(f'PDF Export Error: {str(ex)}', type='negative')

                    ui.button('Download Chat PDF Transcript', on_click=download_chat_pdf).classes('primary-btn flex-1')

            # =========================================================================
            # TAB 5: HANDWRITING OCR (enhanced with editing and custom PDF)
            # =========================================================================
            with ui.tab_panel(t_handwriting):
                ui.label('Handwriting to Digital Text Transcription').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload a scanned handwritten note (PNG, JPG) or PDF. The AI will convert it to clean digital text, detecting tables if present.').classes('markdown-body mb-2')

                # File upload
                ocr_file_data = {'bytes': None, 'type': None, 'name': None}
                ocr_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
                async def handle_ocr_upload(e):
                    try:
                        data = await e.file.read()
                        ocr_file_data['bytes'] = data
                        ocr_file_data['type'] = detect_mime_type(e.file.name, data)
                        ocr_file_data['name'] = e.file.name
                        ocr_status_label.set_text(f'File Ready: {e.file.name} ({(len(data)/1024/1024):.1f} MB)')
                        ocr_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                        ui.notify(f'File uploaded: {e.file.name}', type='positive')
                    except Exception as ex:
                        ui.notify(f'Error: {str(ex)}', type='negative')

                ui.upload(label='Upload Handwriting Image or PDF', auto_upload=True, on_upload=handle_ocr_upload).props('flat dark').classes('w-full mb-4')

                # Output area: editable text and download buttons
                ocr_output = ui.column().classes('w-full')
                ocr_export = ui.row().classes('w-full gap-4 mt-4')
                transcribed_text_holder = {'text': ''}
                # We'll create a textarea for editing
                text_editor = None

                async def run_ocr():
                    if not client:
                        ui.notify('GEMINI_API_KEY missing!', type='negative')
                        return
                    if not ocr_file_data['bytes']:
                        ui.notify('Please upload a handwriting file first.', type='warning')
                        return

                    ocr_output.clear()
                    ocr_export.clear()
                    with ocr_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Transcribing handwriting...').classes('self-center text-sm')

                    try:
                        # Build contents: prompt with instruction to format tables as Markdown
                        prompt = """
You are an expert OCR system. Transcribe the handwritten text from the provided image(s).
- If you detect any tabular data (rows and columns), format it as a proper Markdown table with a header row and a separator line (|---|...|).
- Return only the transcribed text and tables, without any additional commentary, headings, or formatting.
- If there are multiple pages, combine them in order.
"""
                        contents = [prompt]

                        # Process file – send pages as PNG
                        if ocr_file_data['type'] == 'application/pdf':
                            try:
                                doc = fitz.open(stream=ocr_file_data['bytes'], filetype="pdf")
                                for page_num in range(min(6, len(doc))):
                                    page = doc.load_page(page_num)
                                    mat = fitz.Matrix(2.0, 2.0)
                                    pix = page.get_pixmap(matrix=mat)
                                    img_bytes = pix.tobytes("png")
                                    img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                                    contents.append(img_part)
                                doc.close()
                            except Exception:
                                contents.append(types.Part.from_bytes(data=ocr_file_data['bytes'], mime_type='application/pdf'))
                        else:
                            img_part = types.Part.from_bytes(data=ocr_file_data['bytes'], mime_type=ocr_file_data['type'])
                            contents.append(img_part)

                        # Call Gemini
                        response_text = await call_gemini(contents, temperature=0, timeout=240)
                        transcribed = sanitize_ai_markdown(response_text)  # clean
                        transcribed_text_holder['text'] = transcribed

                        # Display the transcribed text in a textarea for editing
                        ocr_output.clear()
                        with ocr_output:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Transcribed Text (editable)').classes('text-xl font-bold text-white mb-2')
                                # Use a textarea with the content
                                text_editor = ui.textarea(value=transcribed, placeholder='Edit the transcribed text here...').classes('w-full markdown-body').style('min-height: 300px; background: #0a1a3a; color: white; border: 1px solid #FF8C00;')
                                # Preview of rendered markdown (optional)
                                ui.label('Preview:').classes('text-lg font-bold text-white mt-2')
                                preview_container = ui.column().classes('w-full')
                                def update_preview():
                                    preview_container.clear()
                                    with preview_container:
                                        ui.markdown(text_editor.value).classes('markdown-body')
                                text_editor.on('input', update_preview)
                                # Initial preview
                                update_preview()

                        # Export buttons using the current text from the editor
                        with ocr_export:
                            def download_ocr_pdf():
                                try:
                                    # Get current text from editor
                                    current_text = text_editor.value if text_editor else transcribed_text_holder['text']
                                    meta = current_meta('OCR')
                                    # Use custom PDF with no doc_title/subtitle, and no ticket in header
                                    # We'll use build_report_pdf with show_ticket=False and empty doc_title/subtitle
                                    pdf_bytes = build_report_pdf(
                                        doc_title="",  # empty to hide
                                        subtitle="",   # empty to hide
                                        body_markdown=current_text,
                                        meta=meta,
                                        logo_bytes=logo_bytes_holder['bytes'],
                                        show_ticket=False  # hides Batch Ticket ID
                                    )
                                    ui.download(pdf_bytes, filename=f"Handwriting_Transcription_{ticket_input.value}.pdf")
                                    ui.notify('PDF report downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Error: {str(ex)}', type='negative')

                            def download_ocr_txt():
                                try:
                                    current_text = text_editor.value if text_editor else transcribed_text_holder['text']
                                    txt_bytes = current_text.encode('utf-8')
                                    ui.download(txt_bytes, filename=f"Handwriting_Transcription_{ticket_input.value}.txt")
                                    ui.notify('TXT file downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'TXT Error: {str(ex)}', type='negative')

                            ui.button('Download PDF Report', on_click=download_ocr_pdf).classes('primary-btn flex-1')
                            ui.button('Download TXT', on_click=download_ocr_txt).classes('primary-btn flex-1')

                    except Exception as ex:
                        ocr_output.clear()
                        with ocr_output:
                            ui.notify(f'Transcription failed: {str(ex)}', type='negative')
                            ui.label('Error occurred. Please try again with a clearer image.').classes('text-red-400')

                ui.button('Transcribe Handwriting', on_click=run_ocr).classes('primary-btn')

                # Initial placeholder
                with ocr_output:
                    ui.markdown('*Upload a file and click "Transcribe Handwriting" to start.*').classes('text-sm text-[#A9B6D0]')

            # =========================================================================
            # TAB 6: JOB BOARD (FIXED)
            # =========================================================================
            with ui.tab_panel(t_jobs):
                ui.label('Engineering Job Board - Egypt').classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Search for the latest engineering jobs in Egypt. Results are scraped from **Wuzzuf** (fallback to **Bayt** if needed). No API key required.').classes('markdown-body mb-2')

                # Search inputs
                with ui.row().classes('w-full gap-4 mb-4'):
                    search_input = ui.input(label='Search for jobs', placeholder='e.g., Civil Engineer', value='Civil Engineer').classes('flex-1')
                    location_input = ui.input(label='Location (optional)', placeholder='e.g., Cairo').classes('flex-1')
                    search_button = ui.button('Search Jobs', on_click=lambda: search_jobs()).classes('primary-btn')

                # Filter input (client‑side filtering)
                filter_input = ui.input(label='Filter results', placeholder='Type to filter title, company, description...', on_change=lambda: filter_jobs()).classes('w-full mb-2')

                # Container for results
                results_container = ui.column().classes('w-full')
                jobs_data = []  # store current job list

                def display_jobs(jobs, filter_text=''):
                    """Render job cards with optional client‑side filtering."""
                    results_container.clear()
                    with results_container:
                        if not jobs:
                            ui.label('No jobs found. Try a different search.').classes('text-white')
                            return
                        filtered = jobs
                        if filter_text:
                            f_lower = filter_text.lower()
                            filtered = [j for j in jobs if f_lower in j['title'].lower() or f_lower in j['company'].lower() or f_lower in j['description'].lower()]
                        if not filtered:
                            ui.label('No jobs match the filter.').classes('text-white')
                            return
                        # Display each job as a card
                        for job in filtered:
                            with ui.card().classes('w-full bg-[#0d1a35] border border-[#2c3f6b] rounded-lg p-3 mb-2'):
                                with ui.row().classes('w-full justify-between'):
                                    ui.label(job['title']).classes('text-lg font-bold text-white')
                                    ui.label(job['company']).classes('text-sm text-[#A9B6D0]')
                                ui.label(job['location']).classes('text-sm text-[#A9B6D0]')
                                desc = job['description'][:200] + ('...' if len(job['description']) > 200 else '')
                                ui.label(desc).classes('text-sm text-white mt-1')
                                ui.link('View Job', job['url'], new_tab=True).classes('text-[#4FC3F7] hover:text-[#FF8C00]')

                def filter_jobs():
                    """Client‑side filter triggered by filter_input changes."""
                    filter_text = filter_input.value.strip()
                    display_jobs(jobs_data, filter_text)

                async def search_jobs():
                    """Scrape jobs from Wuzzuf/Bayt and display."""
                    query = search_input.value.strip()
                    if not query:
                        ui.notify('Please enter a search term.', type='warning')
                        return
                    location = location_input.value.strip()
                    if location:
                        query += f' {location}'
                    ui.notify(f'Searching for "{query}"...', type='info')
                    results_container.clear()
                    with results_container:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Fetching job listings...').classes('self-center text-sm')

                    # Run synchronous scraper in thread
                    jobs = await run.io_bound(scrape_jobs, query)
                    jobs_data.clear()
                    jobs_data.extend(jobs)
                    display_jobs(jobs_data)  # initial display without filter

        # ---------------- FOOTER (unchanged) ----------------
        ui.html('''
        <div class="app-footer">
            <b>Multi-Standard Engineering Quality Assurance Portal</b> &nbsp;|&nbsp; Automated compliance verification across ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO standards.<br>
            <b>Official Direct Contacts:</b>
            LinkedIn: <a href="https://www.linkedin.com/in/mohamed-abd-al-aty-a326a1214/" target="_blank">Mohamed Abd Al Aty</a> &nbsp;|&nbsp;
            Email: <a href="mailto:mohamedabdalaty63@gmail.com">mohamedabdalaty63@gmail.com</a><br>
            <i>Specialized in QA/QC, Civil Engineering Standards &amp; Automated Compliance.</i> &copy; 2026 Eng. Mohamed Abd Al Aty. All rights reserved.<br>
            <span style="color: #FFFFFF; font-weight: 600;">Disclaimer:</span> These AI modules have high accuracy and are specified for the Egyptian codes, but results should be rechecked by a qualified engineer before any decision-making.
        </div>
        ''')


ui.run(
    host='0.0.0.0',
    port=int(os.environ.get('PORT', 8080)),
    title='Multi-Standard Engineering Auditor',
    favicon='🏗️',
    reload=False,
    reconnect_timeout=30.0,
)
