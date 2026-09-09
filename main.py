import io
import datetime
import os
import uuid
import re
import asyncio
import json
import time
import random
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import qrcode
import pypdf
import fitz  # PyMuPDF
import requests
import cloudscraper
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from io import BytesIO
import ezdxf
from ezdxf.math import Vec2

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
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

GEMINI_MODEL = "gemini-3.5-flash-lite"

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 32
USABLE_WIDTH = PAGE_WIDTH - (2 * MARGIN)

# =====================================================================================
# CODE-COMPLIANCE & TEXT SANITIZATION
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
    body_style = ParagraphStyle("SigBody", fontSize=8, textColor=colors.HexColor("#334155"), leading=11)
    qr_lab_img = ReportLabImage(qr_img_buffer, width=38, height=38)
    sign_text = f"<b>Prepared by Engineer:</b><br/>{engineer_name}<br/><br/>_________________<br/>(Signature &amp; Date)"
    sign_cell = Paragraph(sign_text, body_style)
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
    build_pdf_footer_signature_and_qr(story, styles, qr_buf, meta['engineer'])
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

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
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")

async def call_gemini_json(contents, temperature=0.1, timeout=240):
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
        return response.text
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")

# =====================================================================================
# BOQ CALCULATION ENGINE (unchanged)
# =====================================================================================
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
    new_obj = {}
    for k, v in obj.items():
        if k in aliases:
            new_obj[aliases[k]] = v
        else:
            new_obj[k] = v
    return new_obj

async def extract_architectural_with_ai(element_type, file_bytes, file_type, user_params, code_basis, retry=True):
    contents = []
    schema_info = ARCH_SCHEMAS.get(element_type)
    if not schema_info:
        raise ValueError(f"Unsupported architectural element: {element_type}")

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
        aliases = schema_info.get('field_aliases', {})
        norm_data = normalize_keys(data, aliases)
        return norm_data, raw_response
    except Exception as e:
        if retry:
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
    schema_info = ARCH_SCHEMAS.get(element_type)
    if not schema_info:
        return [], 0, []

    required = schema_info['required']
    results = []
    total_quantity = 0
    missing_fields = []

    for req in required:
        if req not in data or data[req] is None:
            missing_fields.append(req)

    if missing_fields:
        return results, total_quantity, [{'label': 'General', 'idx': 0, 'missing': missing_fields}]

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
    rows = []
    for r in results:
        rows.append({
            'Item': f"{element_type.capitalize()} - {r['label']}",
            'Count': 1,
            'Unit': r['unit'],
            'Quantity (net)': round(r['quantity'], 2),
            'Wastage %': wastage,
            'Quantity (with waste)': round(r['quantity'] * (1 + wastage/100), 2),
            'Unit Rate (EGP)': round(UNIT_RATES.get(r['label'], 0), 2),
            'Total Cost (EGP)': round(r['quantity'] * (1 + wastage/100) * UNIT_RATES.get(r['label'], 0), 2)
        })
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


async def extract_mass_with_ai(element_type, file_bytes, file_type, user_params, code_basis, retry=True):
    contents = []
    schema_info = MASS_SCHEMAS.get(element_type)
    if not schema_info:
        raise ValueError(f"Unsupported element type: {element_type}")

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

    if file_type == 'application/pdf':
        try:
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
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            doc.close()
        except Exception as e:
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        img_part = types.Part.from_bytes(data=file_bytes, mime_type=file_type)
        contents.append(img_part)

    try:
        response_text = await call_gemini_json(contents, temperature=0, timeout=240)
        json_str = response_text.strip()
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        data = json.loads(json_str)
        return data
    except Exception as e:
        if retry:
            prompt2 = f"""
Return a JSON array of objects with fields: {', '.join(schema_info['required'])}.
{extra_instruction}
If the drawing is unclear, return an empty array [].
"""
            contents2 = [prompt2]
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
                return []
        else:
            return []

def compute_mass_from_ai_data(element_type, data, user_params):
    schema_info = MASS_SCHEMAS.get(element_type)
    required = schema_info['required']
    results = []
    total_concrete = 0

    for group in data:
        all_present = True
        for req in required:
            if req not in group or group[req] is None:
                all_present = False
                break
        if not all_present:
            continue

        if element_type == 'columns':
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
# MIME TYPE DETECTION
# =====================================================================================
def detect_mime_type(filename: str, data: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in ['.png']:
        return 'image/png'
    elif ext in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif ext in ['.pdf']:
        return 'application/pdf'
    if data.startswith(b'\x89PNG'):
        return 'image/png'
    if data.startswith(b'\xff\xd8'):
        return 'image/jpeg'
    if data.startswith(b'%PDF'):
        return 'application/pdf'
    return 'image/jpeg'

# =====================================================================================
# JOB SCRAPING FUNCTIONS (unchanged)
# =====================================================================================
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "").strip()
JSEARCH_HOST = "jsearch.p.rapidapi.com"
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://wuzzuf.net/",
    "DNT": "1",
}

def log(msg):
    print(f"[JOB-SCRAPER] {msg}")

scraper = cloudscraper.create_scraper()
scraper.headers.update(HEADERS)

def scrape_jsearch(query, max_results=20):
    jobs = []
    if not RAPIDAPI_KEY:
        log("JSearch: No API key – skipping.")
        return jobs
    params_list = [
        {"query": query, "page": "1", "num_pages": "1", "engine": "google_jobs"},
        {"query": f"{query} Egypt", "page": "1", "num_pages": "1", "engine": "google_jobs"},
        {"query": query, "page": "1", "num_pages": "1", "country": "eg", "engine": "google_jobs"},
    ]
    for params in params_list:
        try:
            resp = requests.get(
                f"https://{JSEARCH_HOST}/search",
                headers={"X-RapidAPI-Key": RAPIDAPI_KEY, "X-RapidAPI-Host": JSEARCH_HOST},
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            log(f"JSearch {params} -> status={resp.status_code}, len={len(resp.text)}")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                if data:
                    for item in data[:max_results]:
                        title = item.get("job_title")
                        if not title:
                            continue
                        company = item.get("employer_name") or "N/A"
                        city = item.get("job_city") or ""
                        country = item.get("job_country") or "Egypt"
                        location = ", ".join(p for p in [city, country] if p) or "Egypt"
                        desc = (item.get("job_description") or "No description")[:400]
                        url = item.get("job_apply_link") or item.get("job_google_link") or ""
                        source = item.get("job_publisher") or "JSearch"
                        if url:
                            jobs.append({
                                "title": title,
                                "company": company,
                                "location": location,
                                "description": desc,
                                "url": url,
                                "source": source,
                            })
                    if jobs:
                        break
        except Exception as e:
            log(f"JSearch exception: {e}")
    log(f"JSearch parsed {len(jobs)} jobs")
    return jobs

def scrape_wuzzuf_direct(query, max_results=20):
    jobs = []
    url = f"https://wuzzuf.net/search/jobs/?q={quote_plus(query)}&a=hpb"
    try:
        resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Wuzzuf GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            log(f"Wuzzuf non-200 body preview: {resp.text[:200]}")
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Wuzzuf request failed: {e}")
        return jobs

    for a in soup.select('a[href*="/jobs/p/"]'):
        href = a.get("href")
        title = a.get_text(strip=True)
        if not href or not title:
            continue
        full_url = href if href.startswith("http") else f"https://wuzzuf.net{href}"
        card = a
        for _ in range(8):
            card = card.parent
            if card is None:
                break
            if len(card.find_all("a")) >= 2:
                break
        company, location, desc = "", "", ""
        if card is not None:
            company_link = card.find("a", href=re.compile(r"/employers/"))
            company = company_link.get_text(strip=True) if company_link else ""
            loc_elem = card.find("span", class_=re.compile(r"location", re.I)) or card.find("div", class_=re.compile(r"location", re.I))
            if loc_elem:
                location = loc_elem.get_text(strip=True)
            chunks = [t.get_text(strip=True) for t in card.find_all(["span", "div"]) if t.get_text(strip=True)]
            chunks = [t for t in chunks if t not in (title, company, location)]
            desc = " | ".join(dict.fromkeys(chunks))[:400]
        jobs.append({
            "title": title,
            "company": company or "N/A",
            "location": location or "Egypt",
            "description": desc or "No description preview.",
            "url": full_url,
            "source": "Wuzzuf",
        })
        if len(jobs) >= max_results:
            break

    if not jobs:
        log(f"Wuzzuf: No job links found. HTML snippet: {re.sub(r'\s+', ' ', str(soup))[:500]}")
    else:
        log(f"Wuzzuf parsed {len(jobs)} jobs")
    return jobs

def scrape_bayt_direct(query, max_results=20):
    jobs = []
    url = f"https://www.bayt.com/en/egypt/jobs/?search={quote_plus(query)}"
    try:
        resp = scraper.get(url, timeout=REQUEST_TIMEOUT)
        log(f"Bayt GET {url} -> status={resp.status_code}, len={len(resp.text)}")
        if resp.status_code != 200:
            log(f"Bayt non-200 body preview: {resp.text[:200]}")
            return jobs
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log(f"Bayt request failed: {e}")
        return jobs

    cards = soup.select('li.has-pointer') or soup.select('div.job-card')
    for card in cards[:max_results]:
        try:
            a = card.find("h2") and card.find("h2").find("a")
            if not a:
                a = card.find("a", href=re.compile(r"/job/"))
            if not a:
                continue
            title = a.get_text(strip=True)
            href = a.get("href")
            full_url = href if href.startswith("http") else f"https://www.bayt.com{href}"
            company_el = card.select_one(".company-name, .jb-company")
            company = company_el.get_text(strip=True) if company_el else "N/A"
            loc_el = card.select_one(".location, .t-mute.t-small")
            location = loc_el.get_text(strip=True) if loc_el else "Egypt"
            desc_el = card.select_one("p")
            desc = desc_el.get_text(strip=True) if desc_el else "No description."
            jobs.append({
                "title": title,
                "company": company,
                "location": location,
                "description": desc,
                "url": full_url,
                "source": "Bayt",
            })
        except Exception as e:
            log(f"Bayt card parse error: {e}")
    if not jobs:
        log(f"Bayt: No job cards found. HTML snippet: {re.sub(r'\s+', ' ', str(soup))[:500]}")
    else:
        log(f"Bayt parsed {len(jobs)} jobs")
    return jobs

def scrape_jobs(query, location=""):
    full_query = f"{query} {location}".strip() if location else query
    all_jobs = []
    try:
        all_jobs.extend(scrape_jsearch(full_query))
    except Exception as e:
        log(f"JSearch top-level error: {e}")
    if len(all_jobs) < 3:
        log("JSearch returned few results – trying Wuzzuf.")
        try:
            all_jobs.extend(scrape_wuzzuf_direct(full_query))
        except Exception as e:
            log(f"Wuzzuf top-level error: {e}")
    if len(all_jobs) < 3:
        log("Wuzzuf also returned few results – trying Bayt.")
        try:
            all_jobs.extend(scrape_bayt_direct(full_query))
        except Exception as e:
            log(f"Bayt top-level error: {e}")
    seen = set()
    deduped = []
    for job in all_jobs:
        if job["url"] in seen:
            continue
        seen.add(job["url"])
        deduped.append(job)
    deduped.sort(key=lambda j: 0 if j["source"] == "Wuzzuf" else 1)
    log(f"TOTAL jobs after dedup: {len(deduped)}")
    return deduped

# =====================================================================================
# PROGRESS TRACKER FUNCTIONS (unchanged)
# =====================================================================================
def parse_progress_from_gemini_response(raw_text):
    raw_text = raw_text.strip()
    raw_text = re.sub(r'^```json\s*', '', raw_text)
    raw_text = re.sub(r'\s*```$', '', raw_text)
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1:
        raw_text = raw_text[start:end+1]
    try:
        return json.loads(raw_text)
    except:
        return {}

async def extract_progress_from_image(file_bytes, file_type):
    if not client:
        raise Exception("GEMINI_API_KEY missing.")
    prompt = """
You are a construction project progress reporter. Analyze the provided image (which may show site photos, documents, or progress reports).
Extract the following information and return a JSON object with these fields (if visible):
- "date": date in YYYY-MM-DD format (if any text states a date)
- "description": a brief description of the work shown (e.g., "Concrete pour for foundation", "Steel fixing for column")
- "progress_percent": estimated progress percentage (0-100) for the activity shown
- "category": category of work (e.g., "Concrete", "Steel", "Finishing", "Excavation", "MEP")
- "location": specific location if mentioned (e.g., "Block A", "Floor 2")

If any field is not visible, set it to null.
Return ONLY the JSON, no extra text.
Example: {"date": "2026-03-15", "description": "Formwork installation for slab", "progress_percent": 60, "category": "Formwork", "location": "Block B"}
"""
    contents = [prompt]
    if file_type == 'application/pdf':
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            doc.close()
        except:
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=file_type))
    raw_response = await call_gemini_json(contents, temperature=0, timeout=120)
    data = parse_progress_from_gemini_response(raw_response)
    return data

def process_excel_file(file_bytes, filename):
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        df.columns = df.columns.str.lower().str.strip()
        progress_cols = [c for c in df.columns if 'progress' in c or 'percent' in c]
        date_cols = [c for c in df.columns if 'date' in c]
        desc_cols = [c for c in df.columns if 'desc' in c or 'note' in c]
        cat_cols = [c for c in df.columns if 'cat' in c or 'type' in c]
        loc_cols = [c for c in df.columns if 'loc' in c or 'area' in c]
        date_col = date_cols[0] if date_cols else None
        desc_col = desc_cols[0] if desc_cols else None
        progress_col = progress_cols[0] if progress_cols else None
        cat_col = cat_cols[0] if cat_cols else None
        loc_col = loc_cols[0] if loc_cols else None
        keep_cols = []
        if date_col: keep_cols.append(date_col)
        if desc_col: keep_cols.append(desc_col)
        if progress_col: keep_cols.append(progress_col)
        if cat_col: keep_cols.append(cat_col)
        if loc_col: keep_cols.append(loc_col)
        if keep_cols:
            df = df[keep_cols]
        rename_map = {}
        if date_col: rename_map[date_col] = 'date'
        if desc_col: rename_map[desc_col] = 'description'
        if progress_col: rename_map[progress_col] = 'progress_percent'
        if cat_col: rename_map[cat_col] = 'category'
        if loc_col: rename_map[loc_col] = 'location'
        df = df.rename(columns=rename_map)
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        if 'progress_percent' in df.columns:
            df['progress_percent'] = pd.to_numeric(df['progress_percent'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return pd.DataFrame()

async def generate_progress_overview(df, start_date, end_date, description):
    if df.empty or not client:
        return "No data available for overview."
    csv_data = df.to_csv(index=False)
    prompt = f"""
You are a project management analyst. Given the following progress data for a construction project from {start_date} to {end_date}, 
with description: "{description}", provide a concise executive summary (2-3 paragraphs) highlighting:
- Overall progress status
- Key activities and their progress
- Any potential risks or delays
- Recommendations for next steps

Data (CSV format):
{csv_data}
"""
    try:
        overview = await call_gemini(prompt, temperature=0.3, timeout=120)
        return overview
    except Exception as e:
        return f"Error generating overview: {str(e)}"

# =====================================================================================
# DXF FUNCTIONS (FIXED)
# =====================================================================================
def load_dxf_doc(file_bytes):
    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode('utf-8')
    try:
        return ezdxf.recover.readbytes(file_bytes)[0]
    except Exception:
        try:
            return ezdxf.read(io.TextIOWrapper(io.BytesIO(file_bytes), encoding='utf-8', errors='ignore'))
        except Exception as e:
            try:
                return ezdxf.read(io.BytesIO(file_bytes))
            except Exception as e2:
                raise Exception(f"Error reading DXF: {str(e2)}")

def detect_dxf_layers(doc):
    layers = {}
    try:
        for entity in doc.modelspace():
            layer = entity.dxf.layer
            if layer not in layers:
                layers[layer] = {'count': 0, 'types': set(), 'keywords': []}
            layers[layer]['count'] += 1
            layers[layer]['types'].add(entity.dxftype())
    except Exception as e:
        print(f"Error detecting layers: {e}")
        return {}
    for layer in layers:
        lc = layer.lower()
        keywords = []
        if 'wall' in lc: keywords.append('wall')
        if 'column' in lc: keywords.append('column')
        if 'beam' in lc: keywords.append('beam')
        if 'slab' in lc: keywords.append('slab')
        if 'footing' in lc or 'foundation' in lc: keywords.append('foundation')
        if 'room' in lc: keywords.append('room')
        if 'door' in lc: keywords.append('door')
        if 'window' in lc: keywords.append('window')
        if 'area' in lc: keywords.append('area')
        layers[layer]['keywords'] = keywords
    return layers

def extract_areas_from_dxf(doc, unit='mm', workflow='architectural'):
    if unit == 'mm':
        area_scale = 1e-6
    elif unit == 'cm':
        area_scale = 1e-4
    else:
        area_scale = 1.0

    results = []
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
            if entity.closed:
                try:
                    if entity.dxftype() == 'LWPOLYLINE':
                        points = [(p.x, p.y) for p in entity.get_points()]
                    else:
                        points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                    area = 0.0
                    for i in range(len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[(i+1) % len(points)]
                        area += x1*y2 - x2*y1
                    area = abs(area) / 2.0
                    label = ""
                    centroid_x = sum(p[0] for p in points) / len(points)
                    centroid_y = sum(p[1] for p in points) / len(points)
                    for txt in msp.query('TEXT MTEXT'):
                        if txt.dxftype() == 'TEXT':
                            pos = txt.dxf.insert
                        else:
                            pos = txt.dxf.insert
                        if abs(pos.x - centroid_x) < 10 and abs(pos.y - centroid_y) < 10:
                            label = txt.dxf.text
                            break
                    raw_area = area * area_scale
                    results.append({
                        'layer': entity.dxf.layer,
                        'area_m2': round(raw_area, 4),
                        'label': label.strip() if label else '',
                        'vertices': len(points)
                    })
                except Exception as e:
                    print(f"Skipping entity: {e}")
                    continue
    return results

# =====================================================================================
# AUTOCAD LAYOUT GENERATOR (NEW TOOL)
# =====================================================================================
def generate_autocad_layout(plot_area_m2, street_width_m, location):
    """
    Generate a simple 2D floor plan DXF and BOQ based on Egyptian code.
    Returns: (dxf_bytes, boq_data, layout_info)
    """
    footprint_ratio = 0.6
    max_footprint = plot_area_m2 * footprint_ratio
    if street_width_m >= 12:
        max_floors = 4
    elif street_width_m >= 8:
        max_floors = 3
    elif street_width_m >= 6:
        max_floors = 2
    else:
        max_floors = 1

    plot_width = 10.0
    plot_length = plot_area_m2 / plot_width

    front_setback = 3.0
    rear_setback = 2.0
    side_setback = 1.5

    building_width = plot_width - 2 * side_setback
    building_length = plot_length - front_setback - rear_setback

    building_area = building_width * building_length
    if building_area > max_footprint:
        scale = (max_footprint / building_area) ** 0.5
        building_width *= scale
        building_length *= scale
        building_area = building_width * building_length

    if building_width < 5: building_width = 5
    if building_length < 5: building_length = 5

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    x0 = side_setback
    y0 = front_setback
    msp.add_lwpolyline([(x0, y0), (x0+building_width, y0), (x0+building_width, y0+building_length), (x0, y0+building_length)], close=True, dxfattribs={'layer': 'WALLS'})

    wall_y = y0 + 0.4 * building_length
    msp.add_line((x0, wall_y), (x0 + building_width, wall_y), dxfattribs={'layer': 'WALLS'})

    wall_x = x0 + 0.6 * building_width
    msp.add_line((wall_x, wall_y), (wall_x, y0 + building_length), dxfattribs={'layer': 'WALLS'})

    wall_x2 = x0 + 0.3 * building_width
    msp.add_line((wall_x2, y0), (wall_x2, wall_y), dxfattribs={'layer': 'WALLS'})

    msp.add_text("LIVING", dxfattribs={'height': 0.3, 'insert': (x0 + 0.15*building_width, y0 + 0.2*building_length)})
    msp.add_text("KITCHEN", dxfattribs={'height': 0.3, 'insert': (x0 + 0.45*building_width, wall_y + 0.2*(building_length-wall_y))})
    msp.add_text("BEDROOM 1", dxfattribs={'height': 0.3, 'insert': (x0 + 0.7*building_width, wall_y + 0.3*(building_length-wall_y))})
    msp.add_text("BEDROOM 2", dxfattribs={'height': 0.3, 'insert': (x0 + 0.7*building_width, wall_y + 0.6*(building_length-wall_y))})
    msp.add_text("BATH", dxfattribs={'height': 0.3, 'insert': (x0 + 0.15*building_width, wall_y + 0.6*(building_length-wall_y))})

    dxf_buffer = io.StringIO()
    doc.write(dxf_buffer)
    dxf_bytes = dxf_buffer.getvalue().encode('utf-8')

    floor_height = 3.0
    concrete_per_floor = building_area * 0.2
    total_concrete = concrete_per_floor * max_floors
    rebar_kg = total_concrete * 100
    wall_length = 2*(building_width+building_length) + (building_length) + (building_width) + (building_width*0.3)
    wall_height = floor_height * max_floors
    wall_area = wall_length * wall_height
    brick_count = wall_area * 50
    flooring_area = building_area * max_floors
    paint_area = wall_area * 2

    boq_data = {
        'Item': ['Concrete (m³)', 'Rebar (kg)', 'Bricks (nos)', 'Flooring (m²)', 'Paint (m²)'],
        'Quantity': [round(total_concrete, 2), round(rebar_kg, 2), round(brick_count, 0), round(flooring_area, 2), round(paint_area, 2)],
        'Unit': ['m³', 'kg', 'nos', 'm²', 'm²']
    }
    rates = {
        'Concrete (m³)': 2500,
        'Rebar (kg)': 15,
        'Bricks (nos)': 2.5,
        'Flooring (m²)': 150,
        'Paint (m²)': 30
    }
    boq_data['Unit Rate (EGP)'] = [rates.get(item, 0) for item in boq_data['Item']]
    boq_data['Total Cost (EGP)'] = [round(boq_data['Quantity'][i] * boq_data['Unit Rate (EGP)'][i], 2) for i in range(len(boq_data['Item']))]

    layout_info = {
        'plot_area': plot_area_m2,
        'street_width': street_width_m,
        'location': location,
        'max_floors': max_floors,
        'footprint_area': round(building_area, 2),
        'building_width': round(building_width, 2),
        'building_length': round(building_length, 2),
    }

    return dxf_bytes, pd.DataFrame(boq_data), layout_info

# =====================================================================================
# STYLING - MODERN & PROFESSIONAL
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
    .output-card {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        width: 100% !important;
        max-width: none !important;
        box-sizing: border-box;
    }
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
# MAIN APP LAYOUT
# =====================================================================================
@ui.page('/')
def main_page():
    ui.query('body').style('width: 100vw; height: 100vh; overflow-x: hidden;')

    # ---------------- SIDEBAR ----------------
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
        # Title block
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

        # Tabs
        with ui.tabs().classes('w-full text-white bg-[#0d1a35] rounded-lg') as tabs:
            t_dash = ui.tab('Concrete Cube Verifier').classes('text-white font-bold')
            t_audit = ui.tab('AI Multi-Standard Auditor').classes('text-white font-bold')
            t_defect = ui.tab('Defect Diagnostic').classes('text-white font-bold')
            t_chat = ui.tab('AI Chatbot').classes('text-white font-bold')
            t_handwriting = ui.tab('Handwriting OCR').classes('text-white font-bold')
            t_jobs = ui.tab('Job Board').classes('text-white font-bold')
            t_progress = ui.tab('Progress Tracker').classes('text-white font-bold')
            t_dxf = ui.tab('DXF Area Extractor').classes('text-white font-bold')
            t_autocad = ui.tab('AutoCAD Layout Generator').classes('text-white font-bold')

        with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent mt-4'):

            # --------------------------------------------------------------
            # TAB 1: CONCRETE CUBE VERIFIER
            # --------------------------------------------------------------
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

            # --------------------------------------------------------------
            # TAB 8: DXF AREA EXTRACTOR
            # --------------------------------------------------------------
            with ui.tab_panel(t_dxf):
                ui.label('DXF Area & Geometry Extractor').classes('text-2xl font-bold text-white mb-4')
                ui.label('Upload a CAD drawing (.dxf) to automatically detect closed polylines, layers, and calculate areas.').classes('text-sm text-[#A9B6D0] mb-4')
                
                dxf_result_container = ui.column().classes('w-full')
                dxf_bytes_holder = {'bytes': None}

                async def handle_dxf_upload(e):
                    try:
                        dxf_bytes_holder['bytes'] = await e.file.read()
                        ui.notify('DXF file loaded successfully!', type='positive')
                    except Exception as ex:
                        ui.notify(f'Error reading DXF file: {str(ex)}', type='negative')

                ui.upload(label='Upload CAD File (.dxf)', auto_upload=True, on_upload=handle_dxf_upload).props('flat dark accept=.dxf').classes('w-full mb-4')

                unit_select = ui.select(label='Drawing Unit', options=['mm', 'cm', 'm'], value='mm').classes('w-full mb-4')

                def run_dxf_extraction():
                    dxf_result_container.clear()
                    if not dxf_bytes_holder['bytes']:
                        ui.notify('Please upload a DXF file first.', type='warning')
                        return
                    try:
                        doc = load_dxf_doc(dxf_bytes_holder['bytes'])
                        areas = extract_areas_from_dxf(doc, unit=unit_select.value)
                        with dxf_result_container:
                            if not areas:
                                ui.label('No closed polylines found in the DXF file.').classes('text-amber-400')
                            else:
                                ui.label(f'Found {len(areas)} closed areas:').classes('text-lg font-bold text-white mb-2')
                                df_areas = pd.DataFrame(areas)
                                ui.table.from_pandas(df_areas).classes('w-full')
                    except Exception as ex:
                        with dxf_result_container:
                            ui.label(f'Error reading DXF: {str(ex)}').classes('text-red-400')

                ui.button('Extract Areas & Layers', on_click=run_dxf_extraction).classes('primary-btn mb-4')

            # --------------------------------------------------------------
            # TAB 9: AUTOCAD LAYOUT GENERATOR
            # --------------------------------------------------------------
            with ui.tab_panel(t_autocad):
                ui.label('AutoCAD Layout & BOQ Generator').classes('text-2xl font-bold text-white mb-4')
                ui.label('Generate a code-compliant 2D floor plan DXF and full BOQ based on plot dimensions and Egyptian regulations.').classes('text-sm text-[#A9B6D0] mb-4')

                with ui.row().classes('w-full gap-4 mb-4'):
                    plot_area_input = ui.number(label='Plot Area (m²)', value=200.0, step=10.0).classes('flex-1')
                    street_width_input = ui.number(label='Street Width (m)', value=10.0, step=1.0).classes('flex-1')
                    layout_location_input = ui.input(label='Location / District', value='Cairo').classes('flex-1')

                autocad_result_container = ui.column().classes('w-full')
                generated_dxf_holder = {'bytes': None}

                def run_autocad_generator():
                    autocad_result_container.clear()
                    try:
                        p_area = float(plot_area_input.value or 0)
                        s_width = float(street_width_input.value or 0)
                        loc = layout_location_input.value
                        if p_area <= 0:
                            ui.notify('Please enter a valid plot area.', type='warning')
                            return
                        
                        dxf_bytes, df_boq, info = generate_autocad_layout(p_area, s_width, loc)
                        generated_dxf_holder['bytes'] = dxf_bytes

                        with autocad_result_container:
                            ui.label(f'Generated Layout Information (Max Floors: {info["max_floors"]}, Footprint: {info["footprint_area"]} m²)').classes('text-lg font-bold text-[#FF8C00] mb-2')
                            ui.table.from_pandas(df_boq).classes('w-full mb-4')
                            
                            ui.download(dxf_bytes, filename='generated_layout.dxf', label='Download Generated DXF').classes('primary-btn')
                            ui.notify('Layout and BOQ generated successfully!', type='positive')
                    except Exception as ex:
                        with autocad_result_container:
                            ui.label(f'Error generating layout: {str(ex)}').classes('text-red-400')

                ui.button('Generate Layout & BOQ', on_click=run_autocad_generator).classes('primary-btn mb-4')

    ui.html('''
    <div class="app-footer">
        Smart Egy-Civil AI Auditor &copy; 2026 &nbsp;|&nbsp; Built with Fast AI &amp; NiceGUI &nbsp;|&nbsp; 
        <a href="https://egy-civil.com" target="_blank">Documentation &amp; Standards Portal</a>
    </div>
    ''')

ui.run(port=8080, host='0.0.0.0', title='Smart Egy-Civil AI Auditor', favicon='👷‍♂️')
