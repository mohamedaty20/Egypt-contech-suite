import io
import datetime
import os
import uuid
import re
import asyncio
import json
import traceback
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import qrcode
import pypdf
import fitz
import ezdxf
from nicegui import app, ui, run
from config import CODE_BASIS_OPTIONS, get_code_directive, NO_LATEX_RULE, client, fcu_input, etc.
# ... actually, we need to import all necessary functions from services and utils
from services.ai_service import call_gemini, call_gemini_json, extract_architectural_with_ai, extract_mass_with_ai, extract_progress_from_image, generate_progress_overview
from services.pdf_service import sanitize_ai_markdown, build_pdf_styles, markdown_to_pdf_flowables, generate_qr_code, build_report_pdf, build_pdf_footer_signature_and_qr
from services.dxf_service import detect_dxf_layers, extract_areas_from_dxf, build_complete_project
from services.scraper_service import scrape_jobs, detect_mime_type
from utils.boq import normalize_keys, compute_architectural_quantities, generate_arch_boq_table, compute_mass_from_ai_data, generate_boq_table, compute_rebar_quantities, generate_charts

# ---- Helper functions for progress & dxf pdf reports (they were at the end of the original) ----
def generate_progress_pdf(pdf_data, engineer_name, project_name, logo_bytes, ticket_id):
    from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, HRFlowable, Image as ReportLabImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from config import USABLE_WIDTH, MARGIN, PAGE_WIDTH, PAGE_HEIGHT
    from services.pdf_service import build_pdf_styles, generate_qr_code, build_pdf_footer_signature_and_qr, markdown_to_pdf_flowables
    import io, uuid
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=PAGE_WIDTH, rightMargin=MARGIN, leftMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN)
    styles = build_pdf_styles()
    story = []

    unique_uid = f"PROGRESS-{uuid.uuid4().hex[:8].upper()}"
    qr_buf = generate_qr_code(f"UID: {unique_uid} | Progress Report - {project_name}")

    title_style = ParagraphStyle("DocTitle", fontSize=14, textColor=colors.HexColor("#1B2A4A"),
                                  spaceAfter=3, fontName="Helvetica-Bold", leading=17)
    sub_style = ParagraphStyle("DocSub", fontSize=9, textColor=colors.HexColor("#B45309"),
                                spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", fontSize=8, textColor=colors.HexColor("#334155"),
                                 leading=11.5, fontName="Helvetica")

    company_name = "Smart Egypt Civil AI"
    start_str = pdf_data['start_date'].strftime('%Y-%m-%d') if pdf_data['start_date'] else 'N/A'
    end_str = pdf_data['end_date'].strftime('%Y-%m-%d') if pdf_data['end_date'] else 'N/A'
    meta_html = f"""
    <b>Company:</b> {company_name} &nbsp;|&nbsp; <b>Project:</b> {project_name}<br/>
    <b>Engineer in Charge:</b> {engineer_name} &nbsp;|&nbsp; <b>Date Range:</b> {start_str} to {end_str}<br/>
    <b>Phase:</b> {pdf_data['description']}<br/>
    <b>Report UID:</b> <font color="#CC0000"><b>{unique_uid}</b></font>
    """
    right_cell = ReportLabImage(io.BytesIO(logo_bytes), width=70, height=32) if logo_bytes else ""
    try:
        header_table_data = [
            [Paragraph(f"<b>PROGRESS TRACKING REPORT</b>", title_style), right_cell],
            [Paragraph("Consolidated Progress Summary", sub_style), ""],
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
        story.append(Paragraph("PROGRESS TRACKING REPORT", title_style))
        story.append(Paragraph("Consolidated Progress Summary", sub_style))
        story.append(Paragraph(meta_html, meta_style))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor("#FF8C00"), spaceAfter=8))

    df = pdf_data['df'].copy()
    if not df.empty:
        cols_to_show = [col for col in df.columns if col in ['date', 'description', 'progress_percent', 'category', 'location']]
        if 'source' in df.columns:
            cols_to_show.append('source')
        df_display = df[cols_to_show].fillna('')
        if 'date' in df_display.columns:
            df_display['date'] = df_display['date'].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
        table_data = [cols_to_show]
        for _, row in df_display.iterrows():
            table_data.append([str(row[col]) for col in cols_to_show])
        if len(table_data) > 20:
            table_data = table_data[:20]
        col_widths = [USABLE_WIDTH / len(cols_to_show)] * len(cols_to_show)
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B2A4A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    for fig in [pdf_data['fig_bar'], pdf_data['fig_scatter'], pdf_data['fig_pie'], pdf_data['fig_line']]:
        if fig:
            try:
                img_bytes = fig.to_image(format="png", width=400, height=300, scale=2)
                img_flowable = ReportLabImage(io.BytesIO(img_bytes), width=USABLE_WIDTH*0.45, height=USABLE_WIDTH*0.45*0.75)
                story.append(img_flowable)
                story.append(Spacer(1, 6))
            except Exception as e:
                print(f"Could not embed chart: {e}")

    if pdf_data['overview']:
        story.append(Paragraph("AI Overview", styles['h2']))
        story.extend(markdown_to_pdf_flowables(pdf_data['overview'], styles))
        story.append(Spacer(1, 6))

    build_pdf_footer_signature_and_qr(story, styles, qr_buf, engineer_name)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_dxf_pdf(df, total_area, filename, workflow, units, engineer_name, project_name, logo_bytes, ticket_id):
    from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, HRFlowable, Image as ReportLabImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from config import USABLE_WIDTH, MARGIN, PAGE_WIDTH, PAGE_HEIGHT
    from services.pdf_service import build_pdf_styles, generate_qr_code, build_pdf_footer_signature_and_qr
    import io, uuid
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=PAGE_WIDTH, rightMargin=MARGIN, leftMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN)
    styles = build_pdf_styles()
    story = []

    unique_uid = f"DXF-{uuid.uuid4().hex[:8].upper()}"
    qr_buf = generate_qr_code(f"UID: {unique_uid} | DXF Area Report - {project_name}")

    title_style = ParagraphStyle("DocTitle", fontSize=14, textColor=colors.HexColor("#1B2A4A"),
                                  spaceAfter=3, fontName="Helvetica-Bold", leading=17)
    sub_style = ParagraphStyle("DocSub", fontSize=9, textColor=colors.HexColor("#B45309"),
                                spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", fontSize=8, textColor=colors.HexColor("#334155"),
                                 leading=11.5, fontName="Helvetica")

    company_name = "Smart Egypt Civil AI"
    meta_html = f"""
    <b>Company:</b> {company_name} &nbsp;|&nbsp; <b>Project:</b> {project_name}<br/>
    <b>Engineer in Charge:</b> {engineer_name} &nbsp;|&nbsp; <b>File:</b> {filename}<br/>
    <b>Workflow:</b> {workflow} &nbsp;|&nbsp; <b>Units:</b> {units}<br/>
    <b>Report UID:</b> <font color="#CC0000"><b>{unique_uid}</b></font>
    """
    right_cell = ReportLabImage(io.BytesIO(logo_bytes), width=70, height=32) if logo_bytes else ""
    try:
        header_table_data = [
            [Paragraph(f"<b>DXF AREA EXTRACTION REPORT</b>", title_style), right_cell],
            [Paragraph("Area Takeoff from DXF Drawing", sub_style), ""],
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
        story.append(Paragraph("DXF AREA EXTRACTION REPORT", title_style))
        story.append(Paragraph("Area Takeoff from DXF Drawing", sub_style))
        story.append(Paragraph(meta_html, meta_style))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor("#FF8C00"), spaceAfter=8))

    if not df.empty:
        cols_to_show = ['layer', 'label', 'area_m2', 'vertices']
        table_data = [cols_to_show]
        for _, row in df.iterrows():
            table_data.append([str(row[col]) for col in cols_to_show])
        if len(table_data) > 20:
            table_data = table_data[:20]
        col_widths = [USABLE_WIDTH / len(cols_to_show)] * len(cols_to_show)
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B2A4A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        total_para = Paragraph(f"<b>Total Net Area: {total_area:.4f} m²</b>", styles['h2'])
        story.append(total_para)
        story.append(Spacer(1, 6))

    build_pdf_footer_signature_and_qr(story, styles, qr_buf, engineer_name)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_autocad_pdf(info, boq_df, engineer_name, project_name, logo_bytes, ticket_id):
    from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, HRFlowable, Image as ReportLabImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from config import USABLE_WIDTH, MARGIN, PAGE_WIDTH, PAGE_HEIGHT
    from services.pdf_service import build_pdf_styles, generate_qr_code, build_pdf_footer_signature_and_qr
    import io, uuid
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=PAGE_WIDTH, rightMargin=MARGIN, leftMargin=MARGIN,
                             topMargin=MARGIN, bottomMargin=MARGIN)
    styles = build_pdf_styles()
    story = []

    unique_uid = f"LAYOUT-{uuid.uuid4().hex[:8].upper()}"
    qr_buf = generate_qr_code(f"UID: {unique_uid} | Layout Report - {project_name}")

    title_style = ParagraphStyle("DocTitle", fontSize=14, textColor=colors.HexColor("#1B2A4A"),
                                  spaceAfter=3, fontName="Helvetica-Bold", leading=17)
    sub_style = ParagraphStyle("DocSub", fontSize=9, textColor=colors.HexColor("#B45309"),
                                spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", fontSize=8, textColor=colors.HexColor("#334155"),
                                 leading=11.5, fontName="Helvetica")

    company_name = "Smart Egypt Civil AI"
    meta_html = f"""
    <b>Company:</b> {company_name} &nbsp;|&nbsp; <b>Project:</b> {project_name}<br/>
    <b>Engineer in Charge:</b> {engineer_name}<br/>
    <b>Plot Area:</b> {info['plot_area']} m² &nbsp;|&nbsp; <b>Street Width:</b> {info['street_width']} m<br/>
    <b>Location:</b> {info['location']} &nbsp;|&nbsp; <b>Max Floors:</b> {info['max_floors']} (used {info['num_floors']})<br/>
    <b>Report UID:</b> <font color="#CC0000"><b>{unique_uid}</b></font>
    """
    right_cell = ReportLabImage(io.BytesIO(logo_bytes), width=70, height=32) if logo_bytes else ""
    try:
        header_table_data = [
            [Paragraph(f"<b>AUTOCAD LAYOUT REPORT</b>", title_style), right_cell],
            [Paragraph("Generated Floor Plan & BOQ", sub_style), ""],
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
        story.append(Paragraph("AUTOCAD LAYOUT REPORT", title_style))
        story.append(Paragraph("Generated Floor Plan & BOQ", sub_style))
        story.append(Paragraph(meta_html, meta_style))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor("#FF8C00"), spaceAfter=8))

    desc = f"Footprint: {info['footprint_area']} m², Building dimensions: {info['building_width']:.2f} x {info['building_length']:.2f} m"
    story.append(Paragraph(desc, styles['body']))
    story.append(Spacer(1, 6))

    if not boq_df.empty:
        cols_to_show = ['Item', 'Quantity', 'Unit', 'Unit Rate (EGP)', 'Total Cost (EGP)']
        table_data = [cols_to_show]
        for _, row in boq_df.iterrows():
            table_data.append([str(row[col]) for col in cols_to_show])
        col_widths = [USABLE_WIDTH / len(cols_to_show)] * len(cols_to_show)
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B2A4A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
            ('FONTSIZE',
