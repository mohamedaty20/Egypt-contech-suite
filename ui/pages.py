# ui/pages.py – complete file with all tools restored
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

# ----- Imports from our modules -----
from config import (
    CODE_BASIS_OPTIONS,
    get_code_directive,
    NO_LATEX_RULE,
    client,
    sanitize_ai_markdown,
    USABLE_WIDTH,
    MARGIN,
    PAGE_WIDTH,
    PAGE_HEIGHT,
    UNIT_RATES,
)
from services.ai_service import (
    call_gemini,
    call_gemini_json,
    extract_architectural_with_ai,
    extract_mass_with_ai,
    extract_progress_from_image,
    generate_progress_overview,
)
from services.pdf_service import (
    build_pdf_styles,
    markdown_to_pdf_flowables,
    generate_qr_code,
    build_report_pdf,
    build_pdf_footer_signature_and_qr,
)
from services.dxf_service import (
    detect_dxf_layers,
    extract_areas_from_dxf,
    build_complete_project,
)
from services.scraper_service import scrape_jobs, detect_mime_type
from utils.boq import (
    normalize_keys,
    compute_architectural_quantities,
    generate_arch_boq_table,
    compute_mass_from_ai_data,
    generate_boq_table,
    compute_rebar_quantities,
    generate_charts,
)

# ----- Helper functions (originally at the end of the monolith) -----

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

def generate_progress_pdf(pdf_data, engineer_name, project_name, logo_bytes, ticket_id):
    from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, HRFlowable, Image as ReportLabImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
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
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        total_cost = boq_df['Total Cost (EGP)'].sum()
        total_para = Paragraph(f"<b>Total Estimated Cost: {total_cost:,.2f} EGP</b>", styles['h2'])
        story.append(total_para)
        story.append(Spacer(1, 6))

    build_pdf_footer_signature_and_qr(story, styles, qr_buf, engineer_name)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------------------------------------------------------------
# Main NiceGUI Page – all 9 tabs restored
# ----------------------------------------------------------------------
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

            # ===== TAB 1: CONCRETE CUBE VERIFIER =====
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

            # ===== TAB 2: AI MULTI-STANDARD AUDITOR =====
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

            # ===== TAB 3: DEFECT DIAGNOSTIC =====
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

            # ===== TAB 4: AI CHATBOT =====
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

            # ===== TAB 5: HANDWRITING OCR =====
            with ui.tab_panel(t_handwriting):
                ui.label('Handwriting to Digital Text Transcription').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload a scanned handwritten note (PNG, JPG) or PDF. The AI will convert it to clean digital text, detecting tables if present.').classes('markdown-body mb-2')
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
                ocr_output = ui.column().classes('w-full')
                ocr_export = ui.row().classes('w-full gap-4 mt-4')
                transcribed_text_holder = {'text': ''}
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
                        prompt = """
You are an expert OCR system. Transcribe the handwritten text from the provided image(s).
- If you detect any tabular data (rows and columns), format it as a proper Markdown table with a header row and a separator line (|---|...|).
- Return only the transcribed text and tables, without any additional commentary, headings, or formatting.
- If there are multiple pages, combine them in order.
"""
                        contents = [prompt]
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
                        response_text = await call_gemini(contents, temperature=0, timeout=240)
                        transcribed = sanitize_ai_markdown(response_text)
                        transcribed_text_holder['text'] = transcribed
                        ocr_output.clear()
                        with ocr_output:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Transcribed Text (editable)').classes('text-xl font-bold text-white mb-2')
                                text_editor = ui.textarea(value=transcribed, placeholder='Edit the transcribed text here...').classes('w-full markdown-body').style('min-height: 300px; background: #0a1a3a; color: white; border: 1px solid #FF8C00;')
                                ui.label('Preview:').classes('text-lg font-bold text-white mt-2')
                                preview_container = ui.column().classes('w-full')
                                def update_preview():
                                    preview_container.clear()
                                    with preview_container:
                                        ui.markdown(text_editor.value).classes('markdown-body')
                                text_editor.on('input', update_preview)
                                update_preview()
                        with ocr_export:
                            def download_ocr_pdf():
                                try:
                                    current_text = text_editor.value if text_editor else transcribed_text_holder['text']
                                    meta = current_meta('OCR')
                                    pdf_bytes = build_report_pdf(
                                        doc_title="",
                                        subtitle="",
                                        body_markdown=current_text,
                                        meta=meta,
                                        logo_bytes=logo_bytes_holder['bytes'],
                                        show_ticket=False
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
                with ocr_output:
                    ui.markdown('*Upload a file and click "Transcribe Handwriting" to start.*').classes('text-sm text-[#A9B6D0]')

            # ===== TAB 6: JOB BOARD =====
            with ui.tab_panel(t_jobs):
                ui.label('Engineering Job Board - Egypt').classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Search for the latest engineering jobs in Egypt. Uses **JSearch** (RapidAPI) if the key is set, otherwise falls back to direct Wuzzuf and Bayt scraping with Cloudflare bypass.').classes('markdown-body mb-2')
                key_status = ui.label(
                    '🔑 RapidAPI key: ' + ('✅ Set' if RAPIDAPI_KEY else '❌ Not set – using Wuzzuf/Bayt fallback.')
                ).classes('text-sm text-[#A9B6D0] mb-2')
                debug_output = ui.label('Debug: waiting for search...').classes('text-xs text-[#A9B6D0] mb-2')
                with ui.row().classes('w-full gap-4 mb-4'):
                    search_input = ui.input(label='Search for jobs', placeholder='e.g., Civil Engineer', value='Civil Engineer').classes('flex-1')
                    location_input = ui.input(label='Location (optional)', placeholder='e.g., Cairo').classes('flex-1')
                    search_button = ui.button('Search Jobs', on_click=lambda: search_jobs()).classes('primary-btn')
                filter_input = ui.input(label='Filter results', placeholder='Type to filter title, company, description...', on_change=lambda: filter_jobs()).classes('w-full mb-2')
                results_container = ui.column().classes('w-full')
                jobs_data = []
                def display_jobs(jobs, filter_text=''):
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
                    display_jobs(jobs_data, filter_input.value.strip())
                async def search_jobs():
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
                    jobs = await run.io_bound(scrape_jobs, query)
                    jobs_data.clear()
                    jobs_data.extend(jobs)
                    counts = {}
                    for j in jobs:
                        counts[j['source']] = counts.get(j['source'], 0) + 1
                    debug_info = f"JSearch: {counts.get('JSearch', 0)}, Wuzzuf: {counts.get('Wuzzuf', 0)}, Bayt: {counts.get('Bayt', 0)} | Total: {len(jobs)}"
                    debug_output.set_text(f'Debug: {debug_info}')
                    if jobs:
                        debug_output.classes(replace='text-xs text-emerald-400 mb-2')
                    else:
                        debug_output.classes(replace='text-xs text-red-400 mb-2')
                    display_jobs(jobs_data)

            # ===== TAB 7: PROGRESS TRACKER =====
            with ui.tab_panel(t_progress):
                ui.label('📊 Project Progress Tracker').classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Upload multiple files (images, PDFs, Excel) from different people to track project progress. The AI will extract data and create a unified summary with charts and an overview.').classes('markdown-body mb-2')

                uploaded_files = []
                upload_status = ui.label('No files uploaded yet.').classes('text-xs text-amber-400 mb-2')

                async def handle_progress_upload(e):
                    try:
                        data = await e.file.read()
                        fname = e.file.name
                        ftype = detect_mime_type(fname, data)
                        uploaded_files.append({'bytes': data, 'name': fname, 'type': ftype})
                        upload_status.set_text(f'{len(uploaded_files)} file(s) uploaded.')
                        upload_status.classes(replace='text-xs text-emerald-400 mb-2')
                        ui.notify(f'Uploaded: {fname}', type='positive')
                    except Exception as ex:
                        ui.notify(f'Upload error: {str(ex)}', type='negative')

                ui.label('Upload files (multiple allowed)').classes('text-white text-sm font-semibold mb-1')
                ui.upload(auto_upload=True, on_upload=handle_progress_upload, multiple=True).props('flat dark').classes('w-full mb-4')

                with ui.row().classes('w-full gap-4 mb-4'):
                    ui.label('Start Date').classes('text-white text-sm font-semibold')
                    start_date = ui.date(value=datetime.date.today() - datetime.timedelta(days=30)).classes('w-40')
                    ui.label('End Date').classes('text-white text-sm font-semibold')
                    end_date = ui.date(value=datetime.date.today()).classes('w-40')

                description_input = ui.input(label='Project Phase / Description', placeholder='e.g., Foundation Work', value='Foundation and Structure').classes('w-full mb-4')

                progress_output = ui.column().classes('w-full')
                progress_charts = ui.column().classes('w-full')
                progress_overview = ui.column().classes('w-full')
                progress_export = ui.row().classes('w-full gap-4 mt-4')

                async def run_progress_analysis():
                    if not client:
                        ui.notify('GEMINI_API_KEY missing!', type='negative')
                        return
                    if not uploaded_files:
                        ui.notify('Please upload at least one file.', type='warning')
                        return

                    progress_output.clear()
                    progress_charts.clear()
                    progress_overview.clear()
                    progress_export.clear()

                    with progress_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Processing files and extracting progress data...').classes('self-center text-sm')

                    try:
                        all_rows = []
                        for f in uploaded_files:
                            ext = os.path.splitext(f['name'])[1].lower()
                            if ext in ['.xlsx', '.xls']:
                                df = process_excel_file(f['bytes'], f['name'])
                                if not df.empty:
                                    df['source'] = f['name']
                                    all_rows.append(df)
                            elif f['type'] in ['image/png', 'image/jpeg', 'application/pdf']:
                                data = await extract_progress_from_image(f['bytes'], f['type'])
                                if data:
                                    row = {
                                        'date': data.get('date'),
                                        'description': data.get('description'),
                                        'progress_percent': data.get('progress_percent'),
                                        'category': data.get('category'),
                                        'location': data.get('location'),
                                        'source': f['name']
                                    }
                                    all_rows.append(pd.DataFrame([row]))
                            else:
                                ui.notify(f'Skipping unsupported file: {f["name"]}', type='warning')

                        if not all_rows:
                            ui.notify('No data could be extracted from the uploaded files.', type='warning')
                            progress_output.clear()
                            with progress_output:
                                ui.label('No data found in the uploaded files. Please check file contents.').classes('text-white')
                            return

                        combined_df = pd.concat(all_rows, ignore_index=True)
                        if 'date' in combined_df.columns:
                            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce')
                        if 'progress_percent' in combined_df.columns:
                            combined_df['progress_percent'] = pd.to_numeric(combined_df['progress_percent'], errors='coerce')

                        display_df = combined_df.copy()
                        if 'date' in display_df.columns:
                            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                        display_df = display_df.fillna('N/A')

                        progress_output.clear()
                        with progress_output:
                            ui.label('📋 Progress Summary Table').classes('text-xl font-bold text-white mb-2')
                            columns = [
                                {'name': col, 'label': col.replace('_', ' ').title(), 'field': col, 'sortable': True}
                                for col in display_df.columns if col != 'source'
                            ]
                            if 'source' in display_df.columns:
                                columns.append({'name': 'source', 'label': 'Source File', 'field': 'source', 'sortable': True})
                            ui.table(columns=columns, rows=display_df.to_dict('records'), row_key='index').classes('w-full text-white')

                        fig_bar, fig_scatter, fig_pie, fig_line = None, None, None, None
                        if not combined_df.empty:
                            if 'category' in combined_df.columns and 'progress_percent' in combined_df.columns:
                                avg_progress = combined_df.groupby('category')['progress_percent'].mean().reset_index()
                                if not avg_progress.empty:
                                    fig_bar = px.bar(avg_progress, x='category', y='progress_percent',
                                                     title='Average Progress by Category',
                                                     color='category', template='plotly_dark')
                                    fig_bar.update_layout(paper_bgcolor='#0d1a35', plot_bgcolor='#0d1a35', font_color='white')

                            if 'date' in combined_df.columns and 'progress_percent' in combined_df.columns:
                                df_time = combined_df.dropna(subset=['date', 'progress_percent'])
                                if not df_time.empty:
                                    fig_scatter = px.scatter(df_time, x='date', y='progress_percent',
                                                            color='category', title='Progress Over Time',
                                                            template='plotly_dark')
                                    fig_scatter.update_layout(paper_bgcolor='#0d1a35', plot_bgcolor='#0d1a35', font_color='white')

                            if 'category' in combined_df.columns:
                                cat_counts = combined_df['category'].value_counts().reset_index()
                                cat_counts.columns = ['category', 'count']
                                if not cat_counts.empty:
                                    fig_pie = px.pie(cat_counts, names='category', values='count',
                                                     title='Category Distribution', template='plotly_dark')
                                    fig_pie.update_layout(paper_bgcolor='#0d1a35', plot_bgcolor='#0d1a35', font_color='white')

                            if 'date' in combined_df.columns and 'progress_percent' in combined_df.columns:
                                df_time = combined_df.dropna(subset=['date', 'progress_percent']).sort_values('date')
                                if not df_time.empty:
                                    df_time['cumulative'] = df_time['progress_percent'].cumsum()
                                    fig_line = px.line(df_time, x='date', y='cumulative',
                                                       title='Cumulative Progress Over Time',
                                                       template='plotly_dark')
                                    fig_line.update_layout(paper_bgcolor='#0d1a35', plot_bgcolor='#0d1a35', font_color='white')

                        progress_charts.clear()
                        with progress_charts:
                            ui.label('📈 Charts').classes('text-xl font-bold text-white mb-2')
                            chart_grid = ui.row().classes('w-full gap-4')
                            with chart_grid:
                                if fig_bar:
                                    ui.plotly(fig_bar).classes('w-full md:w-1/2')
                                if fig_scatter:
                                    ui.plotly(fig_scatter).classes('w-full md:w-1/2')
                                if fig_pie:
                                    ui.plotly(fig_pie).classes('w-full md:w-1/2')
                                if fig_line:
                                    ui.plotly(fig_line).classes('w-full md:w-1/2')

                        overview_text = await generate_progress_overview(
                            combined_df,
                            start_date.value.strftime('%Y-%m-%d') if start_date.value else 'N/A',
                            end_date.value.strftime('%Y-%m-%d') if end_date.value else 'N/A',
                            description_input.value
                        )
                        progress_overview.clear()
                        with progress_overview:
                            ui.label('📝 AI Overview').classes('text-xl font-bold text-white mb-2')
                            ui.markdown(overview_text).classes('markdown-body')

                        pdf_data = {
                            'df': combined_df,
                            'fig_bar': fig_bar,
                            'fig_scatter': fig_scatter,
                            'fig_pie': fig_pie,
                            'fig_line': fig_line,
                            'overview': overview_text,
                            'start_date': start_date.value,
                            'end_date': end_date.value,
                            'description': description_input.value
                        }
                        progress_export.clear()
                        with progress_export:
                            def download_progress_pdf():
                                try:
                                    pdf_bytes = generate_progress_pdf(
                                        pdf_data,
                                        engineer_input.value,
                                        project_name_input.value,
                                        logo_bytes_holder['bytes'],
                                        ticket_input.value
                                    )
                                    ui.download(pdf_bytes, filename=f"Progress_Report_{ticket_input.value}.pdf")
                                    ui.notify('PDF downloaded!', type='positive')
                                except Exception as e:
                                    ui.notify(f'PDF generation error: {str(e)}', type='negative')

                            ui.button('Download Progress PDF', on_click=download_progress_pdf).classes('primary-btn')

                    except Exception as e:
                        progress_output.clear()
                        with progress_output:
                            ui.notify(f'Analysis failed: {str(e)}', type='negative')
                            ui.label(f'Error: {str(e)}').classes('text-red-400')

                ui.button('Run AI Analysis', on_click=run_progress_analysis).classes('primary-btn mt-4')

            # ===== TAB 8: DXF AREA EXTRACTOR =====
            with ui.tab_panel(t_dxf):
                ui.label('📐 DXF Area Extractor').classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Upload a DXF file (architectural or structural) to extract areas of closed polylines. The tool auto-detects layers and lets you choose workflow and units.').classes('markdown-body mb-2')

                dxf_file_data = {'bytes': None, 'name': None}
                dxf_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')

                async def handle_dxf_upload(e):
                    try:
                        data = await e.file.read()
                        if isinstance(data, str):
                            data = data.encode('utf-8')
                        dxf_file_data['bytes'] = data
                        dxf_file_data['name'] = e.file.name
                        dxf_status_label.set_text(f'File Ready: {e.file.name} ({(len(data)/1024):.1f} KB)')
                        dxf_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                        ui.notify(f'Uploaded: {e.file.name}', type='positive')
                        try:
                            doc = ezdxf.read(io.BytesIO(data))
                            layers = detect_dxf_layers(doc)
                            layer_info = "\n".join([f"{layer}: {info['count']} entities, keywords: {', '.join(info['keywords'])}" for layer, info in layers.items()])
                            detected_layers_label.set_text(f"Detected layers:\n{layer_info}")
                            detected_layers_label.classes(replace='text-xs text-white')
                        except Exception as ex:
                            detected_layers_label.set_text(f"Error reading DXF: {str(ex)}")
                            detected_layers_label.classes(replace='text-xs text-red-400')
                            traceback.print_exc()
                    except Exception as ex:
                        ui.notify(f'Upload error: {str(ex)}', type='negative')
                        traceback.print_exc()

                ui.label('Upload DXF File').classes('text-white text-sm font-semibold mb-1')
                ui.upload(auto_upload=True, on_upload=handle_dxf_upload, multiple=False).props('flat dark').classes('w-full mb-4')

                detected_layers_label = ui.label('Detected layers will appear here after upload.').classes('text-xs text-[#A9B6D0] mb-2')

                with ui.row().classes('w-full gap-4 mb-4'):
                    workflow_select = ui.select(
                        label='Workflow',
                        options=['Architectural BOQ', 'Structural Mass', 'Site Layout'],
                        value='Architectural BOQ'
                    ).classes('flex-1')
                    unit_select = ui.select(
                        label='Drawing Units',
                        options=['mm', 'cm', 'm'],
                        value='mm'
                    ).classes('flex-1')

                dxf_output = ui.column().classes('w-full')
                dxf_export = ui.row().classes('w-full gap-4 mt-4')
                dxf_data_holder = {'df': None, 'total_area': 0}

                async def process_dxf():
                    if dxf_file_data['bytes'] is None:
                        ui.notify('Please upload a DXF file first.', type='warning')
                        return
                    dxf_output.clear()
                    dxf_export.clear()
                    with dxf_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Processing DXF file...').classes('self-center text-sm')

                    try:
                        data = dxf_file_data['bytes']
                        if isinstance(data, str):
                            data = data.encode('utf-8')
                        doc = ezdxf.read(io.BytesIO(data))
                        workflow = workflow_select.value
                        unit = unit_select.value
                        areas = extract_areas_from_dxf(doc, unit=unit, workflow=workflow.lower().split()[0])

                        if not areas:
                            ui.notify('No closed polylines found in the DXF.', type='warning')
                            dxf_output.clear()
                            with dxf_output:
                                ui.label('No closed polylines found. Please check the DXF file.').classes('text-white')
                            return

                        df = pd.DataFrame(areas)
                        df = df.sort_values('area_m2', ascending=False)
                        dxf_data_holder['df'] = df
                        total = df['area_m2'].sum()
                        dxf_data_holder['total_area'] = total

                        dxf_output.clear()
                        with dxf_output:
                            ui.label('📋 Extracted Areas (m²)').classes('text-xl font-bold text-white mb-2')
                            columns = [
                                {'name': 'layer', 'label': 'Layer', 'field': 'layer', 'sortable': True},
                                {'name': 'label', 'label': 'Label', 'field': 'label', 'sortable': True},
                                {'name': 'area_m2', 'label': 'Area (m²)', 'field': 'area_m2', 'sortable': True},
                                {'name': 'vertices', 'label': 'Vertices', 'field': 'vertices', 'sortable': True},
                            ]
                            ui.table(columns=columns, rows=df.to_dict('records'), row_key='index').classes('w-full text-white')
                            ui.label(f'Total Net Area: {total:.4f} m²').classes('text-lg font-bold text-[#FF8C00] mt-2')

                        dxf_export.clear()
                        with dxf_export:
                            def download_dxf_pdf():
                                try:
                                    pdf_bytes = generate_dxf_pdf(
                                        df,
                                        total,
                                        dxf_file_data['name'],
                                        workflow_select.value,
                                        unit_select.value,
                                        engineer_input.value,
                                        project_name_input.value,
                                        logo_bytes_holder['bytes'],
                                        ticket_input.value
                                    )
                                    ui.download(pdf_bytes, filename=f"DXF_Area_Report_{ticket_input.value}.pdf")
                                    ui.notify('PDF downloaded!', type='positive')
                                except Exception as e:
                                    ui.notify(f'PDF generation error: {str(e)}', type='negative')
                                    traceback.print_exc()

                            ui.button('Download DXF Report PDF', on_click=download_dxf_pdf).classes('primary-btn')

                    except Exception as e:
                        dxf_output.clear()
                        with dxf_output:
                            ui.notify(f'Processing error: {str(e)}', type='negative')
                            ui.label(f'Error: {str(e)}').classes('text-red-400')
                            traceback.print_exc()

                ui.button('Process DXF', on_click=process_dxf).classes('primary-btn mt-4')
                with dxf_output:
                    ui.markdown('*Upload a DXF and click "Process DXF" to extract areas.*').classes('text-sm text-[#A9B6D0]')

            # ===== TAB 9: AUTOCAD LAYOUT GENERATOR (UPGRADED) =====
            with ui.tab_panel(t_autocad):
                ui.label('🏗️ AutoCAD Layout Generator – Full Architectural & Structural').classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Enter your plot details and generate a comprehensive DXF with architectural plan, structural grid, reinforcement, and BOQ with grand totals.').classes('markdown-body mb-2')

                with ui.row().classes('w-full gap-4 flex-wrap'):
                    # Plot & Building
                    with ui.column().classes('input-card flex-1'):
                        ui.label('Plot & Building').classes('font-bold text-white')
                        plot_area_input = ui.number(label='Plot Area (m²)', value=200, min=50, max=1000).classes('w-full')
                        street_width_input = ui.number(label='Street Width (m)', value=10, min=4, max=40).classes('w-full')
                        location_select = ui.select(
                            label='Location',
                            options=['Cairo', 'Giza', 'Alexandria', 'Delta', 'Other'],
                            value='Cairo'
                        ).classes('w-full')
                        num_floors_input = ui.number(label='Number of Floors', value=2, min=1, max=10).classes('w-full')
                        floor_height_input = ui.number(label='Floor Height (m)', value=3.0, step=0.1).classes('w-full')

                    # Architectural
                    with ui.column().classes('input-card flex-1'):
                        ui.label('Architectural Details').classes('font-bold text-white')
                        wall_thickness_mm = ui.number(label='Wall Thickness (mm)', value=200, step=10).classes('w-full')
                        door_width_mm = ui.number(label='Door Width (mm)', value=900, step=50).classes('w-full')
                        door_height_mm = ui.number(label='Door Height (mm)', value=2100, step=50).classes('w-full')
                        window_width_mm = ui.number(label='Window Width (mm)', value=1200, step=50).classes('w-full')
                        window_height_mm = ui.number(label='Window Height (mm)', value=1200, step=50).classes('w-full')

                    # Structural
                    with ui.column().classes('input-card flex-1'):
                        ui.label('Structural Grid').classes('font-bold text-white')
                        column_spacing_x = ui.number(label='Column Spacing X (m)', value=4.0, step=0.5).classes('w-full')
                        column_spacing_y = ui.number(label='Column Spacing Y (m)', value=4.0, step=0.5).classes('w-full')
                        beam_width_mm = ui.number(label='Beam Width (mm)', value=300, step=10).classes('w-full')
                        beam_depth_mm = ui.number(label='Beam Depth (mm)', value=500, step=10).classes('w-full')
                        slab_thickness_mm = ui.number(label='Slab Thickness (mm)', value=150, step=10).classes('w-full')

                    # Footings & Rebar
                    with ui.column().classes('input-card flex-1'):
                        ui.label('Footings & Reinforcement').classes('font-bold text-white')
                        footing_width_mm = ui.number(label='Footing Width (mm)', value=800, step=50).classes('w-full')
                        footing_depth_mm = ui.number(label='Footing Depth (mm)', value=400, step=50).classes('w-full')
                        footing_length_mm = ui.number(label='Footing Length (mm)', value=800, step=50).classes('w-full')
                        rebar_main_diam_mm = ui.number(label='Main Bar Diameter (mm)', value=16, step=2).classes('w-full')
                        rebar_stirrup_diam_mm = ui.number(label='Stirrup Diameter (mm)', value=10, step=2).classes('w-full')
                        rebar_spacing_mm = ui.number(label='Stirrup Spacing (mm)', value=200, step=10).classes('w-full')

                autocad_output = ui.column().classes('w-full')
                autocad_export = ui.row().classes('w-full gap-4 mt-4')
                autocad_data_holder = {'dxf': None, 'boq': None, 'info': None}

                async def generate_enhanced_autocad():
                    autocad_output.clear()
                    autocad_export.clear()
                    with autocad_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Generating full architectural & structural DXF with BOQ...').classes('self-center text-sm')

                    try:
                        params = {
                            'plot_polygon': None,
                            'plot_area_m2': plot_area_input.value or 200,
                            'street_width_m': street_width_input.value or 10,
                            'location': location_select.value or 'Cairo',
                            'num_floors': int(num_floors_input.value or 2),
                            'floor_height_m': floor_height_input.value or 3.0,
                            'wall_thickness_mm': int(wall_thickness_mm.value or 200),
                            'door_width_mm': int(door_width_mm.value or 900),
                            'door_height_mm': int(door_height_mm.value or 2100),
                            'window_width_mm': int(window_width_mm.value or 1200),
                            'window_height_mm': int(window_height_mm.value or 1200),
                            'column_spacing_x': column_spacing_x.value or 4.0,
                            'column_spacing_y': column_spacing_y.value or 4.0,
                            'beam_width_mm': int(beam_width_mm.value or 300),
                            'beam_depth_mm': int(beam_depth_mm.value or 500),
                            'slab_thickness_mm': int(slab_thickness_mm.value or 150),
                            'footing_width_mm': int(footing_width_mm.value or 800),
                            'footing_depth_mm': int(footing_depth_mm.value or 400),
                            'footing_length_mm': int(footing_length_mm.value or 800),
                            'rebar_main_diam_mm': int(rebar_main_diam_mm.value or 16),
                            'rebar_stirrup_diam_mm': int(rebar_stirrup_diam_mm.value or 10),
                            'rebar_spacing_mm': int(rebar_spacing_mm.value or 200),
                            'project_name': project_name_input.value,
                            'engineer': engineer_input.value,
                            'date': datetime.date.today().strftime('%Y-%m-%d'),
                            'num_units_per_floor': 2,   # you can later add a UI control
                        }

                        result = await asyncio.to_thread(build_complete_project, params)

                        autocad_data_holder['dxf'] = result['dxf']
                        autocad_data_holder['boq'] = result['boq']
                        autocad_data_holder['info'] = result['info']

                        autocad_output.clear()
                        with autocad_output:
                            ui.label('✅ Enhanced Layout Generated').classes('text-xl font-bold text-green-400 mb-2')
                            info = result['info']
                            ui.markdown(f"""
                            **Plot Area:** {info['plot_area']:.2f} m²  
                            **Street Width:** {info['street_width']} m  
                            **Location:** {info['location']}  
                            **Max Allowed Floors:** {info['max_floors']} (used {info['num_floors']})  
                            **Footprint Area:** {info['footprint_area']} m²  
                            **Building Dimensions:** {info['building_width']:.2f} m x {info['building_length']:.2f} m
                            """).classes('text-white')
                            ui.label('📋 Bill of Quantities (with Grand Total)').classes('text-xl font-bold text-white mt-4 mb-2')
                            boq_df = pd.DataFrame(result['boq'])
                            columns = [
                                {'name': 'Item', 'label': 'Item', 'field': 'Item', 'sortable': True},
                                {'name': 'Quantity', 'label': 'Quantity', 'field': 'Quantity', 'sortable': True},
                                {'name': 'Unit', 'label': 'Unit', 'field': 'Unit', 'sortable': True},
                                {'name': 'Unit Rate (EGP)', 'label': 'Unit Rate (EGP)', 'field': 'Unit Rate (EGP)', 'sortable': True},
                                {'name': 'Total Cost (EGP)', 'label': 'Total Cost (EGP)', 'field': 'Total Cost (EGP)', 'sortable': True},
                            ]
                            ui.table(columns=columns, rows=boq_df.to_dict('records'), row_key='index').classes('w-full text-white')
                            total_cost = boq_df['Total Cost (EGP)'].sum()
                            ui.label(f'🏷️ Grand Total Cost: {total_cost:,.2f} EGP').classes('text-2xl font-bold text-[#FF8C00] mt-2')

                        autocad_export.clear()
                        with autocad_export:
                            def download_dxf():
                                if autocad_data_holder['dxf']:
                                    ui.download(autocad_data_holder['dxf'], filename=f"Enhanced_Layout_{info['plot_area']:.0f}m2.dxf")
                                    ui.notify('DXF downloaded!', type='positive')
                                else:
                                    ui.notify('No DXF generated.', type='warning')
                            def download_pdf():
                                try:
                                    pdf_bytes = generate_autocad_pdf(
                                        info,
                                        boq_df,
                                        engineer_input.value,
                                        project_name_input.value,
                                        logo_bytes_holder['bytes'],
                                        ticket_input.value
                                    )
                                    ui.download(pdf_bytes, filename=f"Layout_Report_{info['plot_area']:.0f}m2.pdf")
                                    ui.notify('PDF downloaded!', type='positive')
                                except Exception as e:
                                    ui.notify(f'PDF error: {str(e)}', type='negative')
                                    traceback.print_exc()

                            ui.button('Download DXF', on_click=download_dxf).classes('primary-btn')
                            ui.button('Download PDF Report', on_click=download_pdf).classes('primary-btn')

                    except Exception as e:
                        autocad_output.clear()
                        with autocad_output:
                            ui.notify(f'Generation failed: {str(e)}', type='negative')
                            ui.label(f'Error: {str(e)}').classes('text-red-400')
                            traceback.print_exc()

                ui.button('Generate Enhanced Layout & BOQ', on_click=generate_enhanced_autocad).classes('primary-btn mt-4')
                with autocad_output:
                    ui.markdown('*Enter plot details and click to generate a high‑quality DXF with architectural, structural, and reinforcement layers.*').classes('text-sm text-[#A9B6D0]')

        # ---------------- FOOTER ----------------
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
