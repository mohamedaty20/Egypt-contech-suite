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

# ----- Helper functions for PDF reports (they were originally at the end) -----
def generate_progress_pdf(pdf_data, engineer_name, project_name, logo_bytes, ticket_id):
    from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph, HRFlowable, Image as ReportLabImage
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from config import USABLE_WIDTH, MARGIN, PAGE_WIDTH, PAGE_HEIGHT
    from services.pdf_service import build_pdf_styles, generate_qr_code, build_pdf_footer_signature_and_qr, markdown_to_pdf_flowables
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
# Main NiceGUI Page
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

            # ===== TAB 1: CONCRETE CUBE VERIFIER (full) =====
            with ui.tab_panel(t_dash):
                # ... (the entire code for tab 1 is unchanged – keep your existing code here)
                # For brevity I'll not repeat the full code; you already have it in your original file.
                # But the imports are now correct, so this tab will work.
                # (Paste your existing tab 1 code here)
                pass

            # ===== TAB 2: AI AUDITOR =====
            with ui.tab_panel(t_audit):
                # ... (your existing code)
                pass

            # ===== TAB 3: DEFECT DIAGNOSTIC =====
            with ui.tab_panel(t_defect):
                # ... (your existing code)
                pass

            # ===== TAB 4: AI CHATBOT =====
            with ui.tab_panel(t_chat):
                # ... (your existing code)
                pass

            # ===== TAB 5: HANDWRITING OCR =====
            with ui.tab_panel(t_handwriting):
                # ... (your existing code)
                pass

            # ===== TAB 6: JOB BOARD =====
            with ui.tab_panel(t_jobs):
                # ... (your existing code)
                pass

            # ===== TAB 7: PROGRESS TRACKER =====
            with ui.tab_panel(t_progress):
                # ... (your existing code)
                pass

            # ===== TAB 8: DXF AREA EXTRACTOR =====
            with ui.tab_panel(t_dxf):
                # ... (your existing code)
                pass

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

                # Output and Export containers
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
                            'num_units_per_floor': 2,   # you can add a UI control later
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
