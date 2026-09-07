import io
import datetime
import os
import uuid
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import qrcode
import pypdf

# Dotenv & FastAPI / NiceGUI
from dotenv import load_dotenv
from fastapi import FastAPI
from nicegui import app, ui

# Google GenAI SDK
from google import genai
from google.genai import types

# ReportLab for Professional PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
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

# Load environment variables securely from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

# --- CUSTOM TAILWIND & DARK THEME STYLING ---
app.native.window_args = {"resizable": True}

ui.add_head_html('''
<style>
    ::-webkit-scrollbar {
        display: none !important;
        width: 0px !important;
        background: transparent !important;
    }
    html {
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    body {
        background-color: #031338 !important;
        color: #FFFFFF !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin: 0;
        padding: 0;
    }
    .custom-card {
        background-color: #1B2A4A;
        border: 1px solid #FF8C00;
        border-radius: 8px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 20px;
    }
    .primary-btn {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 2px solid #FF8C00 !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
        padding: 8px 16px;
    }
    .primary-btn:hover {
        background-color: #1E222D !important;
        color: #FF8C00 !important;
        border: 2px solid #00BFFF !important;
    }
    input, select, textarea, .q-field__control {
        background-color: #1E222D !important;
        color: #FFFFFF !important;
        border: 1px solid #FF8C00 !important;
        border-radius: 4px;
    }
    .q-field__native, .q-field__input, .q-field__label {
        color: #FFFFFF !important;
    }
    .q-menu, .q-popover, .q-virtual-scroll__content {
        background-color: #1B2A4A !important;
        color: #FFFFFF !important;
        border: 1px solid #FF8C00 !important;
    }
    .q-item {
        color: #FFFFFF !important;
        background-color: #1B2A4A !important;
    }
    .q-item:hover {
        background-color: #000000 !important;
        color: #FF8C00 !important;
    }
    .app-footer {
        position: relative;
        width: 100%;
        background-color: #1B2A4A;
        border-top: 2px solid #FF8C00;
        padding: 16px 20px;
        margin-top: 50px;
        text-align: center;
        color: #FFFFFF;
        font-size: 12px;
        box-sizing: border-box;
    }
</style>
''', shared=True)

# --- PDF & EXPORT GENERATION HELPERS ---
def generate_qr_code(data_str):
    qr = qrcode.QRCode(version=1, box_size=5, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def clean_for_reportlab(text):
    if not text:
        return ""
    text = str(text)
    # Strip LaTeX delimiters and clean formatting artifacts completely
    text = text.replace('$', '').replace('\\ge', '>=').replace('\\le', '<=')
    text = text.replace('\\frac', ' ').replace('\\sum', 'Sum').replace('\\sigma', 'sigma')
    text = text.replace('\\max', 'Max').replace('\\bar', ' ').replace('{', '').replace('}', '')
    text = re.sub(r'\|?\s*[:-]+[:?-]*\s*\|?', '', text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'#+\s*(.*)', r'<font color="#1B2A4A"><b>\1</b></font><br/>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = text.replace('\n', '<br/>')
    return text

def clean_ai_text(text):
    if not text:
        return ""
    # Clean LaTeX dollar signs, backslashes, and markdown artifacts for pristine UI display
    text = text.replace('$', '').replace('\\times', '*').replace('\\ge', '>=').replace('\\le', '<=')
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def build_pdf_header(story, doc_title, subtitle, logo_bytes, engineer, project, location, rep_date, ticket_id, unique_hash):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=13, textColor=colors.HexColor("#1B2A4A"), spaceAfter=4, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("DocSub", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#FF8C00"), spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#333333"), leading=9, fontName="Helvetica")

    meta_html = f"""
    <b>Project Name:</b> {project} &nbsp;|&nbsp; <b>Location:</b> {location}<br/>
    <b>Engineer in Charge:</b> {engineer} &nbsp;|&nbsp; <b>Audit Date:</b> {rep_date}<br/>
    <b>Batch Ticket ID:</b> {ticket_id} &nbsp;|&nbsp; <b>Verification UID:</b> <font color="#CC0000"><b>{unique_hash}</b></font><br/>
    <b>Governing Standards:</b> ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO
    """
    
    if logo_bytes:
        try:
            logo_img = ReportLabImage(io.BytesIO(logo_bytes), width=80, height=30)
            header_table_data = [[Paragraph(f"<b>{doc_title}</b>", title_style), logo_img],
                                 [Paragraph(subtitle, sub_style), ""],
                                 [Paragraph(meta_html, meta_style), ""]]
            t_head = Table(header_table_data, colWidths=[410, 130])
            t_head.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('ALIGN', (1,0), (1,-1), 'RIGHT'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ]))
            story.append(t_head)
        except Exception:
            story.append(Paragraph(doc_title, title_style))
            story.append(Paragraph(subtitle, sub_style))
            story.append(Paragraph(meta_html, meta_style))
    else:
        story.append(Paragraph(doc_title, title_style))
        story.append(Paragraph(subtitle, sub_style))
        story.append(Paragraph(meta_html, meta_style))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1B2A4A"), spaceAfter=8))

def build_pdf_footer_and_signatures(story, qr_img_buffer):
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#222222"), leading=8, fontName="Helvetica")
    sec_style = ParagraphStyle("SecTitle", parent=styles["Heading2"], fontSize=8, textColor=colors.HexColor("#1B2A4A"), spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold")

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Corporate Engineering Approvals & Compliance Sign-Off</b>", sec_style))
    
    qr_lab_img = ReportLabImage(qr_img_buffer, width=35, height=35)
    sign_cell_1 = Paragraph("<b>Prepared By:</b><br/>QA/QC Engineer:<br/><br/>_________________", body_style)
    sign_cell_2 = Paragraph("<b>Technical Director:</b><br/>Chief Engineer:<br/><br/>_________________", body_style)
    sign_cell_3 = Paragraph("<b>Client / Consultant:</b><br/>Official Stamp:<br/><br/>_________________", body_style)
    qr_cell = [Paragraph("<b>QR Verification:</b>", body_style), qr_lab_img]

    t_sign = Table([[sign_cell_1, sign_cell_2, sign_cell_3, qr_cell]], colWidths=[135, 135, 135, 95])
    t_sign.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ALIGN", (3, 0), (3, 0), "CENTER"),
    ]))
    story.append(t_sign)

# --- MAIN APP LAYOUT ---
@ui.page('/')
def main_page():
    with ui.row().classes('w-full items-center justify-between bg-[#1B2A4A] px-6 py-4 rounded-lg border border-[#FF8C00] mb-4 shadow-lg'):
        ui.label('Multi-Disciplinary Civil, Geotechnical & Pavement Engineering Auditor').classes('text-3xl font-bold text-white')
        ui.label('Eng. Mohamed Abd Al Aty').classes('text-base text-[#00BFFF] font-semibold')

    ticker_html = """
    <div style="overflow: hidden; white-space: nowrap; background-color: #1B2A4A; color: #FFFFFF; padding: 8px 0; font-weight: bold; font-size: 13px; margin-bottom: 15px; border-radius: 4px; border: 1px solid #FF8C00;">
      <div style="display: inline-block; padding-left: 100%; animation: marquee 28s linear infinite;">
        <span style="color: #FF8C00;">[CORE ACTIVE]</span> ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Advanced Geotechnical & Concrete Calculation Sheet &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Active Site Inspection Portal
      </div>
    </div>
    <style>
    @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
    """
    ui.add_head_html(ticker_html)

    # --- SIDEBAR CONFIGURATION WITH COLLAPSE ARROW BUTTON ---
    supp_code_select = None
    sidebar = ui.left_drawer().classes('bg-[#1B2A4A] text-white p-4').style('width: 340px;')
    with sidebar:
        with ui.row().classes('w-full items-center justify-between mb-2'):
            ui.label('PROJECT METADATA').classes('text-white font-bold text-base')
            ui.button(icon='menu', on_click=sidebar.toggle).classes('primary-btn p-1 text-xs')
            
        project_name_input = ui.input(label='Project Name', value='Highway Expansion Project').classes('w-full mb-2')
        pour_location_input = ui.input(label='Structural Element / Chainage', value='Highway Section Ch. 12+500').classes('w-full mb-4')

        ui.label('Governing Standards Core').classes('text-white font-bold text-sm mb-1')
        ui.markdown('*ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO*')
        
        supp_code_select = ui.select(
            label='Supplementary Standard',
            options=[
                "None (Strictly Core)",
                "ACI 318-25 — Structural Concrete",
                "IBC — International Building Code",
                "BS EN 1992 / Eurocode 2 + UK Annex",
                "AASHTO LRFD Bridge Design"
            ],
            value="None (Strictly Core)"
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

    # Sidebar toggle floating button if closed
    ui.button(icon='menu', on_click=sidebar.toggle).classes('fixed top-4 left-4 z-50 bg-[#1B2A4A] text-white border border-[#FF8C00] p-2 rounded shadow-lg')

    # --- TABS NAVIGATION ---
    with ui.tabs().classes('w-full text-white bg-[#1B2A4A] rounded-lg') as tabs:
        t_dash = ui.tab('Concrete Calculation Sheet & Verifier').classes('text-white font-bold')
        t_audit = ui.tab('AI Multi-Standard Auditor').classes('text-white font-bold')
        t_defect = ui.tab('Defect Diagnostic').classes('text-white font-bold')
        t_chat = ui.tab('AI Chatbot').classes('text-white font-bold')
        t_handbook = ui.tab('Technical Codes Handbook').classes('text-white font-bold')

    with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent'):
        
        # --- TAB 1: CONCRETE CALCULATION SHEET & VERIFIER ---
        with ui.tab_panel(t_dash):
            ui.label('Comprehensive Concrete Cube Calculation Sheet & Statistical Verifier (ECP 203)').classes('text-2xl font-bold text-white mb-4')
            
            with ui.row().classes('w-full gap-4 mb-4'):
                with ui.column().classes('custom-card flex-1'):
                    ui.label('7-Day Cubes (Comma Separated N/mm2)').classes('font-bold text-white text-sm')
                    c7_input = ui.input(value='21.0, 22.5, 20.5').classes('w-full')
                with ui.column().classes('custom-card flex-1'):
                    ui.label('14-Day Cubes (Comma Separated N/mm2)').classes('font-bold text-white text-sm')
                    c14_input = ui.input(value='26.0, 27.2, 25.8').classes('w-full')
                with ui.column().classes('custom-card flex-1'):
                    ui.label('28-Day Cubes (Comma Separated N/mm2)').classes('font-bold text-white text-sm')
                    c28_input = ui.input(value='32.5, 34.0, 31.0, 35.5, 29.0, 33.0').classes('w-full')

            ai_cube_result_holder = {'text': ''}

            def run_verification():
                result_output_area.clear()
                export_buttons_area.clear()
                chart_area.clear()
                
                if not client:
                    ui.notify('Gemini API key missing in .env!', type='negative')
                    return

                with result_output_area:
                    ui.spinner('ios', size='lg').classes('self-center text-[#00BFFF]')
                    ui.label('Running master AI statistical evaluation & code compliance verification...').classes('self-center text-sm')

                try:
                    supp_val = supp_code_select.value if supp_code_select else 'None (Strictly Core)'
                    
                    prompt = f"""
                    You are an elite Senior Concrete Quality Assurance and Structural Engineering Expert. 
                    Perform a complete, professional, exhaustive statistical evaluation and code verification for concrete cube test results.
                    
                    PROJECT PARAMETERS:
                    - Governing Core Standards: Egyptian Code ECP 203 (Primary), ECP 202, ASTM C39.
                    - Supplementary Standard: {supp_val}
                    - Specified 28-Day Characteristic Compressive Strength (f_cu): {fcu_input.value} N/mm²
                    - 7-Day Crushing Test Values: {c7_input.value} N/mm²
                    - 14-Day Crushing Test Values: {c14_input.value} N/mm²
                    - 28-Day Crushing Test Values: {c28_input.value} N/mm²
                    - Mix Details: Cement = {cement_input.value} kg/m³, Water = {water_input.value} kg/m²
                    - Truck No: {truck_input.value} | Ticket ID: {ticket_input.value}

                    REQUIREMENTS:
                    1. Use strictly METRIC (SI) plain text units (N/mm², MPa, kg/m³). DO NOT use complex LaTeX math strings or backslashes for formulas. Write them out in simple readable text (e.g., Mean, Standard Deviation S, CoV %, Characteristic Strength).
                    2. Provide clear Markdown Data Tables wrapped properly for each stage showing Specimen ID, Crushing Load, Deviation from Mean, and Individual Limit Check vs 0.85 * target limit.
                    3. Deliver a clear final compliance verdict (PASS / FAIL) based on ECP 203 criteria.
                    """

                    response = client.models.generate_content(
                        model='gemini-3.5-flash-lite',
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    
                    res_text = clean_ai_text(response.text)
                    ai_cube_result_holder['text'] = res_text

                    result_output_area.clear()
                    with result_output_area:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('AI-Powered Comprehensive Concrete Calculation Sheet & Statistical Proof').classes('text-xl font-bold text-white mb-2')
                            ui.markdown(res_text)

                    def parse_vals(txt):
                        try:
                            vals = [float(x.strip()) for x in txt.split(',') if x.strip()]
                            return sum(vals)/len(vals) if vals else 0
                        except:
                            return 0

                    m7 = parse_vals(c7_input.value)
                    m14 = parse_vals(c14_input.value)
                    m28 = parse_vals(c28_input.value)
                    target_fcu = float(fcu_input.value) if fcu_input.value else 30.0

                    with chart_area:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=['7-Day', '14-Day', '28-Day', 'Target Grade'], 
                            y=[m7, m14, m28, target_fcu], 
                            mode='lines+markers+text',
                            text=[f"{m7:.1f}", f"{m14:.1f}", f"{m28:.1f}", f"{target_fcu:.1f}"], 
                            textposition="top center",
                            line=dict(color='#00BFFF', width=3), 
                            marker=dict(size=10, color='#FF8C00')
                        ))
                        fig.add_hline(y=target_fcu, line_dash="dash", line_color="#22C55E", annotation_text=f"Target f_cu ({target_fcu} N/mm²)", annotation_position="bottom right")
                        fig.update_layout(
                            title='Compressive Strength Evolution & Target Threshold',
                            template='plotly_dark',
                            paper_bgcolor='#1B2A4A',
                            plot_bgcolor='#1B2A4A',
                            margin=dict(t=40, b=20, l=40, r=20),
                            height=300
                        )
                        ui.plotly(fig).classes('w-full mt-4')

                    with export_buttons_area:
                        unique_uid = f"ECP-AI-{uuid.uuid4().hex[:8].upper()}"

                        def download_pdf_report():
                            try:
                                buffer = io.BytesIO()
                                doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                                story = []
                                rep_date = datetime.date.today().strftime('%Y-%m-%d')
                                qr_buf = generate_qr_code(f"UID: {unique_uid} | ECP 203 AI Calculation Sheet - {project_name_input.value}")

                                build_pdf_header(
                                    story, 
                                    "AI CONCRETE CUBE CALCULATION & VERIFICATION REPORT", 
                                    f"Governing Standard: ECP 203 & {supp_val}", 
                                    logo_bytes_holder['bytes'], 
                                    engineer_input.value, 
                                    project_name_input.value, 
                                    pour_location_input.value, 
                                    rep_date,
                                    ticket_input.value,
                                    unique_uid
                                )
                                styles = getSampleStyleSheet()
                                body_style = ParagraphStyle("CubeBody", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=12)
                                
                                sanitized_text = clean_for_reportlab(ai_cube_result_holder['text'])
                                story.append(Paragraph(sanitized_text, body_style))
                                story.append(Spacer(1, 10))
                                build_pdf_footer_and_signatures(story, qr_buf)

                                doc.build(story)
                                buffer.seek(0)
                                ui.download(buffer.getvalue(), filename=f"AI_Concrete_Calculation_Sheet_{ticket_input.value}.pdf")
                                ui.notify('Official Calculation Sheet PDF downloaded successfully!', type='positive')
                            except Exception as ex:
                                ui.notify(f'PDF Generation Error: {str(ex)}', type='negative')

                        def download_csv_export():
                            df = pd.DataFrame({
                                "Field": ["Project Name", "Location", "Specified f_cu", "Selected Standard", "Truck No", "Batch Ticket", "Verification UID", "AI Findings Summary"],
                                "Value": [
                                    project_name_input.value, 
                                    pour_location_input.value, 
                                    str(fcu_input.value), 
                                    supp_val, 
                                    truck_input.value, 
                                    ticket_input.value, 
                                    unique_uid,
                                    ai_cube_result_holder['text'][:300].replace('\n', ' ')
                                ]
                            })
                            ui.download(df.to_csv(index=False).encode('utf-8'), filename=f"AI_Concrete_Calculation_{ticket_input.value}.csv")
                            ui.notify('CSV Calculation Sheet downloaded successfully!', type='positive')

                        ui.button('Download Official Calculation PDF', on_click=download_pdf_report).classes('primary-btn flex-1')
                        ui.button('Export Calculation CSV', on_click=download_csv_export).classes('primary-btn flex-1')

                except Exception as ex:
                    result_output_area.clear()
                    with result_output_area:
                        ui.notify(f'Calculation Error: {str(ex)}', type='negative')

            stage_selector = ui.select(
                label='Select Stage Display Filter',
                options=['All Stages', '7-Day Stage', '14-Day Stage', '28-Day Stage'],
                value='All Stages',
                on_change=run_verification
            ).classes('w-full md:w-1/3 mb-4')

            result_output_area = ui.column().classes('w-full')
            chart_area = ui.column().classes('w-full')
            export_buttons_area = ui.row().classes('w-full gap-4 mt-4')

            ui.button('Run AI Statistical Calculation & Verification', on_click=run_verification).classes('primary-btn q-my-md')
            run_verification() # Initial render

        # --- TAB 2: AI MULTI-STANDARD AUDITOR ---
        with ui.tab_panel(t_audit):
            ui.label('AI Multi-Standard Engineering Auditor (Master Suite)').classes('text-2xl font-bold text-white mb-2')
            ui.markdown('Upload any PDF specification, mix design, or site report to audit against **ECP 203, 202, 104, ASTM, AASHTO, BS, EN, and ISO** with rigorous formulas and comparative tables.')
            
            audit_focus = ui.select(
                label='Audit Focus',
                options=[
                    "Multi-Standard Structural & Geotechnical Compliance",
                    "Roads, Pavements & Subgrade Materials (ECP 104 & AASHTO)",
                    "Soil Mechanics & Foundations (ECP 202 & ASTM / ISO)",
                    "Reinforced Concrete Structures (ECP 203 & ACI / BS EN)"
                ],
                value="Multi-Standard Structural & Geotechnical Compliance"
            ).classes('w-full mb-4')

            audit_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
            uploaded_file_data = {'bytes': None, 'name': None, 'type': None}

            async def handle_audit_upload(e):
                try:
                    uploaded_file_data['bytes'] = await e.file.read()
                    uploaded_file_data['name'] = e.file.name
                    uploaded_file_data['type'] = 'application/pdf' if e.file.name.lower().endswith('.pdf') else 'image/jpeg'
                    
                    audit_status_label.set_text(f'File Ready: {e.file.name}')
                    audit_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    ui.notify(f'Successfully loaded: {e.file.name}', type='positive')
                except Exception as ex:
                    ui.notify(f'Error reading file: {str(ex)}', type='negative')

            ui.upload(label='Select PDF or Image File', auto_upload=True, on_upload=handle_audit_upload).props('flat dark').classes('w-full mb-4 bg-[#1E293B] rounded-lg')

            audit_output_container = ui.column().classes('w-full')
            audit_export_container = ui.row().classes('w-full gap-4 mt-4')
            audit_result_text_holder = {'text': ''}

            def run_ai_audit():
                if not client:
                    ui.notify('Gemini API key missing in .env!', type='negative')
                    return
                if not uploaded_file_data['bytes']:
                    ui.notify('Please upload a file first!', type='warning')
                    return

                audit_output_container.clear()
                audit_export_container.clear()
                
                with audit_output_container:
                    ui.spinner('ios', size='lg').classes('self-center text-[#00BFFF]')
                    ui.label('Executing master-level multi-standard engineering audit & formula verification...').classes('self-center text-sm')

                try:
                    supp_val = supp_code_select.value if supp_code_select else 'None'
                    prompt = f"""
                    You are a Principal Civil, Geotechnical and Highway Engineering Consultant and Lead Auditor specializing in core Egyptian Codes (ECP 203, ECP 202, ECP 104) alongside international frameworks (ASTM, AASHTO, BS EN, ISO).
                    Audit Focus: {audit_focus}
                    Active Supplementary Standard: {supp_val}
                    Perform a comprehensive, rigorous technical audit of the provided document or image. Structure your report using clear markdown sections and clean data tables in proper plain text without raw LaTeX or formatting errors.
                    """
                    
                    contents = [prompt]
                    if uploaded_file_data['type'] == 'application/pdf':
                        reader = pypdf.PdfReader(io.BytesIO(uploaded_file_data['bytes']))
                        text = "".join([p.extract_text() or "" for p in reader.pages])
                        contents.append(f"Extracted PDF Text:\n{text}")
                    else:
                        img_part = types.Part.from_bytes(data=uploaded_file_data['bytes'], mime_type=uploaded_file_data['type'])
                        contents.append(img_part)

                    config = types.GenerateContentConfig(temperature=0.1)
                    response = client.models.generate_content(
                        model='gemini-3.5-flash-lite', 
                        contents=contents,
                        config=config
                    )
                    audit_result_text = clean_ai_text(response.text)
                    audit_result_text_holder['text'] = audit_result_text
                    
                    audit_output_container.clear()
                    with audit_output_container:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Master Engineering Audit Findings & Code Compliance Report').classes('text-xl font-bold text-white mb-2')
                            ui.markdown(audit_result_text)

                    with audit_export_container:
                        unique_uid = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"

                        def download_audit_pdf():
                            try:
                                buffer = io.BytesIO()
                                doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                                story = []
                                rep_date = datetime.date.today().strftime('%Y-%m-%d')
                                qr_buf = generate_qr_code(f"UID: {unique_uid} | AI Audit - {project_name_input.value}")

                                build_pdf_header(
                                    story, 
                                    "AI MULTI-STANDARD ENGINEERING AUDIT REPORT", 
                                    f"Focus: {audit_focus}", 
                                    logo_bytes_holder['bytes'], 
                                    engineer_input.value, 
                                    project_name_input.value, 
                                    pour_location_input.value, 
                                    rep_date,
                                    ticket_input.value,
                                    unique_uid
                                )
                                styles = getSampleStyleSheet()
                                body_style = ParagraphStyle("AuditBody", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=12)
                                
                                sanitized_text = clean_for_reportlab(audit_result_text_holder['text'])
                                story.append(Paragraph(sanitized_text, body_style))
                                story.append(Spacer(1, 10))
                                build_pdf_footer_and_signatures(story, qr_buf)

                                doc.build(story)
                                buffer.seek(0)
                                ui.download(buffer.getvalue(), filename=f"AI_Audit_Report_{ticket_input.value}.pdf")
                                ui.notify('AI Audit PDF Report downloaded successfully!', type='positive')
                            except Exception as ex:
                                ui.notify(f'PDF Export Error: {str(ex)}', type='negative')

                        def download_audit_csv():
                            df = pd.DataFrame({
                                "Audit Field": ["Project Name", "Focus", "Source File", "Engineer", "Verification UID", "Summary Findings"],
                                "Value": [project_name_input.value, audit_focus, uploaded_file_data['name'], engineer_input.value, unique_uid, audit_result_text_holder['text'][:300].replace('\n', ' ')]
                            })
                            ui.download(df.to_csv(index=False).encode('utf-8'), filename=f"AI_Audit_{ticket_input.value}.csv")
                            ui.notify('AI Audit CSV Summary downloaded!', type='positive')

                        ui.button('Download AI Audit PDF Report', on_click=download_audit_pdf).classes('primary-btn flex-1')
                        ui.button('Export Audit CSV', on_click=download_audit_csv).classes('primary-btn flex-1')

                except Exception as ex:
                    audit_output_container.clear()
                    with audit_output_container:
                        ui.notify(f'Error: {str(ex)}', type='negative')

            ui.button('Execute Master AI Audit & Compliance Check', on_click=run_ai_audit).classes('primary-btn')

        # --- TAB 3: DEFECT DIAGNOSTIC ---
        with ui.tab_panel(t_defect):
            ui.label('AI Crack, Pavement & Geotechnical Defect Diagnostic').classes('text-2xl font-bold text-white mb-2')
            ui.markdown('Upload site defect photos for automated classification and repair protocols conforming to ECP 203, ECP 104, Sika, and Fosroc standards.')
            
            defect_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
            defect_file_data = {'bytes': None, 'type': None}
            defect_result_holder = {'text': ''}

            async def handle_defect_upload(e):
                try:
                    defect_file_data['bytes'] = await e.file.read()
                    defect_file_data['type'] = 'image/jpeg'
                    defect_status_label.set_text(f'File Ready: {e.file.name}')
                    defect_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    ui.notify(f'Successfully loaded defect image: {e.file.name}', type='positive')
                except Exception as ex:
                    ui.notify(f'Error reading file: {str(ex)}', type='negative')

            ui.upload(label='Select Site Defect Photo (JPG/PNG)', auto_upload=True, on_upload=handle_defect_upload).props('flat dark').classes('w-full mb-4 bg-[#1E293B] rounded-lg')
            defect_output = ui.column().classes('w-full')
            defect_export_area = ui.row().classes('w-full gap-4 mt-4')

            def run_defect_diagnosis():
                if not client or not defect_file_data['bytes']:
                    ui.notify('API key missing or image not uploaded!', type='negative')
                    return
                defect_output.clear()
                defect_export_area.clear()
                with defect_output:
                    ui.spinner('ios', size='lg').classes('self-center text-[#00BFFF]')
                    ui.label('Analyzing defect and matching local repair products (Sika/Fosroc)...').classes('self-center text-sm')

                try:
                    img = types.Part.from_bytes(data=defect_file_data['bytes'], mime_type=defect_file_data['type'])
                    prompt = "Perform forensic structural evaluation and list repair products (Sika/Fosroc) complying with ECP 203 and ASTM in clean formatted tables."
                    response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=[prompt, img])
                    res_text = clean_ai_text(response.text)
                    defect_result_holder['text'] = res_text

                    defect_output.clear()
                    with defect_output:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Forensic Diagnosis & Repair Protocol').classes('text-xl font-bold text-white mb-2')
                            ui.markdown(res_text)

                    with defect_export_area:
                        unique_uid = f"DEFECT-{uuid.uuid4().hex[:8].upper()}"

                        def download_defect_pdf():
                            try:
                                buffer = io.BytesIO()
                                doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                                story = []
                                rep_date = datetime.date.today().strftime('%Y-%m-%d')
                                qr_buf = generate_qr_code(f"UID: {unique_uid} | Defect Diagnostic - {project_name_input.value}")

                                build_pdf_header(
                                    story, 
                                    "AI DEFECT DIAGNOSTIC & REPAIR REPORT", 
                                    "Forensic Structural Evaluation & Sika/Fosroc Repair Protocols", 
                                    logo_bytes_holder['bytes'], 
                                    engineer_input.value, 
                                    project_name_input.value, 
                                    pour_location_input.value, 
                                    rep_date,
                                    ticket_input.value,
                                    unique_uid
                                )
                                styles = getSampleStyleSheet()
                                body_style = ParagraphStyle("DefectBody", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=12)
                                
                                sanitized_text = clean_for_reportlab(defect_result_holder['text'])
                                story.append(Paragraph(sanitized_text, body_style))
                                story.append(Spacer(1, 10))
                                build_pdf_footer_and_signatures(story, qr_buf)

                                doc.build(story)
                                buffer.seek(0)
                                ui.download(buffer.getvalue(), filename=f"Defect_Diagnostic_Report_{ticket_input.value}.pdf")
                                ui.notify('Defect Diagnostic PDF downloaded successfully!', type='positive')
                            except Exception as ex:
                                ui.notify(f'PDF Export Error: {str(ex)}', type='negative')

                        ui.button('Download Defect PDF Report', on_click=download_defect_pdf).classes('primary-btn flex-1')

                except Exception as ex:
                    defect_output.clear()
                    with defect_output:
                        ui.notify(f'Diagnosis failed: {ex}', type='negative')

            ui.button('Diagnose Defect & Get Repair Protocol', on_click=run_defect_diagnosis).classes('primary-btn')

        # --- TAB 4: AI CHATBOT ---
        with ui.tab_panel(t_chat):
            ui.label('Core-Code Intelligent Assistant Chatbot (Master Engine)').classes('text-2xl font-bold text-white mb-2')
            ui.markdown('Ask any engineering, mix design, geotechnical, or pavement question based strictly on core codes (**ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO**).')

            chat_container = ui.column().classes('custom-card w-full h-96 overflow-y-auto mb-4')
            chat_messages = [{"role": "assistant", "content": "Hello! I am your Multi-Standard Engineering Assistant. How can I assist you with your civil, geotechnical, or concrete queries today?"}]

            def render_chat():
                chat_container.clear()
                with chat_container:
                    for msg in chat_messages:
                        bg = 'bg-[#121E36]' if msg['role'] == 'assistant' else 'bg-[#2A3F6A]'
                        with ui.row().classes(f'w-full p-3 rounded-lg mb-2 {bg}'):
                            ui.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

            render_chat()
            user_msg = ui.input(placeholder='Type your engineering question here...').classes('w-full mb-2')

            def send_chat():
                q = user_msg.value
                if not q or not q.strip(): return
                chat_messages.append({"role": "user", "content": q})
                user_msg.value = ''
                render_chat()

                if not client:
                    chat_messages.append({"role": "assistant", "content": "GEMINI_API_KEY is not configured."})
                    render_chat()
                    return

                try:
                    system_prompt = (
                        "You are an elite Senior Civil, Geotechnical, and Structural Quality Engineering Expert "
                        "acting as a master multi-standard technical assistant. "
                        "\n\nSTRICT FORMATTING & COMPLIANCE RULES:"
                        "\n1. STANDARD COMPLIANCE: Ground all technical answers, design formulas, specifications, and "
                        "recommendations strictly in the requested codes: ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO. Cite exact clauses."
                        "\n2. UNIT SYSTEM: Use strictly METRIC (SI) units (mm, cm, m, MPa, kN, kg/m³, °C). DO NOT use imperial units."
                        "\n3. NO LATEX / NO RAW MATH BLOCKS: DO NOT use LaTeX double-dollar signs ($$), backslashes for math symbols, or broken markdown formatting like four asterisks (****). Write all formulas clearly in clean readable plain text (e.g., Mean = sum(x)/n, Standard Deviation S, CoV %). Use clean standard markdown tables and headings."
                    )

                    res = client.models.generate_content(
                        model='gemini-3.5-flash-lite',
                        contents=q,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            system_instruction=system_prompt
                        )
                    )
                    
                    cleaned_response = clean_ai_text(res.text).replace('$$', '').replace('****', '**')
                    chat_messages.append({"role": "assistant", "content": cleaned_response})
                except Exception as e:
                    chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                render_chat()

            with ui.row().classes('w-full gap-4 mt-2'):
                ui.button('Send Query', on_click=send_chat).classes('primary-btn flex-1')

                def download_chat_pdf():
                    try:
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                        story = []
                        rep_date = datetime.date.today().strftime('%Y-%m-%d')
                        unique_uid = f"CHAT-{uuid.uuid4().hex[:8].upper()}"
                        qr_buf = generate_qr_code(f"UID: {unique_uid} | Chat Transcript - {project_name_input.value}")

                        build_pdf_header(
                            story, 
                            "AI ENGINEERING ASSISTANT TRANSCRIPT", 
                            "Official Q&A Consultation Record", 
                            logo_bytes_holder['bytes'], 
                            engineer_input.value, 
                            project_name_input.value, 
                            pour_location_input.value, 
                            rep_date,
                            ticket_input.value,
                            unique_uid
                        )
                        styles = getSampleStyleSheet()
                        body_style = ParagraphStyle("ChatBody", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=12)
                        
                        full_chat_text = "<br/><br/>".join([f"<b>{m['role'].upper()}:</b> {clean_for_reportlab(m['content'])}" for m in chat_messages])
                        story.append(Paragraph(full_chat_text, body_style))
                        story.append(Spacer(1, 10))
                        build_pdf_footer_and_signatures(story, qr_buf)

                        doc.build(story)
                        buffer.seek(0)
                        ui.download(buffer.getvalue(), filename=f"AI_Chat_Transcript_{ticket_input.value}.pdf")
                        ui.notify('Chat Transcript PDF downloaded successfully!', type='positive')
                    except Exception as ex:
                        ui.notify(f'PDF Export Error: {str(ex)}', type='negative')

                ui.button('Download Chat PDF Transcript', on_click=download_chat_pdf).classes('primary-btn flex-1')

        # --- TAB 5: TECHNICAL HANDBOOK ---
        with ui.tab_panel(t_handbook):
            ui.label('Multi-Standard Civil Engineering Technical Handbook').classes('text-2xl font-bold text-white mb-4')
            with ui.tabs().classes('w-full text-white bg-[#1B2A4A] rounded-lg') as hb_tabs:
                h1 = ui.tab('ECP 203 & Concrete').classes('text-white font-bold')
                h2 = ui.tab('ECP 202 & Soils').classes('text-white font-bold')
                h3 = ui.tab('ECP 104 & Roads').classes('text-white font-bold')
                h4 = ui.tab('International Standards').classes('text-white font-bold')

            with ui.tab_panels(hb_tabs, value=h1).classes('w-full bg-transparent'):
                with ui.tab_panel(h1):
                    ui.markdown('### Egyptian Code for Reinforced Concrete Structures (ECP 203) - Comprehensive Reference')
                with ui.tab_panel(h2):
                    ui.markdown('### Egyptian Code for Soil Mechanics & Foundations (ECP 202)')
                with ui.tab_panel(h3):
                    ui.markdown('### Egyptian Code for Roads, Highways and Airfields (ECP 104)')
                with ui.tab_panel(h4):
                    ui.markdown('### International Standards (ASTM, AASHTO, BS EN, ISO)')

    # --- PROFESSIONAL FOOTER BAR ---
    ui.markdown('''
    <div class="app-footer">
        <b>Multi-Standard Engineering Quality Assurance Portal</b> &nbsp;|&nbsp; Automated compliance verification across ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO standards.<br>
        <b>Official Direct Contacts & Professional Network:</b> 
        LinkedIn: <a href="https://www.linkedin.com/in/mohamed-abdalaty" target="_blank" style="color: #00BFFF; text-decoration: underline;">Mohamed Abd Al Aty</a> &nbsp;|&nbsp; 
        Email: <a href="mailto:mohamedabdalaty63@gmail.com" style="color: #00BFFF; text-decoration: underline;">mohamedabdalaty63@gmail.com</a><br>
        <i>Specialized in Geotechnical QA/QC, Civil Engineering Standards & Automated Compliance.</i> &copy; 2026 Eng. Mohamed Abd Al Aty. All rights reserved.
    </div>
    ''')

ui.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), title='Multi-Standard Engineering Auditor', favicon='🏗️', reload=False)
