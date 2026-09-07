import io
import datetime
import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import qrcode
import pypdf
from PIL import Image

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
    body {
        background-color: #031338 !important;
        color: #FFFFFF !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .custom-card {
        background-color: #1B2A4A;
        border: 1px solid #FF8C00;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 15px;
    }
    .primary-btn {
        background-color: #000000 !important;
        color: #FFFFFF !important;
        border: 2px solid #FF8C00 !important;
        font-weight: 700 !important;
        border-radius: 6px !important;
    }
    .primary-btn:hover {
        background-color: #1E222D !important;
        color: #FF8C00 !important;
        border: 2px solid #00BFFF !important;
    }
    input, select, textarea {
        background-color: #1E222D !important;
        color: #FFFFFF !important;
        border: 1px solid #FF8C00 !important;
        border-radius: 4px;
        padding: 6px;
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

def format_markdown_for_reportlab(text):
    if not text:
        return ""
    cleaned = re.sub(r'\$(.*?)\$', r'\1', text)
    cleaned = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cleaned = re.sub(r'[\-\|\:]+', ' ', cleaned)
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cleaned)
    cleaned = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cleaned)
    cleaned = re.sub(r'^\s*[\*\-]\s+', '&bull; ', cleaned, flags=re.MULTILINE)
    return cleaned

def build_pdf_header(story, doc_title, subtitle, logo_bytes, engineer, project, location, rep_date):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1B2A4A"), spaceAfter=4, alignment=1, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("DocSub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#444444"), spaceAfter=8, alignment=1, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=11, fontName="Helvetica")

    if logo_bytes:
        try:
            story.append(ReportLabImage(io.BytesIO(logo_bytes), width=80, height=30))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    story.append(Paragraph(format_markdown_for_reportlab(doc_title), title_style))
    story.append(Paragraph(format_markdown_for_reportlab(subtitle), sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1B2A4A"), spaceAfter=6))

    meta_html = f"""
    <b>Project:</b> {project} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Location:</b> {location}<br/>
    <b>Engineer:</b> {engineer} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> {rep_date} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Governing Codes:</b> ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO
    """
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 8))

def build_pdf_footer_and_signatures(story, qr_img_buffer):
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=10, fontName="Helvetica")
    sec_style = ParagraphStyle("SecTitle", parent=styles["Heading2"], fontSize=10, textColor=colors.HexColor("#1B2A4A"), spaceBefore=8, spaceAfter=4, fontName="Helvetica-Bold")

    story.append(Spacer(1, 8))
    story.append(Paragraph("<b>Engineering Approvals & Multi-Standard Compliance Sign-Off</b>", sec_style))
    
    qr_lab_img = ReportLabImage(qr_img_buffer, width=45, height=45)
    sign_cell_1 = Paragraph("<b>Prepared By:</b><br/>Engineer Sign:<br/>___________________", body_style)
    sign_cell_2 = Paragraph("<b>QA/QC Checked:</b><br/>Inspector Sign:<br/>___________________", body_style)
    sign_cell_3 = Paragraph("<b>Consultant Approved:</b><br/>Stamp & Sign:<br/>___________________", body_style)
    qr_cell = [Paragraph("<b>QR Verify:</b>", body_style), qr_lab_img]

    t_sign = Table([[sign_cell_1, sign_cell_2, sign_cell_3, qr_cell]], colWidths=[130, 130, 130, 100])
    t_sign.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F5F7FA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (3, 0), (3, 0), "CENTER"),
    ]))
    story.append(t_sign)

# --- MAIN APP LAYOUT ---
@ui.page('/')
def main_page():
    # Top Header & Ticker
    with ui.row().classes('w-full items-center justify-between bg-[#1B2A4A] px-6 py-3 rounded-lg border border-[#FF8C00] mb-4'):
        ui.label('🏗️ Multi-Disciplinary Civil, Geotechnical & Pavement Engineering Auditor').classes('text-xl font-bold text-white')
        ui.label('Made by Eng. Mohamed Abd Al Aty').classes('text-sm text-[#00BFFF] font-semibold')

    ticker_html = """
    <div style="overflow: hidden; white-space: nowrap; background-color: #FF8C00; color: #031338; padding: 6px 0; font-weight: bold; font-size: 13px; margin-bottom: 15px; border-radius: 4px;">
      <div style="display: inline-block; padding-left: 100%; animation: marquee 25s linear infinite;">
        🚀 Core Compliance Active: ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ Multi-Disciplinary Engineering & Geotechnical QA/QC Verifier &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; 🏗️ Active Site Inspection Portal
      </div>
    </div>
    <style>
    @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
    """
    ui.add_head_html(ticker_html)

    # --- SIDEBAR CONFIGURATION ---
    with ui.left_drawer().classes('bg-[#1B2A4A] text-white p-4').style('width: 340px;'):
        ui.label('PROJECT METADATA').classes('text-white font-bold text-base mb-2')
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

        fcu_input = ui.number(label='Specified 28-Day Grade fcu (N/mm²)', value=30.0, step=5.0).classes('w-full mb-4')
        
        ui.label('Batch Plant & Site Logs').classes('text-white font-bold text-sm mb-2')
        truck_input = ui.input(label='Mixer Truck No.', value='TRK-104').classes('w-full mb-2')
        ticket_input = ui.input(label='Batch Ticket ID', value='BT-99482').classes('w-full mb-4')

        ui.label('Mix Design Parameters').classes('text-white font-bold text-sm mb-2')
        cement_input = ui.input(label='Cement Content (kg/m³)', value='350.0').classes('w-full mb-2')
        water_input = ui.input(label='Free Water Content (kg/m³)', value='150.0').classes('w-full mb-4')
        
        engineer_input = ui.input(label='Engineer Name', value='Eng. Mohamed Abd Al Aty').classes('w-full mb-2')
        
        logo_status = ui.label('Logo: Not uploaded').classes('text-xs text-amber-400 mb-1')
        logo_upload = ui.upload(label='Upload Company Logo', auto_upload=False).props('flat dark').classes('w-full mb-2')
        logo_bytes_holder = {'bytes': None}
        def process_logo():
            if logo_upload.files:
                logo_bytes_holder['bytes'] = logo_upload.files[0].read()
                logo_status.set_text(f'✅ Logo Loaded: {logo_upload.files[0].name}')
                logo_status.classes(replace='text-xs text-emerald-400 mb-1')
                ui.notify('Company logo loaded successfully!', type='positive')
        ui.button('Confirm Logo', on_click=process_logo).classes('bg-slate-700 text-white text-xs w-full mb-2')

    # --- TABS / SCREENS NAVIGATION ---
    with ui.tabs().classes('w-full text-[#00BFFF]') as tabs:
        t_dash = ui.tab('📊 Concrete Verifier Dashboard', icon='dashboard')
        t_audit = ui.tab('🤖 AI Multi-Standard Auditor', icon='psychology')
        t_defect = ui.tab('🔍 Defect Diagnostic', icon='search')
        t_chat = ui.tab('💬 AI Chatbot', icon='chat')
        t_handbook = ui.tab('📖 Technical Codes Handbook', icon='book')

    with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent'):
        
        # --- TAB 1: CONCRETE VERIFIER DASHBOARD ---
        with ui.tab_panel(t_dash):
            ui.label('1. Input Cube Crushing Results (N/mm²)').classes('text-xl font-bold text-white mb-3')
            
            with ui.row().classes('w-full gap-4 mb-4'):
                with ui.column().classes('custom-card flex-1'):
                    ui.label('7-Day Cubes').classes('font-bold text-[#00BFFF]')
                    c7_input = ui.textarea(value='21.0, 22.5, 20.5').classes('w-full')
                with ui.column().classes('custom-card flex-1'):
                    ui.label('14-Day Cubes').classes('font-bold text-[#00BFFF]')
                    c14_input = ui.textarea(value='26.0, 27.2, 25.8').classes('w-full')
                with ui.column().classes('custom-card flex-1'):
                    ui.label('28-Day Cubes').classes('font-bold text-[#00BFFF]')
                    c28_input = ui.textarea(value='32.5, 34.0, 31.0, 35.5, 29.0, 33.0').classes('w-full')

            result_output_area = ui.column().classes('w-full')
            export_buttons_area = ui.row().classes('w-full gap-4 mt-2')

            def parse_cubes(text):
                try:
                    return [float(x.strip()) for x in text.split(',') if x.strip()]
                except ValueError:
                    return []

            def run_verification():
                result_output_area.clear()
                export_buttons_area.clear()
                
                c7 = parse_cubes(c7_input.value)
                c14 = parse_cubes(c14_input.value)
                c28 = parse_cubes(c28_input.value)
                fcu_val = float(fcu_input.value)

                def evaluate_stage(cubes, ratio):
                    if not cubes or len(cubes) < 3: return None
                    mean_v = np.mean(cubes)
                    std_v = np.std(cubes, ddof=1) if len(cubes) > 1 else 0.0
                    k = 1.91 if len(cubes) < 30 else 1.64
                    fcu_char = max(mean_v - k * std_v, 0.85 * mean_v)
                    target = ratio * fcu_val
                    passed = fcu_char >= target and min(cubes) >= (0.85 * target)
                    return {"mean": mean_v, "std": std_v, "fcu": fcu_char, "target": target, "pass": passed, "count": len(cubes), "min": min(cubes)}

                s7 = evaluate_stage(c7, 0.70)
                s14 = evaluate_stage(c14, 0.85)
                s28 = evaluate_stage(c28, 1.00)

                with result_output_area:
                    with ui.column().classes('custom-card w-full'):
                        ui.label('Evaluation Results & Statistical Compliance (ECP 203)').classes('text-lg font-bold text-white')
                        
                        if s28:
                            color = 'green' if s28['pass'] else 'red'
                            ui.markdown(f"**28-Day Characteristic Strength ($f_{{cu}}$):** `{s28['fcu']:.2f} N/mm²` | **Target:** `{s28['target']} N/mm²` | **Verdict:** :{color}[**{'PASS' if s28['pass'] else 'FAIL'}**]")
                            ui.markdown(f"• **Mean Strength:** `{s28['mean']:.2f} N/mm²` | **Standard Deviation:** `{s28['std']:.2f}` | **Sample Size:** `{s28['count']}`")
                        else:
                            ui.warning('Please provide at least 3 valid cube strength values for 28-day testing.')

                        # Mix audit check
                        try:
                            cem_v = float(cement_input.value)
                            wat_v = float(water_input.value)
                            wc = wat_v / cem_v if cem_v > 0 else 0
                            ui.markdown(f"• **Calculated W/C Ratio:** `{wc:.2f}` (Max allowed under ECP 203: `0.45`)")
                        except ValueError:
                            pass

                # Export Options (PDF & CSV)
                with export_buttons_area:
                    def download_pdf_report():
                        try:
                            buffer = io.BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                            story = []
                            rep_date = datetime.date.today().strftime('%Y-%m-%d')
                            qr_buf = generate_qr_code(f"ECP 203 Audit - {project_name_input.value} - {rep_date}")

                            build_pdf_header(story, "CONCRETE CUBE STATISTICAL COMPLIANCE REPORT", "ECP 203 & ASTM Quality Assurance Verification", logo_bytes_holder['bytes'], engineer_input.value, project_name_input.value, pour_location_input.value, rep_date)
                            
                            styles = getSampleStyleSheet()
                            body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#222222"), leading=12)
                            
                            summary_text = f"""
                            <b>Specified Grade (fcu):</b> {fcu_input.value} N/mm²<br/>
                            <b>Mixer Truck No:</b> {truck_input.value} | <b>Batch Ticket ID:</b> {ticket_input.value}<br/>
                            <b>28-Day Characteristic Strength:</b> {s28['fcu']:.2f} N/mm² ({'PASS' if s28['pass'] else 'FAIL'})<br/>
                            <b>Cement Content:</b> {cement_input.value} kg/m³ | <b>Water Content:</b> {water_input.value} kg/m³
                            """
                            story.append(Paragraph(summary_text, body_style))
                            story.append(Spacer(1, 10))
                            build_pdf_footer_and_signatures(story, qr_buf)

                            doc.build(story)
                            buffer.seek(0)
                            ui.download(buffer.getvalue(), filename=f"Concrete_Audit_Report_{ticket_input.value}.pdf")
                            ui.notify('PDF Report downloaded successfully!', type='positive')
                        except Exception as e:
                            ui.notify(f'PDF Generation Error: {str(e)}', type='negative')

                    def download_csv_export():
                        df = pd.DataFrame({
                            "Parameter": ["Project", "Location", "Specified fcu", "28-Day Characteristic fcu", "Status", "Truck No", "Ticket ID", "Engineer"],
                            "Value": [
                                project_name_input.value, 
                                pour_location_input.value, 
                                str(fcu_input.value), 
                                f"{s28['fcu']:.2f}" if s28 else "N/A", 
                                "PASS" if s28 and s28['pass'] else "FAIL", 
                                truck_input.value, 
                                ticket_input.value, 
                                engineer_input.value
                            ]
                        })
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        ui.download(csv_data, filename=f"Concrete_Summary_{ticket_input.value}.csv")
                        ui.notify('CSV Summary downloaded successfully!', type='positive')

                    ui.button('📥 Download Official PDF Report', on_click=download_pdf_report).classes('primary-btn flex-1')
                    ui.button('📊 Export CSV Summary', on_click=download_csv_export).classes('primary-btn flex-1')

            ui.button('Run Compliance & Statistical Audit', on_click=run_verification).classes('primary-btn q-my-md')

        # --- TAB 2: AI MULTI-STANDARD AUDITOR ---
        with ui.tab_panel(t_audit):
            ui.label('🤖 AI Multi-Standard Engineering Auditor').classes('text-xl font-bold text-white mb-2')
            ui.markdown('Upload any PDF specification, mix design, or image to audit against **ECP 203, 202, 104, ASTM, AASHTO, BS, EN, and ISO**.')
            
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
            audit_file_upload = ui.upload(label='Select PDF or Image File', auto_upload=False).props('flat dark').classes('w-full mb-2 bg-[#1E293B] rounded-lg')
            uploaded_file_data = {'bytes': None, 'name': None, 'type': None}

            def process_audit_file():
                try:
                    if not audit_file_upload.files:
                        ui.notify('Please select a file from your device first!', type='warning')
                        return
                    file = audit_file_upload.files[0]
                    uploaded_file_data['bytes'] = file.read()
                    uploaded_file_data['name'] = file.name
                    uploaded_file_data['type'] = 'application/pdf' if file.name.lower().endswith('.pdf') else 'image/jpeg'
                    
                    audit_status_label.set_text(f'✅ File Ready: {file.name}')
                    audit_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    ui.notify(f'Successfully loaded: {file.name}', type='positive')
                except Exception as ex:
                    ui.notify(f'Error reading file: {str(ex)}', type='negative')

            ui.button('Confirm & Load File Into Memory', on_click=process_audit_file).classes('bg-slate-700 text-white font-bold text-xs mb-4 w-full')

            audit_output_container = ui.column().classes('w-full')

            def run_ai_audit():
                if not client:
                    ui.notify('Gemini API key missing in .env!', type='negative')
                    return
                if not uploaded_file_data['bytes']:
                    ui.notify('Please upload and confirm a file first!', type='warning')
                    return

                audit_output_container.clear()
                with audit_output_container:
                    ui.spinner('ios', size='lg').classes('self-center text-[#00BFFF]')
                    ui.label('Running multi-standard AI engineering audit...').classes('self-center text-sm')

                try:
                    prompt = f"""
                    You are an expert senior civil, geotechnical, and highway engineering consultant specializing in core Egyptian Codes (ECP 203, 202, 104) and international standards (ASTM, AASHTO, BS, EN, ISO).
                    Focus: {audit_focus}. Supplementary code: {supp_code_select.value}.
                    Provide a rigorous technical audit identifying compliance, code violations, risks, and required corrective actions.
                    """
                    
                    contents = [prompt]
                    if uploaded_file_data['type'] == 'application/pdf':
                        reader = pypdf.PdfReader(io.BytesIO(uploaded_file_data['bytes']))
                        text = "".join([p.extract_text() or "" for p in reader.pages])
                        contents.append(f"Extracted PDF Text:\n{text}")
                    else:
                        img_part = types.Part.from_bytes(data=uploaded_file_data['bytes'], mime_type=uploaded_file_data['type'])
                        contents.append(img_part)

                    response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=contents)
                    
                    audit_output_container.clear()
                    with audit_output_container:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Audit Findings & Compliance Breakdown').classes('text-lg font-bold text-white')
                            ui.markdown(response.text)
                except Exception as ex:
                    audit_output_container.clear()
                    with audit_output_container:
                        ui.notify(f'Error: {str(ex)}', type='negative')

            ui.button('Execute AI Audit', on_click=run_ai_audit).classes('primary-btn')

        # --- TAB 3: DEFECT DIAGNOSTIC ---
        with ui.tab_panel(t_defect):
            ui.label('🔍 AI Crack, Pavement & Geotechnical Defect Diagnostic').classes('text-xl font-bold text-white mb-2')
            ui.markdown('Upload site defect photos for automated classification and repair protocols conforming to ECP 203, ECP 104, Sika, and Fosroc standards.')
            
            defect_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
            defect_file_upload = ui.upload(label='Select Site Defect Photo (JPG/PNG)', auto_upload=False).props('flat dark').classes('w-full mb-2 bg-[#1E293B] rounded-lg')
            defect_file_data = {'bytes': None, 'type': None}

            def process_defect_file():
                try:
                    if not defect_file_upload.files:
                        ui.notify('Please select a file from your device first!', type='warning')
                        return
                    file = defect_file_upload.files[0]
                    defect_file_data['bytes'] = file.read()
                    defect_file_data['type'] = 'image/jpeg'
                    
                    defect_status_label.set_text(f'✅ File Ready: {file.name}')
                    defect_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    ui.notify(f'Successfully loaded defect image: {file.name}', type='positive')
                except Exception as ex:
                    ui.notify(f'Error reading file: {str(ex)}', type='negative')

            ui.button('Confirm & Load Defect Image', on_click=process_defect_file).classes('bg-slate-700 text-white font-bold text-xs mb-4 w-full')

            defect_output = ui.column().classes('w-full')

            def run_defect_diagnosis():
                if not client or not defect_file_data['bytes']:
                    ui.notify('API key missing or image not confirmed!', type='negative')
                    return
                defect_output.clear()
                with defect_output:
                    ui.spinner('ios', size='lg').classes('self-center text-[#00BFFF]')
                    ui.label('Analyzing defect and matching local repair products (Sika/Fosroc)...').classes('self-center text-sm')

                try:
                    img = types.Part.from_bytes(data=defect_file_data['bytes'], mime_type=defect_file_data['type'])
                    prompt = f"""
                    You are an expert forensic repair engineer following ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO.
                    Analyze this defect image. Provide:
                    1. Element Identification & Defect Classification.
                    2. Root Cause Analysis.
                    3. Applicable Standards and Remediation Procedure using local products (Sika Egypt / Fosroc).
                    """
                    response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=[prompt, img])
                    defect_output.clear()
                    with defect_output:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Forensic Diagnosis & Repair Protocol').classes('text-lg font-bold text-white')
                            ui.markdown(response.text)
                except Exception as ex:
                    defect_output.clear()
                    with defect_output:
                        ui.notify(f'Diagnosis failed: {ex}', type='negative')

            ui.button('Diagnose Defect & Get Repair Protocol', on_click=run_defect_diagnosis).classes('primary-btn')

        # --- TAB 4: AI CHATBOT ---
        with ui.tab_panel(t_chat):
            ui.label('💬 Core-Code Intelligent Assistant Chatbot').classes('text-xl font-bold text-white mb-2')
            ui.markdown('Ask any engineering, mix design, geotechnical, or pavement question based strictly on core codes (ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO).')

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
                if not user_msg.value.strip(): return
                q = user_msg.value
                chat_messages.append({"role": "user", "content": q})
                user_msg.value = ''
                render_chat()

                if not client:
                    chat_messages.append({"role": "assistant", "content": "⚠️ GEMINI_API_KEY is not configured."})
                    render_chat()
                    return

                try:
                    sys_prompt = f"You are an expert AI engineering assistant specialized in ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO. Supplementary: {supp_code_select.value}."
                    res = client.models.generate_content(model='gemini-3.5-flash-lite', contents=f"{sys_prompt}\n\nQuestion: {q}")
                    chat_messages.append({"role": "assistant", "content": res.text})
                except Exception as e:
                    chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
                render_chat()

            ui.button('Send Query', on_click=send_chat).classes('primary-btn')

        # --- TAB 5: TECHNICAL HANDBOOK ---
        with ui.tab_panel(t_handbook):
            ui.label('📖 Multi-Standard Civil Engineering Technical Handbook').classes('text-xl font-bold text-white mb-4')
            
            with ui.tabs().classes('w-full text-[#00BFFF]') as hb_tabs:
                h1 = ui.tab('ECP 203 & Concrete')
                h2 = ui.tab('ECP 202 & Soils')
                h3 = ui.tab('ECP 104 & Roads')
                h4 = ui.tab('International Standards')

            with ui.tab_panels(hb_tabs, value=h1).classes('w-full bg-transparent'):
                with ui.tab_panel(h1):
                    ui.markdown('''
                    ### Egyptian Code for Reinforced Concrete Structures (ECP 203) - Comprehensive Reference
                    * **Chapter 1: Scope & General Requirements**
                      - Governs design, material specification, batching, mixing, transport, casting, and curing of normal and high-strength concrete.
                    * **Chapter 2: Materials Specifications**
                      - **Cement:** CEM I (Ordinary Portland Cement) or CEM II conforming to ES 4756-1 / EN 197-1. Minimum cement content for structural elements exposed to severe environments is 350 kg/m³.
                      - **Aggregates:** Clean, graded coarse and fine aggregates conforming to ES 1109 / ASTM C33. Maximum aggregate size limited to 1/5 narrowest dimension or 3/4 clear spacing between rebars.
                      - **Water:** Potable water free of organic impurities, chlorides (< 500 ppm for reinforced concrete), and sulfates (< 1000 ppm).
                    * **Chapter 3: Mix Design & Characteristic Strength ($f_{{cu}}$)**
                      - Characteristic strength $f_{{cu}}$ evaluated via standard 150mm cube crushing tests at 28 days.
                      - Statistical compliance: $f_{{cu,min}} \ge f_{{cu}} + 1.64\sigma$ or verified through rolling batches with target mean margin $1.34s$ to $2.33s$.
                      - Maximum water-cement ratio ($W/C$) capped at $0.45$ for standard structural applications and $0.40$ for water-retaining structures.
                    * **Chapter 4: Construction & Curing Protocols**
                      - Continuous curing required for a minimum of 7 days using wet hessian, curing compounds, or ponding.
                      - Formwork stripping times: Sides of beams/columns (24-48 hours), soffits of slabs (7-14 days depending on span and prop conditions).
                    ''')
                with ui.tab_panel(h2):
                    ui.markdown('''
                    ### Egyptian Code for Soil Mechanics & Foundations (ECP 202) - Comprehensive Reference
                    * **Chapter 1: Subsurface Investigation & Soil Exploration**
                      - Mandatory borehole drilling, Standard Penetration Testing (SPT - ASTM D1586), Cone Penetration Testing (CPT), and undisturbed sampling for deep and shallow foundations.
                    * **Chapter 2: Shallow Foundations & Bearing Capacity**
                      - Ultimate bearing capacity calculated using Terzaghi, Meyerhof, or Hansen bearing capacity equations factoring cohesion ($c$), surcharge ($q$), and unit weight ($\gamma$).
                      - Allowable bearing capacity determined by applying a minimum Factor of Safety (F.S. = 3.0 for static loads, 2.5 for seismic/wind combinations). Total settlement limited to 25-50mm.
                    * **Chapter 3: Deep Foundations & Pile Load Testing**
                      - Bored and driven pile design including skin friction ($f_s$) and end bearing ($q_b$) evaluation.
                      - Static load testing mandated up to 2.0 times the working load in accordance with ASTM D1143 / ECP 202 specifications. Integrity testing (PIT / Sonic Logging) required on 100% of major bridge/high-rise piles.
                    * **Chapter 4: Earthworks & Compaction Control**
                      - Subgrade compaction specifications: Minimum 95% to 98% Modified Proctor Maximum Dry Density (ASTM D1557 / AASHTO T180) at optimum moisture content ($\pm 2\%$).
                    ''')
                with ui.tab_panel(h3):
                    ui.markdown('''
                    ### Egyptian Code for Roads, Highways and Airfields (ECP 104) - Comprehensive Reference
                    * **Chapter 1: Highway Geometrics & Classification**
                      - Design speed, horizontal and vertical alignment curves, superelevation, and sight distance requirements for expressways, arterial, and local roads.
                    * **Chapter 2: Subgrade & Embankment Engineering**
                      - CBR (California Bearing Ratio) testing requirements (ASTM D1883). Minimum subgrade CBR of 10% for heavy traffic loads; stabilized subgrade required if CBR < 7%.
                    * **Chapter 3: Unbound Subbase & Base Course Layers**
                      - Crushed stone aggregate base course (ABC) grading limits. Minimum relative compaction of 100% Modified Proctor. Layer thickness tolerances within $\pm 10\text{ mm}$.
                    * **Chapter 4: Bituminous Pavements & Asphalt Mix Design**
                      - Marshall Mix Design method (ASTM D6915 / AASHTO T245): Optimum bitumen content, stability, flow, air voids (3-5%), and voids in mineral aggregate (VMA).
                    ''')
                with ui.tab_panel(h4):
                    ui.markdown('''
                    ### International Engineering Standards (ASTM, AASHTO, BS, EN, ISO)
                    * **ASTM Standards:** ASTM C39 (Compressive Strength of Cylindrical Concrete Specimens), ASTM C143 (Slump Test), ASTM D1557 (Modified Proctor Compaction), ASTM D698 (Standard Proctor).
                    * **AASHTO Specifications:** AASHTO LRFD Bridge Design Specifications, AASHTO M 145 (Classification of Soils and Soil-Aggregate Mixtures for Highway Construction).
                    * **BS EN / Eurocodes:** BS EN 1992 (Eurocode 2: Design of Concrete Structures), BS EN 1997 (Eurocode 7: Geotechnical Design), BS 1881 (Testing Concrete).
                    * **ISO Quality Management:** ISO 9001 (Quality Management Systems in Construction), ISO 14001 (Environmental Management), ISO 45001 (Occupational Health & Safety).
                    ''')

    # --- PROFESSIONAL FOOTER BAR ---
    ui.markdown('''
    <div style="background-color: #1B2A4A; border-top: 2px solid #FF8C00; padding: 15px; margin-top: 30px; text-align: center; color: #FFFFFF; font-size: 12px; border-radius: 6px;">
        <b>🏗️ Multi-Standard Engineering Quality Assurance Portal</b><br>
        Automated compliance verification across ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO standards.<br>
        <b>Official Direct Contacts & Professional Network:</b><br>
        🔗 Connect on LinkedIn: <a href="https://www.linkedin.com/in/mohamed-abdalaty" target="_blank" style="color: #00BFFF; text-decoration: underline;">Mohamed Abd Al Aty</a> &nbsp;|&nbsp; ✉️ Direct Email: <a href="mailto:mohamedabdalaty63@gmail.com" style="color: #00BFFF; text-decoration: underline;">mohamedabdalaty63@gmail.com</a><br>
        <i>Specialized in Geotechnical QA/QC, Civil Engineering Standards & Automated Compliance.</i><br>
        © 2026 Eng. Mohamed Abd Al Aty. All rights reserved. Designed for ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN & ISO compliance.
    </div>
    ''')

import os
ui.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), title='Multi-Standard Engineering Auditor', favicon='🏗️', reload=False)
