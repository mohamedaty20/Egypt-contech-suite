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

# --- CUSTOM TAILWIND & HIGH-CONTRAST DARK THEME STYLING ---
app.native.window_args = {"resizable": True}

ui.add_head_html('''
<style>
    /* Global Styles & Text High Contrast Fixes */
    body, html {
        background-color: #0B0F19 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Card Container Styling */
    .custom-card {
        background-color: #151D2A !important;
        border: 1px solid #1E293B !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4) !important;
        margin-bottom: 15px !important;
    }
    
    /* Input, Select, and Textarea High-Contrast Styling */
    .q-field__native, .q-field__prefix, .q-field__suffix, .q-field__input {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    .q-field__label {
        color: #94A3B8 !important;
    }
    .q-field--outlined .q-field__control {
        background-color: #1E293B !important;
        border-color: #334155 !important;
        border-radius: 8px !important;
    }
    
    /* Fix for Problem #6: Select Dropdown Popup Options Contrast */
    .q-menu, .q-list, .q-item {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
    }
    .q-item--active, .q-item:hover {
        background-color: #334155 !important;
        color: #F59E0B !important;
    }
    
    /* Primary Action Buttons */
    .primary-btn {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border-radius: 8px !important;
        padding: 8px 20px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .primary-btn:hover {
        background: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.5) !important;
    }

    /* Tabs Styling */
    .q-tab {
        color: #94A3B8 !important;
        font-weight: 600 !important;
    }
    .q-tab--active {
        color: #38BDF8 !important;
    }
</style>
''', shared=True)

# --- AI MARKDOWN SANITIZER (FIX FOR PROBLEM #3) ---
def clean_ai_markdown(text: str) -> str:
    """Removes raw LaTeX symbols ($$, \$), ///, ***, and stray formatting artifacts from Gemini output."""
    if not text:
        return ""
    # Replace LaTeX block math $$ ... $$ with clean bold text
    cleaned = re.sub(r'\$\$\s*(.*?)\s*\$\$', r'**\1**', text, flags=re.DOTALL)
    # Replace inline LaTeX $ ... $ with clean text
    cleaned = re.sub(r'\$(.*?)\$', r'\1', cleaned)
    # Remove triple slashes /// or excess stars ***
    cleaned = re.sub(r'///+', '', cleaned)
    cleaned = re.sub(r'\*{3,}', '**', cleaned)
    # Clean up double hashes or unnecessary escaped markdown symbols
    cleaned = cleaned.replace('\\', '')
    return cleaned

# --- PDF HELPER FUNCTIONS ---
def format_markdown_for_reportlab(text):
    if not text:
        return ""
    cleaned = clean_ai_markdown(text)
    cleaned = cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', cleaned)
    cleaned = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cleaned)
    cleaned = re.sub(r'^\s*[\*\-]\s+', '&bull; ', cleaned, flags=re.MULTILINE)
    return cleaned

def build_pdf_header(story, doc_title, subtitle, logo_bytes, engineer, project, location, rep_date):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=15, textColor=colors.HexColor("#0F172A"), spaceAfter=4, alignment=1, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("DocSub", parent=styles["Normal"], fontSize=9, textColor=colors.HexColor("#475569"), spaceAfter=8, alignment=1, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#1E293B"), leading=11, fontName="Helvetica")

    if logo_bytes:
        try:
            story.append(ReportLabImage(io.BytesIO(logo_bytes), width=80, height=30))
            story.append(Spacer(1, 4))
        except Exception:
            pass

    story.append(Paragraph(format_markdown_for_reportlab(doc_title), title_style))
    story.append(Paragraph(format_markdown_for_reportlab(subtitle), sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#F59E0B"), spaceAfter=6))

    meta_html = f"""
    <b>Project:</b> {project} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Location:</b> {location}<br/>
    <b>Engineer:</b> {engineer} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Date:</b> {rep_date} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Governing Codes:</b> ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO
    """
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 8))

# --- MAIN APP LAYOUT ---
@ui.page('/')
def main_page():
    
    # --- TOP HEADER & TITLE (FIX FOR PROBLEM #4) ---
    with ui.row().classes('w-full items-center justify-between bg-[#151D2A] px-6 py-4 rounded-xl border border-[#1E293B] mb-4 shadow-lg'):
        ui.label('🏗️ Multi-Disciplinary Civil, Geotechnical & Pavement Engineering Auditor') \
            .classes('text-2xl md:text-3xl font-extrabold text-[#F59E0B] tracking-wide leading-tight')
        
        with ui.row().classes('items-center gap-2'):
            ui.icon('verified', color='sky-400').classes('text-2xl')
            ui.label('Eng. Mohamed Abd Al Aty').classes('text-base text-white font-bold tracking-wide')

    ticker_html = """
    <div style="overflow: hidden; white-space: nowrap; background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%); color: #000000; padding: 8px 0; font-weight: 800; font-size: 13px; margin-bottom: 20px; border-radius: 6px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);">
      <div style="display: inline-block; padding-left: 100%; animation: marquee 28s linear infinite;">
        🚀 CORE COMPLIANCE ACTIVE: ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; ⚠️ MULTI-DISCIPLINARY QA/QC VERIFIER &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; 🏗️ ACTIVE SITE INSPECTION PORTAL
      </div>
    </div>
    <style>
    @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
    """
    ui.add_head_html(ticker_html)

    # --- SIDEBAR CONFIGURATION ---
    with ui.left_drawer().classes('bg-[#151D2A] text-white p-5 border-r border-[#1E293B]').style('width: 350px;'):
        ui.label('PROJECT METADATA').classes('text-[#38BDF8] font-extrabold text-sm tracking-wider mb-3 uppercase')
        
        project_name_input = ui.input(label='Project Name', value='Highway Expansion Project').props('outlined dark').classes('w-full mb-3')
        pour_location_input = ui.input(label='Structural Element / Chainage', value='Highway Section Ch. 12+500').props('outlined dark').classes('w-full mb-4')

        ui.label('Governing Standards Core').classes('text-[#F59E0B] font-bold text-sm mb-1')
        ui.label('ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO').classes('text-xs text-slate-300 italic mb-3')
        
        # Fix for Problem #6 (Select menu styling)
        supp_code_select = ui.select(
            label='Supplementary Standard',
            options=[
                "None (Strictly Core Standards)",
                "ACI 318-25 — Structural Concrete",
                "IBC — International Building Code",
                "BS EN 1992 / Eurocode 2 + UK Annex",
                "AASHTO LRFD Bridge Design"
            ],
            value="None (Strictly Core Standards)"
        ).props('outlined dark options-dark').classes('w-full mb-4')

        fcu_input = ui.number(label='Specified 28-Day Grade fcu (N/mm²)', value=30.0, step=5.0).props('outlined dark').classes('w-full mb-4')
        
        ui.label('Batch Plant & Site Logs').classes('text-[#38BDF8] font-bold text-sm mb-2 uppercase tracking-wide')
        truck_input = ui.input(label='Mixer Truck No.', value='TRK-104').props('outlined dark').classes('w-full mb-2')
        ticket_input = ui.input(label='Batch Ticket ID', value='BT-99482').props('outlined dark').classes('w-full mb-4')

        ui.label('Mix Design Parameters').classes('text-[#F59E0B] font-bold text-sm mb-2 uppercase tracking-wide')
        cement_input = ui.input(label='Cement Content (kg/m³)', value='350.0').props('outlined dark').classes('w-full mb-2')
        water_input = ui.input(label='Free Water Content (kg/m³)', value='150.0').props('outlined dark').classes('w-full mb-4')
        
        engineer_input = ui.input(label='Engineer Name', value='Eng. Mohamed Abd Al Aty').props('outlined dark').classes('w-full mb-4')
        
        logo_bytes_holder = {'bytes': None}
        def handle_logo(e):
            logo_bytes_holder['bytes'] = e.content.read()
            ui.notify('Company logo loaded!', type='positive')

        ui.label('Company Logo').classes('text-xs text-slate-400 mb-1')
        logo_upload = ui.upload(auto_upload=True, on_upload=handle_logo).props('flat dark').classes('w-full bg-[#1E293B] rounded-lg')

        # --- RESTORED LINKEDIN PROFILE FOOTER (FIX FOR PROBLEM #2) ---
        ui.separator().classes('my-6 bg-slate-700')
        with ui.column().classes('w-full items-center text-center gap-1'):
            ui.label('Lead Structural & QA Consultant').classes('text-xs text-slate-400 font-medium')
            ui.link('🔗 Eng. Mohamed Abd Al Aty on LinkedIn', 'https://www.linkedin.com/in/mohamed-abd-al-aty', new_tab=True) \
                .classes('text-[#F59E0B] font-bold text-sm hover:text-[#38BDF8] transition-colors')

    # --- TABS / SCREENS NAVIGATION ---
    with ui.tabs().classes('w-full text-white bg-[#151D2A] rounded-xl border border-[#1E293B] p-1 mb-4') as tabs:
        t_dash = ui.tab('📊 Concrete Verifier Dashboard', icon='dashboard')
        t_audit = ui.tab('🤖 AI Multi-Standard Auditor', icon='psychology')
        t_defect = ui.tab('🔍 Defect Diagnostic', icon='search')
        t_chat = ui.tab('💬 AI Chatbot', icon='chat')
        t_handbook = ui.tab('📖 Technical Codes Handbook', icon='book')

    with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent text-white'):
        
        # --- TAB 1: CONCRETE VERIFIER DASHBOARD ---
        with ui.tab_panel(t_dash):
            ui.label('1. Input Cube Crushing Results (N/mm²)').classes('text-xl font-bold text-[#F59E0B] mb-3')
            
            with ui.row().classes('w-full gap-4 mb-4'):
                with ui.column().classes('custom-card flex-1'):
                    ui.label('7-Day Cubes').classes('font-bold text-[#38BDF8]')
                    c7_input = ui.textarea(value='21.0, 22.5, 20.5').props('outlined dark').classes('w-full')
                with ui.column().classes('custom-card flex-1'):
                    ui.label('14-Day Cubes').classes('font-bold text-[#38BDF8]')
                    c14_input = ui.textarea(value='26.0, 27.2, 25.8').props('outlined dark').classes('w-full')
                with ui.column().classes('custom-card flex-1'):
                    ui.label('28-Day Cubes').classes('font-bold text-[#38BDF8]')
                    c28_input = ui.textarea(value='32.5, 34.0, 31.0, 35.5, 29.0, 33.0').props('outlined dark').classes('w-full')

            result_output_area = ui.column().classes('w-full')

            def parse_cubes(text):
                try:
                    return [float(x.strip()) for x in text.split(',') if x.strip()]
                except ValueError:
                    return []

            def run_verification():
                result_output_area.clear()
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

                s28 = evaluate_stage(c28, 1.00)

                with result_output_area:
                    with ui.column().classes('custom-card w-full'):
                        ui.label('Evaluation Results & Statistical Compliance (ECP 203)').classes('text-lg font-bold text-[#38BDF8]')
                        
                        if s28:
                            color = '#10B981' if s28['pass'] else '#EF4444'
                            verdict_text = 'PASS (COMPLIANT)' if s28['pass'] else 'FAIL (NON-COMPLIANT)'
                            ui.markdown(f"**28-Day Characteristic Strength ($f_{{cu}}$):** `{s28['fcu']:.2f} N/mm²` | **Target:** `{s28['target']} N/mm²`")
                            ui.label(f"Verdict: {verdict_text}").style(f"color: {color}; font-weight: 800; font-size: 1.1rem;")
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

            ui.button('Run Compliance & Statistical Audit', on_click=run_verification).classes('primary-btn q-my-md')

        # --- TAB 2: AI MULTI-STANDARD AUDITOR (FIX FOR PROBLEM #1 & #3) ---
        with ui.tab_panel(t_audit):
            ui.label('🤖 AI Multi-Standard Engineering Auditor').classes('text-xl font-bold text-[#F59E0B] mb-2')
            ui.label('Upload any PDF specification, mix design, or site photo to audit against core codes (ECP 203, 202, 104, ASTM, AASHTO, BS, EN, ISO).').classes('text-sm text-slate-300 mb-4')
            
            audit_focus = ui.select(
                label='Audit Focus',
                options=[
                    "Multi-Standard Structural & Geotechnical Compliance",
                    "Roads, Pavements & Subgrade Materials (ECP 104 & AASHTO)",
                    "Soil Mechanics & Foundations (ECP 202 & ASTM / ISO)",
                    "Reinforced Concrete Structures (ECP 203 & ACI / BS EN)"
                ],
                value="Multi-Standard Structural & Geotechnical Compliance"
            ).props('outlined dark options-dark').classes('w-full mb-4')

            uploaded_file_data = {'bytes': None, 'name': None, 'type': None}
            audit_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')

            def handle_audit_upload(e):
                try:
                    uploaded_file_data['bytes'] = e.content.read()
                    uploaded_file_data['name'] = e.name
                    uploaded_file_data['type'] = 'application/pdf' if e.name.lower().endswith('.pdf') else 'image/jpeg'
                    audit_status_label.set_text(f'✅ File Ready: {e.name}')
                    audit_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    ui.notify(f'Successfully loaded: {e.name}', type='positive')
                except Exception as ex:
                    ui.notify(f'Upload error: {str(ex)}', type='negative')

            # Fixed Upload Handler with auto_upload=True
            audit_file_upload = ui.upload(
                label='Upload Document or Image (PDF/PNG/JPG)',
                auto_upload=True,
                on_upload=handle_audit_upload
            ).props('flat dark').classes('w-full mb-2 bg-[#1E293B] rounded-lg')

            audit_output_container = ui.column().classes('w-full mt-4')

            def run_ai_audit():
                if not client:
                    ui.notify('Gemini API key missing in .env file!', type='negative')
                    return
                if not uploaded_file_data['bytes']:
                    ui.notify('Please upload a file first!', type='warning')
                    return

                audit_output_container.clear()
                with audit_output_container:
                    ui.spinner('ios', size='lg').classes('self-center text-[#38BDF8]')
                    ui.label('Running multi-standard AI engineering audit...').classes('self-center text-sm text-slate-300')

                try:
                    prompt = f"""
                    You are an expert senior civil, geotechnical, and highway engineering consultant specializing in core Egyptian Codes (ECP 203, 202, 104) and international standards (ASTM, AASHTO, BS, EN, ISO).
                    Focus: {audit_focus}. Supplementary code: {supp_code_select.value}.
                    Provide a rigorous technical audit identifying compliance, code violations, structural risks, and required corrective actions.
                    Do not output raw LaTeX ($$), write clear standard markdown.
                    """
                    
                    contents = [prompt]
                    if uploaded_file_data['type'] == 'application/pdf':
                        reader = pypdf.PdfReader(io.BytesIO(uploaded_file_data['bytes']))
                        text = "".join([p.extract_text() or "" for p in reader.pages])
                        contents.append(f"Extracted PDF Text:\n{text}")
                    else:
                        img_part = types.Part.from_bytes(data=uploaded_file_data['bytes'], mime_type=uploaded_file_data['type'])
                        contents.append(img_part)

                    # Configured explicitly for gemini-3.5-flash-lite
                    response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=contents)
                    
                    audit_output_container.clear()
                    with audit_output_container:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Audit Findings & Compliance Breakdown').classes('text-lg font-bold text-[#38BDF8] mb-2')
                            # Clean AI Output (Fix for Problem #3)
                            ui.markdown(clean_ai_markdown(response.text))
                except Exception as ex:
                    audit_output_container.clear()
                    with audit_output_container:
                        ui.notify(f'Audit execution failed: {str(ex)}', type='negative')

            ui.button('Execute AI Audit', on_click=run_ai_audit).classes('primary-btn mt-2')

        # --- TAB 3: DEFECT DIAGNOSTIC ---
        with ui.tab_panel(t_defect):
            ui.label('🔍 AI Crack, Pavement & Geotechnical Defect Diagnostic').classes('text-xl font-bold text-[#F59E0B] mb-2')
            ui.label('Upload site defect photos for automated classification and repair protocols conforming to ECP 203, ECP 104, Sika, and Fosroc standards.').classes('text-sm text-slate-300 mb-4')
            
            defect_file_data = {'bytes': None, 'name': None, 'type': None}
            defect_status_label = ui.label('Status: No defect photo loaded').classes('text-xs text-amber-400 font-semibold mb-2')

            def handle_defect_upload(e):
                try:
                    defect_file_data['bytes'] = e.content.read()
                    defect_file_data['name'] = e.name
                    defect_file_data['type'] = 'image/png' if e.name.lower().endswith('.png') else 'image/jpeg'
                    defect_status_label.set_text(f'✅ Photo Ready: {e.name}')
                    defect_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    ui.notify('Defect image loaded successfully!', type='positive')
                except Exception as ex:
                    ui.notify(f'Upload error: {str(ex)}', type='negative')

            defect_upload = ui.upload(
                label='Upload Site Defect Photo',
                auto_upload=True,
                on_upload=handle_defect_upload
            ).props('flat dark').classes('w-full mb-2 bg-[#1E293B] rounded-lg')

            defect_output = ui.column().classes('w-full mt-4')

            def run_defect_diagnosis():
                if not client or not defect_file_data['bytes']:
                    ui.notify('Please upload a defect photo first!', type='warning')
                    return
                defect_output.clear()
                with defect_output:
                    ui.spinner('ios', size='lg').classes('self-center text-[#38BDF8]')
                    ui.label('Analyzing defect and matching local repair products (Sika / Fosroc)...').classes('self-center text-sm text-slate-300')

                try:
                    img = types.Part.from_bytes(data=defect_file_data['bytes'], mime_type=defect_file_data['type'])
                    prompt = f"""
                    You are an expert forensic repair engineer following ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, and ISO.
                    Analyze this defect image. Provide:
                    1. Element Identification & Defect Classification.
                    2. Root Cause Analysis.
                    3. Applicable Standards and Remediation Procedure using local products (Sika Egypt / Fosroc).
                    Write clean markdown without LaTeX symbols.
                    """
                    response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=[prompt, img])
                    defect_output.clear()
                    with defect_output:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Forensic Diagnosis & Repair Protocol').classes('text-lg font-bold text-[#38BDF8] mb-2')
                            ui.markdown(clean_ai_markdown(response.text))
                except Exception as ex:
                    defect_output.clear()
                    with defect_output:
                        ui.notify(f'Diagnosis failed: {ex}', type='negative')

            ui.button('Diagnose Defect & Get Repair Protocol', on_click=run_defect_diagnosis).classes('primary-btn mt-2')

        # --- TAB 4: AI CHATBOT ---
        with ui.tab_panel(t_chat):
            ui.label('💬 Core-Code Intelligent Assistant Chatbot').classes('text-xl font-bold text-[#F59E0B] mb-2')
            ui.label('Ask any engineering, mix design, geotechnical, or pavement question based strictly on core codes.').classes('text-sm text-slate-300 mb-4')

            chat_container = ui.column().classes('custom-card w-full h-96 overflow-y-auto mb-4 border border-[#1E293B] p-4')
            
            chat_messages = [{"role": "assistant", "content": "Hello! I am your Multi-Standard Engineering Assistant. How can I assist you with your civil, geotechnical, structural, or concrete queries today?"}]

            def render_chat():
                chat_container.clear()
                with chat_container:
                    for msg in chat_messages:
                        bg = 'bg-[#1E293B] border-l-4 border-[#38BDF8]' if msg['role'] == 'assistant' else 'bg-[#334155] border-l-4 border-[#F59E0B]'
                        with ui.column().classes(f'w-full p-3 rounded-r-lg mb-3 {bg}'):
                            ui.label(f"{msg['role'].capitalize()}").classes('text-xs font-bold text-slate-400 uppercase tracking-wider mb-1')
                            ui.markdown(clean_ai_markdown(msg['content']))

            render_chat()

            user_msg = ui.input(placeholder='Type your engineering question here...').props('outlined dark').classes('w-full mb-3')

            def send_chat():
                if not user_msg.value.strip(): return
                q = user_msg.value
                chat_messages.append({"role": "user", "content": q})
                user_msg.value = ''
                render_chat()

                if not client:
                    chat_messages.append({"role": "assistant", "content": "⚠️ GEMINI_API_KEY is not configured in .env."})
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

        # --- TAB 5: TECHNICAL HANDBOOK (EXPANDED FIX FOR PROBLEM #5) ---
        with ui.tab_panel(t_handbook):
            ui.label('📖 Multi-Standard Civil Engineering Technical Code Handbook').classes('text-2xl font-bold text-[#F59E0B] mb-2')
            ui.label('Comprehensive reference library for governing Egyptian and International engineering codes, laws, and testing criteria.').classes('text-sm text-slate-300 mb-6')
            
            with ui.tabs().classes('w-full text-white bg-[#1E293B] rounded-lg') as hb_tabs:
                h1 = ui.tab('ECP 203 (Concrete)')
                h2 = ui.tab('ECP 202 (Geotechnical)')
                h3 = ui.tab('ECP 104 (Roads & Pavements)')
                h4 = ui.tab('International Standards')

            with ui.tab_panels(hb_tabs, value=h1).classes('w-full bg-transparent text-white mt-4'):
                
                # ECP 203 Sections
                with ui.tab_panel(h1):
                    with ui.expansion('Chapter 1: Materials & Testing Standards (150mm Cube Acceptance)', icon='science').classes('custom-card w-full'):
                        ui.markdown('''
                        * **Characteristic Strength ($f_{cu}$):** Defined as the strength below which not more than 5% of test results are expected to fall at 28 days using standard 150x150x150 mm cubes.
                        * **Acceptance Criteria (ECP 203):**
                            * Average strength of any 3 consecutive cubes must equal or exceed $f_{cu} + 3 \text{ N/mm}^2$.
                            * Individual test strength must not fall below $f_{cu} - 3 \text{ N/mm}^2$.
                        * **Sampling Rate:** Minimum 1 sample (set of 6 cubes) per 50 m³ poured or per batching shift.
                        ''')
                    
                    with ui.expansion('Chapter 2: Durability & Mix Design Requirements', icon='opacity').classes('custom-card w-full'):
                        ui.markdown('''
                        * **Maximum W/C Ratio:** 0.45 for severe exposure (foundations/marine), 0.50 for moderate exposure.
                        * **Minimum Cement Content:** 350 kg/m³ for reinforced concrete foundations; 300 kg/m³ for superstructure elements.
                        * **Maximum Allowable Chlorides & Sulfates:** Total water-soluble chloride content shall not exceed 0.3% by weight of cement.
                        ''')

                    with ui.expansion('Chapter 3: Structural Detailing & Rebar Cover Laws', icon='grid_view').classes('custom-card w-full'):
                        ui.markdown('''
                        * **Minimum Concrete Cover:**
                            * Foundations in contact with soil: 50 mm (75 mm if cast against unblinded ground).
                            * Exposed Beams & Columns: 25 mm - 35 mm.
                            * Interior Slabs: 20 mm.
                        * **Lap Splice Lengths:** Minimum $40 \phi$ to $60 \phi$ depending on stress zones (tension/compression) and concrete grade.
                        ''')

                # ECP 202 Sections
                with ui.tab_panel(h2):
                    with ui.expansion('Chapter 1: Subsurface Soil Investigations & Boreholes', icon='layers').classes('custom-card w-full'):
                        ui.markdown('''
                        * **Borehole Spacing & Depth:**
                            * Spacing: 15m to 25m grid for residential/commercial structures; 30m to 50m along linear road projects.
                            * Minimum Depth: Must extend into sound bearing strata (at least 1.5 to 2.0 times the footing width below foundation level).
                        * **Standard Penetration Test (SPT):** Evaluates soil density/consistency. N-values recorded every 1.5m.
                        ''')
                    
                    with ui.expansion('Chapter 2: Bearing Capacity & Settlement Criteria', icon='line_weight').classes('custom-card w-full'):
                        ui.markdown('''
                        * **Factor of Safety (FOS):** Minimum FOS = 2.5 to 3.0 for ultimate bearing capacity under static loading.
                        * **Allowable Total Settlement:** Maximum 25 mm for isolated footings; 50 mm for raft/mat foundations on granular soils.
                        ''')

                # ECP 104 Sections
                with ui.tab_panel(h3):
                    with ui.expansion('Chapter 1: Subgrade Preparation & Compaction Quality Control', icon='minor_crash').classes('custom-card w-full'):
                        ui.markdown('''
                        * **California Bearing Ratio (CBR):** Minimum subgrade CBR = 8% at 95% Modified Proctor Density.
                        * **Field Density Testing:** Minimum 98% Maximum Dry Density (MDD) according to AASHTO T180 / ASTM D1557 for subbase and base courses.
                        ''')
                    
                    with ui.expansion('Chapter 2: Bituminous Asphalt Layers QA/QC', icon='edit_road').classes('custom-card w-full'):
                        ui.markdown('''
                        * **Marshall Stability Requirements:** Minimum 8000 N (815 kg) for heavy traffic base/binder layers; 9000 N for wearing surface layers.
                        * **Air Voids Content:** Designed strictly between 3.0% and 5.0% for dense-graded asphalt mixtures.
                        ''')

                # International Standards
                with ui.tab_panel(h4):
                    with ui.expansion('ASTM & AASHTO Testing Equivalencies', icon='public').classes('custom-card w-full'):
                        ui.markdown('''
                        * **ASTM C39:** Standard Test Method for Compressive Strength of Cylindrical Concrete Specimens (Cylinder to Cube factor $\approx 0.80$).
                        * **AASHTO M145:** Standard Classification of Soils and Soil-Aggregate Mixtures for Highway Construction (A-1 to A-7 groups).
                        * **BS EN 1992 (Eurocode 2):** European standard for reinforced concrete design and environmental exposure classes (XC, XS, XD, XF).
                        ''')

# --- RUN NICEGUI APP ---
import os
ui.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), title='Multi-Standard Engineering Auditor', favicon='🏗️', reload=False)
