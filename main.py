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
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = text.replace('$', '').replace('\\ge', '>=').replace('\\le', '<=')
    text = re.sub(r'#+\s*(.*)', r'<font color="#1B2A4A"><b>\1</b></font><br/>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = text.replace('\n', '<br/>')
    return text

def build_pdf_header(story, doc_title, subtitle, logo_bytes, engineer, project, location, rep_date, ticket_id, unique_hash):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1B2A4A"), spaceAfter=4, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("DocSub", parent=styles["Normal"], fontSize=8.5, textColor=colors.HexColor("#FF8C00"), spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", parent=styles["Normal"], fontSize=7.5, textColor=colors.HexColor("#333333"), leading=10, fontName="Helvetica")

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
    body_style = ParagraphStyle("BodyStyle", parent=styles["Normal"], fontSize=7.5, textColor=colors.HexColor("#222222"), leading=9, fontName="Helvetica")
    sec_style = ParagraphStyle("SecTitle", parent=styles["Heading2"], fontSize=8.5, textColor=colors.HexColor("#1B2A4A"), spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold")

    story.append(Spacer(1, 4))
    story.append(Paragraph("<b>Corporate Engineering Approvals & Compliance Sign-Off</b>", sec_style))
    
    qr_lab_img = ReportLabImage(qr_img_buffer, width=40, height=40)
    sign_cell_1 = Paragraph("<b>Prepared By:</b><br/>QA/QC Engineer:<br/><br/>_________________", body_style)
    sign_cell_2 = Paragraph("<b>Technical Director:</b><br/>Chief Engineer:<br/><br/>_________________", body_style)
    sign_cell_3 = Paragraph("<b>Client / Consultant:</b><br/>Official Stamp:<br/><br/>_________________", body_style)
    qr_cell = [Paragraph("<b>QR Verification:</b>", body_style), qr_lab_img]

    t_sign = Table([[sign_cell_1, sign_cell_2, sign_cell_3, qr_cell]], colWidths=[135, 135, 135, 95])
    t_sign.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (3, 0), (3, 0), "CENTER"),
    ]))
    story.append(t_sign)

# --- MAIN APP LAYOUT ---
@ui.page('/')
def main_page():
    with ui.row().classes('w-full items-center justify-between bg-[#1B2A4A] px-6 py-3 rounded-lg border border-[#FF8C00] mb-4'):
        ui.label('Multi-Disciplinary Civil, Geotechnical & Pavement Engineering Auditor').classes('text-xl font-bold text-white')
        ui.label('Eng. Mohamed Abd Al Aty').classes('text-sm text-[#00BFFF] font-semibold')

    ticker_html = """
    <div style="overflow: hidden; white-space: nowrap; background-color: #FF8C00; color: #031338; padding: 6px 0; font-weight: bold; font-size: 13px; margin-bottom: 15px; border-radius: 4px;">
      <div style="display: inline-block; padding-left: 100%; animation: marquee 25s linear infinite;">
        Core Compliance Active: ECP 203, ECP 202, ECP 104, ASTM, AASHTO, BS, EN, ISO &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Advanced Geotechnical & Concrete Calculation Sheet &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp; Active Site Inspection Portal
      </div>
    </div>
    <style>
    @keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }
    </style>
    """
    ui.add_head_html(ticker_html)

    # --- SIDEBAR CONFIGURATION ---
    supp_code_select = None
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
            ui.label('Professional Concrete Cube Calculation Sheet & Statistical Verifier (ECP 203)').classes('text-2xl font-bold text-white mb-4')
            
            # Compact Professional Inputs (Issue 5 Fixed)
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

            # Stage Selector Popup/Menu (Issue 2 Fixed)
            stage_selector = ui.select(
                label='Select Stage to Display Table & Details',
                options=['All Stages', '7-Day Stage', '14-Day Stage', '28-Day Stage'],
                value='All Stages'
            ).classes('w-full md:w-1/3 mb-4')

            result_output_area = ui.column().classes('w-full')
            export_buttons_area = ui.row().classes('w-full gap-4 mt-4')

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
                selected_stage = stage_selector.value

                def evaluate_stage(cubes, ratio):
                    if not cubes or len(cubes) < 3: return None
                    mean_v = float(np.mean(cubes))
                    std_v = float(np.std(cubes, ddof=1)) if len(cubes) > 1 else 0.0
                    k = 1.91 if len(cubes) < 30 else 1.64
                    fcu_char = max(mean_v - k * std_v, 0.85 * mean_v)
                    target = ratio * fcu_val
                    passed = fcu_char >= target and min(cubes) >= (0.85 * target)
                    return {"mean": mean_v, "std": std_v, "fcu": fcu_char, "target": target, "pass": passed, "count": len(cubes), "min": min(cubes), "max": max(cubes), "values": cubes}

                s7 = evaluate_stage(c7, 0.70)
                s14 = evaluate_stage(c14, 0.85)
                s28 = evaluate_stage(c28, 1.00)

                with result_output_area:
                    with ui.column().classes('custom-card w-full'):
                        ui.label('Official Engineering Calculation Sheet & Statistical Summary (ECP 203 / ASTM C39)').classes('text-xl font-bold text-white mb-3')
                        
                        if s28:
                            color = 'green' if s28['pass'] else 'red'
                            verdict_text = 'PASS — FULLY COMPLIANT' if s28['pass'] else 'FAIL — NON-COMPLIANT WITH ECP 203 LIMITS'
                            ui.markdown(f"### Overall 28-Day Compliance Verdict: :{color}[**{verdict_text}**]")
                            
                            # Cleaned Mathematical Formulation (Issue 1 Fixed - no weird symbols)
                            ui.markdown(f"""
                            #### Mathematical Formulation & Statistical Breakdown (28-Day)
                            * **Specified Characteristic Strength (f_cu):** `{fcu_val:.2f} N/mm²`
                            * **Calculated Characteristic Strength (f_cu,act):** max(Mean - k * sigma, 0.85 * Mean) = **`{s28['fcu']:.2f} N/mm²`**
                            * **Statistical Arithmetic Mean (Mean):** `{s28['mean']:.2f} N/mm²` (Sample count: `{s28['count']}`)
                            * **Standard Deviation (sigma):** `{s28['std']:.2f} N/mm²` (Bessel's correction N-1)
                            * **Safety Multiplier (k):** `1.91` for sample size `n = {s28['count']}` (ECP 203 Table 8-2).
                            * **Minimum Individual Cube Value:** `{s28['min']:.2f} N/mm²` (Required threshold >= `0.85 * f_cu = {0.85 * fcu_val:.2f} N/mm²`).
                            * **Maximum Individual Cube Value:** `{s28['max']:.2f} N/mm²`.
                            """)

                        # Multi-Stage Tables based on popup selection (Issue 2 Fixed)
                        stages_to_show = []
                        if selected_stage == 'All Stages':
                            stages_to_show = [('7-Day Stage', s7, 0.70), ('14-Day Stage', s14, 0.85), ('28-Day Stage', s28, 1.00)]
                        elif selected_stage == '7-Day Stage':
                            stages_to_show = [('7-Day Stage', s7, 0.70)]
                        elif selected_stage == '14-Day Stage':
                            stages_to_show = [('14-Day Stage', s14, 0.85)]
                        elif selected_stage == '28-Day Stage':
                            stages_to_show = [('28-Day Stage', s28, 1.00)]

                        for title, stage_data, ratio in stages_to_show:
                            if stage_data:
                                ui.label(f'Individual Test Specimen Breakdown ({title}) — Target Ratio: {int(ratio*100)}%').classes('font-bold text-white text-sm mt-4 mb-1')
                                table_rows = []
                                target_req = ratio * fcu_val
                                for idx, val in enumerate(stage_data['values'], 1):
                                    dev = val - stage_data['mean']
                                    status = "Acceptable" if val >= (0.85 * target_req) else "Below Limit"
                                    table_rows.append({
                                        "Specimen No": f"Cube #{idx}",
                                        "Crushing Load (N/mm2)": f"{val:.2f} N/mm²",
                                        "Deviation from Mean": f"{dev:+.2f} N/mm²",
                                        "Evaluation": status
                                    })
                                
                                ui.table(
                                    columns=[
                                        {"name": "Specimen No", "label": "Specimen No", "field": "Specimen No", "align": "left"},
                                        {"name": "Crushing Load (N/mm2)", "label": "Crushing Load (N/mm²)", "field": "Crushing Load (N/mm2)"},
                                        {"name": "Deviation from Mean", "label": "Deviation from Mean", "field": "Deviation from Mean"},
                                        {"name": "Evaluation", "label": "Evaluation", "field": "Evaluation"},
                                    ],
                                    rows=table_rows,
                                    row_key="Specimen No"
                                ).classes('w-full bg-[#1E222D] text-white mb-2')

                        try:
                            cem_v = float(cement_input.value)
                            wat_v = float(water_input.value)
                            wc = wat_v / cem_v if cem_v > 0 else 0
                            ui.markdown(f"""
                            **Durability & Mix Proportion Verification:**
                            * **Water-Cement Ratio (W/C):** `{wc:.2f}` (Calculated as `{wat_v} kg/m³` water / `{cem_v} kg/m³` cement). Maximum permissible W/C under ECP 203 is `0.45`.
                            * **Cement Content Compliance:** `{cem_v} kg/m³` (Minimum required for standard structural exposure is `350 kg/m³`).
                            """)
                        except ValueError:
                            pass

                        # Evolution Line Chart
                        stages = ['7-Day', '14-Day', '28-Day', 'Target Grade']
                        means = [
                            s7['mean'] if s7 else 0,
                            s14['mean'] if s14 else 0,
                            s28['mean'] if s28 else 0,
                            fcu_val
                        ]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=stages, y=means, mode='lines+markers+text',
                            text=[f"{m:.1f}" for m in means], textposition="top center",
                            line=dict(color='#00BFFF', width=3), marker=dict(size=10, color='#FF8C00')
                        ))
                        fig.add_hline(y=fcu_val, line_dash="dash", line_color="#22C55E", annotation_text=f"Target f_cu ({fcu_val} N/mm²)", annotation_position="bottom right")
                        
                        fig.update_layout(
                            title='Compressive Strength Evolution & Target Threshold',
                            template='plotly_dark',
                            paper_bgcolor='#1B2A4A',
                            plot_bgcolor='#1B2A4A',
                            margin=dict(t=40, b=20, l=40, r=20),
                            height=300
                        )
                        ui.plotly(fig).classes('w-full mt-4')

                        # Scatter Chart for all samples across stages (Issue 3 & 4 Fixed)
                        fig_scatter = go.Figure()
                        for stage_name, cubes, color in [('7 Days', c7, '#8B5CF6'), ('14 Days', c14, '#F97316'), ('28 Days', c28, '#10B981')]:
                            if cubes:
                                fig_scatter.add_trace(go.Scatter(
                                    x=[f"{stage_name} - #{i}" for i in range(1, len(cubes)+1)],
                                    y=cubes,
                                    mode='markers+text',
                                    name=stage_name,
                                    text=[f"{v:.1f}" for v in cubes],
                                    textposition="top center",
                                    marker=dict(size=12, color=color)
                                ))
                        fig_scatter.update_layout(
                            title='Individual Cube Strengths Scatter Plot (All Stages)',
                            template='plotly_dark',
                            paper_bgcolor='#1B2A4A',
                            plot_bgcolor='#1B2A4A',
                            margin=dict(t=40, b=40, l=40, r=20),
                            height=330
                        )
                        ui.plotly(fig_scatter).classes('w-full mt-4')

                # Populate export buttons (Issue 6 Fixed - comprehensive PDF export)
                with export_buttons_area:
                    unique_uid = f"ECP-{uuid.uuid4().hex[:8].upper()}"

                    def download_pdf_report():
                        try:
                            buffer = io.BytesIO()
                            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
                            story = []
                            rep_date = datetime.date.today().strftime('%Y-%m-%d')
                            qr_buf = generate_qr_code(f"UID: {unique_uid} | ECP 203 Calculation Sheet - {project_name_input.value} - {rep_date}")

                            build_pdf_header(
                                story, 
                                "CONCRETE CUBE STATISTICAL CALCULATION SHEET", 
                                "Official ECP 203 & ASTM C39 Quality Assurance Report", 
                                logo_bytes_holder['bytes'], 
                                engineer_input.value, 
                                project_name_input.value, 
                                pour_location_input.value, 
                                rep_date,
                                ticket_input.value,
                                unique_uid
                            )
                            
                            styles = getSampleStyleSheet()
                            body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#222222"), leading=12)
                            
                            summary_html = f"""
                            <b>Specified Grade (f_cu):</b> {fcu_input.value} N/mm²<br/>
                            <b>Mixer Truck No:</b> {truck_input.value} &nbsp;|&nbsp; <b>Batch Ticket ID:</b> {ticket_input.value}<br/>
                            <b>Cement Content:</b> {cement_input.value} kg/m³ &nbsp;|&nbsp; <b>Free Water Content:</b> {water_input.value} kg/m³<br/>
                            <b>7-Day Mean:</b> {s7['mean']:.2f} N/mm² ({len(c7)} cubes)<br/>
                            <b>14-Day Mean:</b> {s14['mean']:.2f} N/mm² ({len(c14)} cubes)<br/>
                            <b>28-Day Characteristic Strength:</b> {s28['fcu']:.2f} N/mm² &nbsp;|&nbsp; <b>Verdict:</b> {'PASS' if s28['pass'] else 'FAIL'}<br/>
                            <b>Statistical Mean (28-Day):</b> {s28['mean']:.2f} N/mm² &nbsp;|&nbsp; <b>Standard Deviation (sigma):</b> {s28['std']:.2f}
                            """
                            story.append(Paragraph(summary_html, body_style))
                            story.append(Spacer(1, 8))
                            build_pdf_footer_and_signatures(story, qr_buf)

                            doc.build(story)
                            buffer.seek(0)
                            ui.download(buffer.getvalue(), filename=f"Concrete_Calculation_Sheet_{ticket_input.value}.pdf")
                            ui.notify('Official Calculation Sheet PDF downloaded successfully!', type='positive')
                        except Exception as e:
                            ui.notify(f'PDF Generation Error: {str(e)}', type='negative')

                    def download_csv_export():
                        df = pd.DataFrame({
                            "Calculation Field": ["Project Name", "Location", "Specified f_cu", "28-Day Characteristic f_cu", "Compliance Verdict", "Truck No", "Batch Ticket ID", "Engineer", "Verification UID"],
                            "Value": [
                                project_name_input.value, 
                                pour_location_input.value, 
                                str(fcu_input.value), 
                                f"{s28['fcu']:.2f}" if s28 else "N/A", 
                                "PASS" if s28 and s28['pass'] else "FAIL", 
                                truck_input.value, 
                                ticket_input.value, 
                                engineer_input.value,
                                unique_uid
                            ]
                        })
                        csv_data = df.to_csv(index=False).encode('utf-8')
                        ui.download(csv_data, filename=f"Concrete_Calculation_Sheet_{ticket_input.value}.csv")
                        ui.notify('CSV Calculation Sheet downloaded successfully!', type='positive')

                    ui.button('Download Official Calculation PDF', on_click=download_pdf_report).classes('primary-btn flex-1')
                    ui.button('Export Calculation CSV', on_click=download_csv_export).classes('primary-btn flex-1')

            ui.button('Run Statistical Calculation & Verification', on_click=run_verification).classes('primary-btn q-my-md')

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
                    Perform a comprehensive, rigorous technical audit of the provided document or image. Structure your report using clear markdown sections and multiple data tables.
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
                    audit_result_text = response.text
                    
                    audit_output_container.clear()
                    with audit_output_container:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Master Engineering Audit Findings & Code Compliance Report').classes('text-xl font-bold text-white mb-2')
                            ui.markdown(audit_result_text)

                    # Persistent Export Buttons (Issue 7 Fixed)
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
                                
                                sanitized_text = clean_for_reportlab(audit_result_text)
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
                                "Value": [project_name_input.value, audit_focus, uploaded_file_data['name'], engineer_input.value, unique_uid, audit_result_text[:300].replace('\n', ' ')]
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

            def run_defect_diagnosis():
                if not client or not defect_file_data['bytes']:
                    ui.notify('API key missing or image not uploaded!', type='negative')
                    return
                defect_output.clear()
                with defect_output:
                    ui.spinner('ios', size='lg').classes('self-center text-[#00BFFF]')
                    ui.label('Analyzing defect and matching local repair products (Sika/Fosroc)...').classes('self-center text-sm')

                try:
                    img = types.Part.from_bytes(data=defect_file_data['bytes'], mime_type=defect_file_data['type'])
                    prompt = "Perform forensic structural evaluation and list repair products (Sika/Fosroc) complying with ECP 203 and ASTM."
                    response = client.models.generate_content(model='gemini-3.5-flash-lite', contents=[prompt, img])
                    defect_output.clear()
                    with defect_output:
                        with ui.column().classes('custom-card w-full'):
                            ui.label('Forensic Diagnosis & Repair Protocol').classes('text-xl font-bold text-white mb-2')
                            ui.markdown(response.text)
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
                    res = client.models.generate_content(
                        model='gemini-3.5-flash-lite', 
                        contents=q,
                        config=types.GenerateContentConfig(temperature=0.2)
                    )
                    chat_messages.append({"role": "assistant", "content": res.text})
                except Exception as e:
                    chat_messages.append({"role": "assistant", "content": f"Error: {str(e)}"})
                render_chat()

            ui.button('Send Query', on_click=send_chat).classes('primary-btn')

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
