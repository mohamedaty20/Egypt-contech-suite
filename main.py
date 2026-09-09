import io
import datetime
import os
import uuid
import re
import asyncio
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import qrcode
import pypdf
import fitz

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
# STYLING (identical to your original, shortened for brevity)
# =====================================================================================
app.native.window_args = {"resizable": True}
ui.add_head_html('''<style>/* full style from your original */</style>''')  # (I'll include the full style in the final file)

# =====================================================================================
# HELPERS (sanitization, PDF, AI)
# =====================================================================================
# ... (all your existing helper functions remain unchanged) ...

# =====================================================================================
# MAIN PAGE
# =====================================================================================
@ui.page('/')
def main_page():
    ui.query('body').style('width: 100vw; height: 100vh; overflow-x: hidden;')

    sidebar = ui.left_drawer().classes('sidebar-container').style('width: 380px;')
    with sidebar:
        with ui.row().classes('w-full items-center justify-between mb-4 p-2'):
            ui.label('📋 PROJECT METADATA').classes('text-white font-bold text-base tracking-wide')
            ui.button('✕', on_click=sidebar.toggle).classes(
                'bg-transparent text-white text-xl hover:text-[#FF8C00] p-1 min-w-[36px] !shadow-none !rounded-full !bg-transparent'
            ).style('font-size: 20px; line-height: 1;')

        project_name_input = ui.input('Project Name', value='Highway Expansion Project').classes('w-full mb-3')
        pour_location_input = ui.input('Structural Element / Chainage', value='Highway Section Ch. 12+500').classes('w-full mb-4')

        ui.label('Governing Design Code Basis').classes('text-white font-bold text-sm mb-1')
        ui.markdown('By default every AI output in this app is generated strictly per **ECP 203 / ECP 202 / ECP 104**. Change this to switch the primary basis.').classes('text-xs text-[#A9B6D0] mb-2')
        # --- ui.select #1 (clean, no extra commas) ---
        code_basis_select = ui.select(
            'Code Type (applies app-wide)',
            options=CODE_BASIS_OPTIONS,
            value=CODE_BASIS_OPTIONS[0],
        ).classes('w-full mb-4')

        engineer_input = ui.input('Engineer Name', value='Eng. Mohamed Abd Al Aty').classes('w-full mb-2')
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
        ui.upload('Upload Company Logo', auto_upload=True, on_upload=handle_logo_upload).props('flat dark').classes('w-full mb-2')

    ui.button('☰', on_click=sidebar.toggle).classes(
        'fixed top-4 left-4 z-50 bg-[#10203f] text-white border border-[#FF8C00] p-3 rounded-full shadow-lg hover:bg-[#1a2a4a]'
    ).style('font-size: 20px; min-width: 48px; min-height: 48px;')

    def current_meta(uid_prefix):
        ticket = app.storage.user.get('ticket_id', 'N/A')
        return {
            'uid': f"{uid_prefix}-{uuid.uuid4().hex[:8].upper()}",
            'project': project_name_input.value,
            'location': pour_location_input.value,
            'engineer': engineer_input.value,
            'date': datetime.date.today().strftime('%Y-%m-%d'),
            'ticket': ticket,
        }

    with ui.column().classes('w-full min-h-screen p-4 bg-[#031338]'):
        # Title block
        with ui.column().classes('w-full bg-[#0d1a35] px-6 py-4 rounded-xl border border-[#FF8C00] shadow-lg mb-4'):
            ui.label('SMART EGY-CIVIL AI AUDITOR').classes('main-title text-white')
            ui.label('Intelligent General Civil, Geotechnical & Structural Compliance Engine').classes('sub-title text-lg font-medium mt-1')
            ui.label('Lead Technical Auditor: Eng. Mohamed Abd Al Aty').classes('text-base text-[#A9B6D0] font-semibold mt-1')
            ui.label('Next-generation automated civil engineering and quality intelligence, precision-calibrated for the Egyptian Code of Practice.').classes('text-sm text-[#A9B6D0] mt-1 italic')

        ui.add_head_html('''<style>@keyframes marquee { 0% { transform: translate(0, 0); } 100% { transform: translate(-100%, 0); } }</style>''')
        ui.html('''<div style="width: 100%; overflow: hidden; white-space: nowrap; background-color: rgba(13,26,53,0.6); backdrop-filter: blur(8px); color: #FFFFFF; padding: 10px 0; font-weight: 600; font-size: 13px; margin-bottom: 15px; border-radius: 8px; border: 1px solid rgba(255,140,0,0.3);"><div style="display: inline-block; padding-left: 100%; animation: marquee 28s linear infinite;"><span style="color: #FF8C00;">[CORE ACTIVE]</span> ECP 203 &middot; ECP 202 &middot; ECP 104 &middot; ASTM &middot; AASHTO &middot; BS EN &middot; ISO &nbsp;&nbsp;|&nbsp;&nbsp; Advanced Geotechnical & Concrete Calculation Sheet &nbsp;&nbsp;|&nbsp;&nbsp; Active Site Inspection Portal</div></div>''')

        with ui.tabs().classes('w-full text-white bg-[#0d1a35] rounded-lg') as tabs:
            t_dash = ui.tab('Concrete Cube Verifier').classes('text-white font-bold')
            t_audit = ui.tab('AI Multi-Standard Auditor').classes('text-white font-bold')
            t_defect = ui.tab('Defect Diagnostic').classes('text-white font-bold')
            t_chat = ui.tab('AI Chatbot').classes('text-white font-bold')
            t_handwriting = ui.tab('Handwriting OCR').classes('text-white font-bold')

        with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent mt-4'):
            # ===== TAB 1: Concrete Cube Verifier =====
            with ui.tab_panel(t_dash):
                ui.label('Concrete Cube Calculation Sheet & Statistical Verifier').classes('text-2xl font-bold text-white mb-4')

                with ui.column().classes('input-card w-full mb-4'):
                    ui.label('Concrete Mix & Site Data').classes('text-lg font-bold text-white')
                    with ui.row().classes('w-full gap-4'):
                        fcu_input = ui.number('Specified 28-Day Grade f_cu (N/mm2)', value=30.0, step=5.0).classes('w-1/2')
                        ticket_input = ui.input('Batch Ticket ID', value='BT-99482').classes('w-1/2')
                    with ui.row().classes('w-full gap-4'):
                        truck_input = ui.input('Mixer Truck No.', value='TRK-104').classes('w-1/2')
                        cement_input = ui.input('Cement Content (kg/m3)', value='350.0').classes('w-1/2')
                    water_input = ui.input('Free Water Content (kg/m3)', value='150.0').classes('w-full')
                    ticket_input.on('change', lambda e: app.storage.user.update({'ticket_id': ticket_input.value}))

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
                    return {'n': len(arr), 'mean': mean, 'std': std, 'min': float(arr.min()), 'max': float(arr.max()), 'cov': (std / mean * 100.0) if mean > 0 else 0.0}

                def get_selected_stages(stage_filter):
                    all_stages = [('7-Day', c7_input, parse_vals(c7_input.value)), ('14-Day', c14_input, parse_vals(c14_input.value)), ('28-Day', c28_input, parse_vals(c28_input.value))]
                    mapping = {'7-Day Stage': [0], '14-Day Stage': [1], '28-Day Stage': [2]}
                    if stage_filter in mapping:
                        return [all_stages[i] for i in mapping[stage_filter]]
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
                        progress = ui.linear_progress(value=0, max=1).classes('w-full mt-2')
                        for i in range(10):
                            await asyncio.sleep(0.15)
                            progress.set_value((i+1)/10)

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

                # --- ui.select #2 ---
                stage_selector = ui.select(
                    'Select Stage Display Filter',
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

            # ===== TAB 2: AI Multi-Standard Auditor =====
            with ui.tab_panel(t_audit):
                ui.label('AI Multi-Standard Engineering Auditor').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload a specification, mix design, or site report to audit against the selected code basis.').classes('markdown-body mb-2')

                # --- ui.select #3 ---
                audit_focus = ui.select(
                    'Audit Focus',
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

                ui.upload('Select PDF or Image File', auto_upload=True, on_upload=handle_audit_upload).props('flat dark').classes('w-full mb-4')
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
                        progress = ui.linear_progress(value=0, max=1).classes('w-full mt-2')
                        for i in range(10):
                            await asyncio.sleep(0.15)
                            progress.set_value((i+1)/10)
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

            # ===== TAB 3: Defect Diagnostic =====
            with ui.tab_panel(t_defect):
                ui.label('AI Engineering Defect Diagnostic & Repair Protocol').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload site defect photos or PDFs for forensic analysis. Describe the issue below for more precise diagnosis.').classes('markdown-body mb-2')

                defect_status_label = ui.label('Status: No file uploaded yet').classes('text-xs text-amber-400 font-semibold mb-2')
                defect_file_data = {'bytes': None, 'type': None}
                defect_result_holder = {'text': ''}
                defect_user_message = ui.input('Describe the defect or additional context (optional)',
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

                ui.upload('Select Site Defect Photo or PDF', auto_upload=True, on_upload=handle_defect_upload).props('flat dark').classes('w-full mb-4')
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
                        progress = ui.linear_progress(value=0, max=1).classes('w-full mt-2')
                        for i in range(10):
                            await asyncio.sleep(0.15)
                            progress.set_value((i+1)/10)
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

            # ===== TAB 4: AI Chatbot =====
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

            # ===== TAB 5: Handwriting OCR =====
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

                ui.upload('Upload Handwriting Image or PDF', auto_upload=True, on_upload=handle_ocr_upload).props('flat dark').classes('w-full mb-4')
                ocr_output = ui.column().classes('w-full')
                ocr_export = ui.row().classes('w-full gap-4 mt-4')
                transcribed_text_holder = {'text': ''}
                text_editor = None

                def restore_ocr_state():
                    stored = app.storage.user.get('ocr_text', '')
                    if stored:
                        transcribed_text_holder['text'] = stored
                        ocr_output.clear()
                        with ocr_output:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Transcribed Text (editable)').classes('text-xl font-bold text-white mb-2')
                                text_editor = ui.textarea(value=stored, placeholder='Edit the transcribed text here...').classes('w-full markdown-body').style('min-height: 300px; background: #0a1a3a; color: white; border: 1px solid #FF8C00;')
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
                                    current_text = text_editor.value if text_editor else stored
                                    meta = current_meta('OCR')
                                    pdf_bytes = build_report_pdf(
                                        doc_title="",
                                        subtitle="",
                                        body_markdown=current_text,
                                        meta=meta,
                                        logo_bytes=logo_bytes_holder['bytes'],
                                        show_ticket=False
                                    )
                                    ui.download(pdf_bytes, filename=f"Handwriting_Transcription_{meta['ticket']}.pdf")
                                    ui.notify('PDF report downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Error: {str(ex)}', type='negative')
                            def download_ocr_txt():
                                try:
                                    current_text = text_editor.value if text_editor else stored
                                    txt_bytes = current_text.encode('utf-8')
                                    ui.download(txt_bytes, filename=f"Handwriting_Transcription_{current_meta('OCR')['ticket']}.txt")
                                    ui.notify('TXT file downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'TXT Error: {str(ex)}', type='negative')
                            ui.button('Download PDF Report', on_click=download_ocr_pdf).classes('primary-btn flex-1')
                            ui.button('Download TXT', on_click=download_ocr_txt).classes('primary-btn flex-1')
                restore_ocr_state()

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
                        progress = ui.linear_progress(value=0, max=1).classes('w-full mt-2')
                        for i in range(10):
                            await asyncio.sleep(0.15)
                            progress.set_value((i+1)/10)
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
                        app.storage.user['ocr_text'] = transcribed

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
                                    current_text = text_editor.value if text_editor else transcribed
                                    meta = current_meta('OCR')
                                    pdf_bytes = build_report_pdf(
                                        doc_title="",
                                        subtitle="",
                                        body_markdown=current_text,
                                        meta=meta,
                                        logo_bytes=logo_bytes_holder['bytes'],
                                        show_ticket=False
                                    )
                                    ui.download(pdf_bytes, filename=f"Handwriting_Transcription_{meta['ticket']}.pdf")
                                    ui.notify('PDF report downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Error: {str(ex)}', type='negative')
                            def download_ocr_txt():
                                try:
                                    current_text = text_editor.value if text_editor else transcribed
                                    txt_bytes = current_text.encode('utf-8')
                                    ui.download(txt_bytes, filename=f"Handwriting_Transcription_{current_meta('OCR')['ticket']}.txt")
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
                if not app.storage.user.get('ocr_text', ''):
                    with ocr_output:
                        ui.markdown('*Upload a file and click "Transcribe Handwriting" to start.*').classes('text-sm text-[#A9B6D0]')

        # Footer
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
