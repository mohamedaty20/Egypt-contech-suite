import os
from dotenv import load_dotenv
from fastapi import FastAPI
from nicegui import app, ui

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# --- CUSTOM TAILWIND STYLING & THEME CONFIG ---
app.native.window_args = {"resizable": True}

# Custom CSS injection with shared=True for global scope
ui.add_head_html('''
<style>
    body {
        background-color: #f8fafc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .custom-card {
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        background-color: white;
        padding: 24px;
    }
</style>
''', shared=True)

@ui.page('/')
def index():
    # --- HEADER / NAVIGATION BAR ---
    with ui.row().classes('w-full items-center justify-between bg-slate-900 text-white px-6 py-4 shadow-md'):
        with ui.row().classes('items-center gap-3'):
            ui.icon('engineering', size='2rem').classes('text-blue-400')
            ui.label('Egypt ConTech Suite').classes('text-xl font-bold tracking-wide')
        with ui.row().classes('gap-4 text-sm text-slate-300'):
            ui.label('Market: Egypt (EGP / ECP 203)').classes('bg-slate-800 px-3 py-1 rounded-full border border-slate-700')

    # --- MAIN CONTAINER ---
    with ui.column().classes('w-full max-w-6xl mx-auto p-6 gap-6'):
        
        # Welcome Banner
        with ui.column().classes('custom-card w-full border-l-4 border-blue-500'):
            ui.label('Professional Technical Office Workspace').classes('text-2xl font-bold text-slate-800')
            ui.label('Automate BOQ extraction, ECP 203 compliance checks, and structural crack inspections instantly with zero hallucination.').classes('text-slate-600')

        # --- TABS INTERFACE FOR MODULES ---
        with ui.tabs().classes('w-full text-blue-600') as tabs:
            boq_tab = ui.tab('BOQ Extractor', icon='table_chart')
            ecp_tab = ui.tab('ECP 203 Checker', icon='verified')
            crack_tab = ui.tab('AI Crack Inspector', icon='broken_image')

        with ui.tab_panels(tabs, value=boq_tab).classes('w-full bg-transparent'):
            
            # --- MODULE 1: BOQ EXTRACTOR ---
            with ui.tab_panel(boq_tab):
                with ui.column().classes('custom-card w-full gap-4'):
                    ui.label('1. Automated Bill of Quantities (BOQ) Extractor').classes('text-lg font-semibold text-slate-800')
                    ui.label('Upload your structural PDF drawings or tables to parse materials and line items instantly.').classes('text-sm text-slate-500')
                    
                    with ui.row().classes('w-full gap-4 items-center'):
                        ui.upload(on_upload=lambda e: ui.notify(f'Uploaded: {e.name}')).classes('max-w-md')
                        ui.button('Extract BOQ via AI', on_click=lambda: ui.notify('Processing blueprint data...')).classes('bg-blue-600 text-white')
                    
                    # Placeholder Results Table
                    ui.label('Extracted Items Preview:').classes('font-medium text-slate-700 mt-2')
                    ui.table(
                        columns=[
                            {'name': 'item', 'label': 'Item Description', 'field': 'item', 'align': 'left'},
                            {'name': 'unit', 'label': 'Unit', 'field': 'unit'},
                            {'name': 'qty', 'label': 'Quantity', 'field': 'qty'},
                            {'name': 'rate', 'label': 'Est. Rate (EGP)', 'field': 'rate'},
                        ],
                        rows=[
                            {'item': 'Concrete Grade R/C 350', 'unit': 'm3', 'qty': 120, 'rate': 4200},
                            {'item': 'High Tensile Steel Rebar', 'unit': 'Ton', 'qty': 14.5, 'rate': 38000},
                        ],
                    ).classes('w-full')

            # --- MODULE 2: ECP 203 CHECKER ---
            with ui.tab_panel(ecp_tab):
                with ui.column().classes('custom-card w-full gap-4'):
                    ui.label('2. Egyptian Code of Practice (ECP 203) Compliance Engine').classes('text-lg font-semibold text-slate-800')
                    with ui.row().classes('w-full gap-4'):
                        ui.number(label='Column Width (b) [mm]', value=300).classes('flex-1')
                        ui.number(label='Column Depth (t) [mm]', value=600).classes('flex-1')
                        ui.number(label='Ultimate Load (Pu) [kN]', value=1500).classes('flex-1')
                    ui.button('Verify ECP 203 Limits', on_click=lambda: ui.notify('Running deterministic safety check...')).classes('bg-slate-800 text-white')

            # --- MODULE 3: AI CRACK INSPECTOR ---
            with ui.tab_panel(crack_tab):
                with ui.column().classes('custom-card w-full gap-4'):
                    ui.label('3. AI Crack Inspector & Local Remediation Report').classes('text-lg font-semibold text-slate-800')
                    ui.label('Upload a site photo of concrete cracking to assess severity and generate local repair protocols (Sika/Fosroc).').classes('text-sm text-slate-500')
                    ui.upload(on_upload=lambda e: ui.notify(f'Image loaded: {e.name}')).classes('max-w-md')
                    ui.button('Generate Remediation Report', on_click=lambda: ui.notify('Analyzing image and matching local compounds...')).classes('bg-emerald-600 text-white')

# Run the app locally
ui.run(host='127.0.0.1', port=8080, title='Egypt ConTech Suite', favicon='🏗️')