# ui/pages.py – Complete fixed file
import io
import json
import datetime
import os
from ezdxf import recover
import uuid
import traceback
import asyncio
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import fitz
import pypdf
import ezdxf
from nicegui import app, ui, run
from google.genai import types
from reportlab.platypus import Paragraph, Spacer

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
from services.boq import (
    extract_elements as boq_extract_elements,
    process_walls as boq_process_walls,
    process_rooms as boq_process_rooms,
    compute_boq as boq_compute_boq,
    PARAM_DEFAULTS as BOQ_PARAM_DEFAULTS,
    extract_elements_from_image as boq_extract_from_image,
)
from services.scraper_service import scrape_jobs, detect_mime_type, RAPIDAPI_KEY
from utils.boq import (
    normalize_keys,
    compute_architectural_quantities,
    generate_arch_boq_table,
    compute_mass_from_ai_data,
    generate_boq_table,
    compute_rebar_quantities,
    generate_charts,
)


# =====================================================================
# ADDITIVE HIGH-TRAFFIC LAYER — PART 1
# =====================================================================

_GEMINI_MAX_CONCURRENT = int(os.environ.get("GEMINI_MAX_CONCURRENT", "8"))
_CPU_MAX_CONCURRENT    = int(os.environ.get(
    "CPU_MAX_CONCURRENT",
    str(max(1, (os.cpu_count() or 2) - 1)),
))
_GEMINI_TIMEOUT_S      = float(os.environ.get("GEMINI_TIMEOUT_S", "240"))

_GEMINI_SEM = None
_CPU_SEM    = None


def _get_gemini_sem():
    global _GEMINI_SEM
    if _GEMINI_SEM is None:
        _GEMINI_SEM = asyncio.Semaphore(_GEMINI_MAX_CONCURRENT)
    return _GEMINI_SEM


def _get_cpu_sem():
    global _CPU_SEM
    if _CPU_SEM is None:
        _CPU_SEM = asyncio.Semaphore(_CPU_MAX_CONCURRENT)
    return _CPU_SEM


def bytes_to_stream(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    buf = io.BytesIO(data)
    buf.seek(0)
    return buf


def stream_to_bytes(buf):
    buf.seek(0)
    return buf.getvalue()


def _call_with_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)


async def cpu_bound_limited(fn, *args, **kwargs):
    sem = _get_cpu_sem()
    async with sem:
        if kwargs:
            return await run.cpu_bound(_call_with_kwargs, fn, args, kwargs)
        return await run.cpu_bound(fn, *args)


async def gemini_limited(coro_factory, timeout=None):
    sem = _get_gemini_sem()
    t = timeout or _GEMINI_TIMEOUT_S
    async with sem:
        return await asyncio.wait_for(coro_factory(), timeout=t)


async def io_bound_limited(fn, *args, **kwargs):
    sem = _get_cpu_sem()
    async with sem:
        if kwargs:
            return await run.io_bound(_call_with_kwargs, fn, args, kwargs)
        return await run.io_bound(fn, *args)


_DXF_BINARY_MAGIC = b"AutoCAD Binary DXF\r\n\x1a\x00"


def _strip_thumbnail_section(text: str) -> str:
    lines = text.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i < n:
        if (i + 3 < n
                and lines[i].strip() == '0'
                and lines[i + 1].strip() == 'SECTION'
                and lines[i + 2].strip() == '2'
                and lines[i + 3].strip().upper() == 'THUMBNAILIMAGE'):
            i += 4
            while i < n:
                if (lines[i].strip() == '0'
                        and i + 1 < n
                        and lines[i + 1].strip() == 'ENDSEC'):
                    i += 2
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)


def _open_dxf_doc_from_bytes(doc_bytes):
    if isinstance(doc_bytes, str):
        doc_bytes = doc_bytes.encode('utf-8')

    head = doc_bytes[:32]
    is_binary = head.startswith(b'AutoCAD Binary DXF') or (b'\x00' in head)
    print(f"[dxf] head={head!r} binary={is_binary}")

    if is_binary:
        try:
            return ezdxf.read(io.BytesIO(doc_bytes))
        except Exception as e:
            print(f"[dxf] binary read failed: {e!r}; trying recover")
            from ezdxf import recover as _recover
            return _recover.read(io.BytesIO(doc_bytes))

    try:
        text = doc_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = doc_bytes.decode('latin-1', errors='replace')

    cleaned = _strip_thumbnail_section(text)

    try:
        return ezdxf.read(io.StringIO(cleaned))
    except Exception as e:
        print(f"[dxf] ascii read failed: {e!r}; trying recover")
        from ezdxf import recover as _recover
        try:
            return _recover.read(io.StringIO(cleaned))
        except Exception as e2:
            print(f"[dxf] recover(StringIO) failed: {e2!r}; trying raw bytes")
            return _recover.read(io.BytesIO(doc_bytes))


def _extract_areas_from_dxf_bytes(doc_bytes, unit, workflow):
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    return extract_areas_from_dxf(doc, unit=unit, workflow=workflow)


# =====================================================================
# BOQ WORKERS (module-level, picklable)
# =====================================================================
def _boq_run_dxf(doc_bytes, units, ext_t_mm, int_t_mm,
                 floor_h_m, door_h_m, win_h_m, wet_tile_m,
                 layer_overrides=None):
    from services.boq import (
        extract_elements as _ex,
        process_walls as _pw,
        process_rooms as _pr,
        compute_boq as _cb,
    )
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    records, _stats = _ex(doc, units=units, layer_overrides=layer_overrides)
    walls = _pw(records, ext_thick=ext_t_mm, int_thick=int_t_mm)
    rooms = _pr(records)
    params = {
        'ext_wall_thickness_m': ext_t_mm / 1000.0,
        'int_wall_thickness_m': int_t_mm / 1000.0,
        'floor_height_m':       floor_h_m,
        'door_height_m':        door_h_m,
        'window_height_m':      win_h_m,
        'wet_tile_height_m':    wet_tile_m,
    }
    result = _cb(walls, rooms, records, params=params)
    return {
        'records': records,
        'walls':   walls,
        'rooms':   rooms,
        'boq':     result,
        'info':    {
            'kind':    'dxf',
            'ext_len': walls['stats']['external_len_m'],
            'int_len': walls['stats']['internal_len_m'],
            'n_rooms': rooms['stats']['total'],
        },
    }


def _pdf_first_page_to_png_bytes(pdf_bytes, zoom=2.0):
    import fitz as _fitz
    doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if len(doc) == 0:
            return None
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=_fitz.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()
def _probe_unknown_layers_dxf(doc_bytes):
    """Return sorted list of layer names the taxonomy could not classify."""
    from services.boq import collect_unknown_layers as _cu
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    return _cu(doc)


def _list_layers_dxf(doc_bytes):
    """Return [{layer, count, category, subtype, confidence, matched}, ...]."""
    from services.boq import list_all_layers as _ll
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    return _ll(doc)


# =====================================================================
# AI layout planner (fallback version)
# =====================================================================
async def plan_architectural_layout(plot_data):
    n_bed    = int(plot_data.get('num_bedrooms', 3) or 3)
    n_bath   = int(plot_data.get('num_bathrooms', 2) or 2)
    user_req = (plot_data.get('user_description') or '').strip()
    area     = float(plot_data.get('plot_area_m2', 200) or 200)
    scale    = max(0.8, min(1.6, area / 200.0))

    if client:
        try:
            prompt = f"""
You are an architect specialising in Egyptian residential buildings.
Design an interior room layout for a family home on a plot of {area:.0f} m2
with {n_bed} bedrooms and {n_bath} bathrooms.
User's special requests: "{user_req or 'none'}"

Return ONLY a JSON object of this exact shape (no prose, no markdown):
{{
  "rooms": [
    {{"name": "Living Room", "zone": "public", "area_m2": 25, "needs_window": true, "priority": 1}},
    {{"name": "Kitchen",     "zone": "public", "area_m2": 10, "needs_window": true, "priority": 2}}
  ]
}}

Rules:
- Provide exactly {n_bed} bedrooms and {n_bath} bathrooms, plus Living Room,
  Kitchen, and one Corridor / Hall.
- zone must be "public" or "private".
- areas must sum to roughly 75% of the plot area.
"""
            raw = await call_gemini_json(prompt, temperature=0.2, timeout=120)
            raw = raw.strip()
            if raw.startswith('```'):
                raw = raw.split('```')[1]
                if raw.startswith('json'):
                    raw = raw[4:]
            start = raw.find('{')
            end = raw.rfind('}')
            if start != -1 and end != -1:
                plan = json.loads(raw[start:end + 1])
                if isinstance(plan.get('rooms'), list) and plan['rooms']:
                    return plan
        except Exception as e:
            print(f"[plan_architectural_layout] AI failed, using fallback: {e}")

    rooms = [
        {"name": "Living Room", "zone": "public",  "area_m2": round(28 * scale, 1),
         "needs_window": True,  "priority": 1},
        {"name": "Kitchen",     "zone": "public",  "area_m2": round(11 * scale, 1),
         "needs_window": True,  "priority": 2},
        {"name": "Dining",      "zone": "public",  "area_m2": round(12 * scale, 1),
         "needs_window": True,  "priority": 3},
    ]
    for i in range(1, n_bed + 1):
        label = "Master Bedroom" if i == 1 else f"Bedroom {i}"
        rooms.append({
            "name": label, "zone": "private",
            "area_m2": round((18 if i == 1 else 14) * scale, 1),
            "needs_window": True, "priority": 10 + i,
        })
    for i in range(1, n_bath + 1):
        rooms.append({
            "name": f"Bathroom {i}", "zone": "private",
            "area_m2": round(4.5 * scale, 1),
            "needs_window": False, "priority": 30 + i,
        })
    rooms.append({
        "name": "Corridor", "zone": "public",
        "area_m2": round(8 * scale, 1),
        "needs_window": False, "priority": 50,
    })
    return {"rooms": rooms}


# =====================================================================
# Local room positioning (grid-based)
# =====================================================================
def _position_rooms(rooms, plot_data):
    from services.floor_plan_grid import build_grid as _bg

    grid = _bg(plot_data)
    all_cells = grid['cells']
    stair = next((c for c in all_cells if c.get('is_stair')), None)
    cells = [c for c in all_cells if not c.get('is_stair')]

    if not cells:
        return {'building': grid['building'], 'corridor': grid['corridor'],
                'entry_wall': grid['entry_wall'], 'stair_cell': stair,
                'rooms': []}

    def _rtype(nm):
        n = (nm or '').lower()
        if 'master' in n: return 'bedroom_master'
        if 'bath' in n or 'wc' in n or 'toilet' in n: return 'bathroom'
        if 'kitchen' in n: return 'kitchen'
        if 'living' in n or 'reception' in n or 'mandara' in n: return 'living'
        if 'dining' in n: return 'dining'
        if 'bedroom' in n: return 'bedroom'
        if 'corridor' in n or 'hall' in n or 'entry' in n or 'foyer' in n: return 'entry'
        return 'bedroom'

    lower_rooms, upper_rooms = [], []
    for r in rooms:
        nm = (r.get('name') or '').lower()
        if nm == 'corridor' or nm.startswith('corridor '):
            continue
        rt = _rtype(r.get('name', ''))
        if r.get('zone') == 'private' or rt in ('bedroom', 'bedroom_master', 'bathroom'):
            upper_rooms.append(r)
        else:
            lower_rooms.append(r)

    lower_cells = [c for c in cells if c['side'] == 'lower']
    upper_cells = [c for c in cells if c['side'] == 'upper']

    placed = []

    def _fill(room_list, cell_list):
        for i, cell in enumerate(cell_list):
            if i < len(room_list):
                r = room_list[i]
                nm = r.get('name', f'Room {i+1}')
                rt = _rtype(nm)
            else:
                nm = f"Room {cell['id']}"
                rt = 'bedroom'
            win_wall = 'S' if cell['side'] == 'lower' else 'N'
            placed.append({
                'name': nm,
                'type': rt,
                'x': cell['x'], 'y': cell['y'],
                'w': cell['w'], 'h': cell['h'],
                'door_wall': cell.get('door_wall'),
                'door_pos': cell['w'] // 2,
                'window_walls': [win_wall] if rt != 'bathroom' else [],
            })

    _fill(lower_rooms, lower_cells)
    _fill(upper_rooms, upper_cells)

    return {
        'building': grid['building'],
        'corridor': grid['corridor'],
        'entry_wall': grid['entry_wall'],
        'stair_cell': stair,
        'rooms': placed,
    }


# =====================================================================
# Excel / PDF helpers
# =====================================================================
def process_excel_file(file_bytes, filename):
    try:
        df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
        df.columns = df.columns.str.lower().str.strip()
        progress_cols = [c for c in df.columns if 'progress' in c or 'percent' in c]
        date_cols     = [c for c in df.columns if 'date' in c]
        desc_cols     = [c for c in df.columns if 'desc' in c or 'note' in c]
        cat_cols      = [c for c in df.columns if 'cat'  in c or 'type' in c]
        loc_cols      = [c for c in df.columns if 'loc'  in c or 'area' in c]
        date_col      = date_cols[0]     if date_cols     else None
        desc_col      = desc_cols[0]     if desc_cols     else None
        progress_col  = progress_cols[0] if progress_cols else None
        cat_col       = cat_cols[0]      if cat_cols      else None
        loc_col       = loc_cols[0]      if loc_cols      else None
        keep_cols = [c for c in [date_col, desc_col, progress_col, cat_col, loc_col] if c]
        if keep_cols:
            df = df[keep_cols]
        rename_map = {}
        if date_col:     rename_map[date_col]     = 'date'
        if desc_col:     rename_map[desc_col]     = 'description'
        if progress_col: rename_map[progress_col] = 'progress_percent'
        if cat_col:      rename_map[cat_col]      = 'category'
        if loc_col:      rename_map[loc_col]      = 'location'
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
    from reportlab.platypus import (
        SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph,
        HRFlowable, Image as ReportLabImage,
    )
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

    start_str = pdf_data['start_date'].strftime('%Y-%m-%d') if pdf_data['start_date'] else 'N/A'
    end_str   = pdf_data['end_date'].strftime('%Y-%m-%d')   if pdf_data['end_date']   else 'N/A'
    meta_html = f"""
    <b>Project:</b> {project_name}<br/>
    <b>Engineer:</b> {engineer_name} &nbsp;|&nbsp; <b>Range:</b> {start_str} to {end_str}<br/>
    <b>Phase:</b> {pdf_data['description']}<br/>
    <b>Report UID:</b> <font color="#CC0000"><b>{unique_uid}</b></font>
    """
    right_cell = ReportLabImage(io.BytesIO(logo_bytes), width=70, height=32) if logo_bytes else ""
    try:
        t_head = Table([[Paragraph("<b>PROGRESS TRACKING REPORT</b>", title_style), right_cell],
                        [Paragraph("Consolidated Progress Summary", sub_style), ""],
                        [Paragraph(meta_html, meta_style), ""]],
                       colWidths=[USABLE_WIDTH - 100, 100])
        t_head.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0)]))
        story.append(t_head)
    except Exception:
        story.append(Paragraph("PROGRESS TRACKING REPORT", title_style))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3, color=colors.HexColor("#FF8C00"), spaceAfter=8))

    df = pdf_data['df'].copy()
    if not df.empty:
        cols_to_show = [c for c in df.columns if c in
                        ['date', 'description', 'progress_percent', 'category', 'location']]
        if 'source' in df.columns:
            cols_to_show.append('source')
        df_display = df[cols_to_show].fillna('')
        if 'date' in df_display.columns:
            df_display['date'] = df_display['date'].apply(
                lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
        table_data = [cols_to_show]
        for _, row in df_display.iterrows():
            table_data.append([str(row[col]) for col in cols_to_show])
        if len(table_data) > 20:
            table_data = table_data[:20]
        col_widths = [USABLE_WIDTH / len(cols_to_show)] * len(cols_to_show)
        t = Table(table_data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B2A4A')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#94A3B8')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    for fig in [pdf_data.get('fig_bar'), pdf_data.get('fig_scatter'),
                pdf_data.get('fig_pie'), pdf_data.get('fig_line')]:
        if fig:
            try:
                img_bytes = fig.to_image(format="png", width=400, height=300, scale=2)
                story.append(ReportLabImage(io.BytesIO(img_bytes),
                                            width=USABLE_WIDTH * 0.45,
                                            height=USABLE_WIDTH * 0.45 * 0.75))
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


def generate_autocad_pdf(info, boq_df, engineer_name, project_name, logo_bytes, ticket_id):
    """Simple, bulletproof layout PDF. No BOQ table."""
    from reportlab.platypus import (
        SimpleDocTemplate, Spacer, Paragraph, HRFlowable,
        Image as ReportLabImage,
    )
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=PAGE_WIDTH, rightMargin=MARGIN,
                             leftMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    styles = build_pdf_styles()
    story = []
    unique_uid = f"LAYOUT-{uuid.uuid4().hex[:8].upper()}"

    def _safe(k, default=''):
        try:
            v = info.get(k, default)
            return v if v is not None else default
        except Exception:
            return default

    title_style = ParagraphStyle("DocTitle", fontSize=14,
                                  textColor=colors.HexColor("#1B2A4A"),
                                  spaceAfter=3, fontName="Helvetica-Bold", leading=17)
    sub_style = ParagraphStyle("DocSub", fontSize=9,
                                textColor=colors.HexColor("#B45309"),
                                spaceAfter=6, fontName="Helvetica-Bold")
    meta_style = ParagraphStyle("MetaStyle", fontSize=8,
                                 textColor=colors.HexColor("#334155"),
                                 leading=11.5, fontName="Helvetica")

    meta_html = (
        f"<b>Project:</b> {project_name}<br/>"
        f"<b>Engineer:</b> {engineer_name}<br/>"
        f"<b>Plot Area:</b> {_safe('plot_area', 0)} m2 &nbsp;|&nbsp; "
        f"<b>Street:</b> {_safe('street_width', 0)} m<br/>"
        f"<b>Location:</b> {_safe('location', '')} &nbsp;|&nbsp; "
        f"<b>Floors:</b> {_safe('num_floors', 0)} of {_safe('max_floors', 0)}<br/>"
        f"<b>Footprint:</b> {_safe('footprint_area', 0)} m2 &nbsp;|&nbsp; "
        f"<b>Building:</b> {_safe('building_width', 0)} x {_safe('building_length', 0)} m<br/>"
        f"<b>Coverage:</b> {_safe('coverage_ratio', 'n/a')} &nbsp;|&nbsp; "
        f"<b>Rooms:</b> {_safe('num_rooms', 0)}<br/>"
        f"<b>Report UID:</b> <font color='#CC0000'><b>{unique_uid}</b></font>"
    )

    story.append(Paragraph("AUTOCAD LAYOUT REPORT", title_style))
    story.append(Paragraph("Generated Floor Plan", sub_style))
    story.append(Paragraph(meta_html, meta_style))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1.3,
                            color=colors.HexColor("#FF8C00"), spaceAfter=8))

    try:
        if logo_bytes:
            story.append(ReportLabImage(io.BytesIO(logo_bytes), width=70, height=32))
            story.append(Spacer(1, 6))
    except Exception as _e:
        print(f"[pdf] logo embed failed: {_e!r}")

    try:
        qr_buf = generate_qr_code(f"UID: {unique_uid} | Layout - {project_name}")
        build_pdf_footer_signature_and_qr(story, styles, qr_buf, engineer_name)
    except Exception as _e:
        print(f"[pdf] footer failed: {_e!r}")

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def _dark_table(**kwargs):
    return ui.table(**kwargs).classes('w-full text-white').props('dark flat bordered')


# =====================================================================
# ADDITIVE HIGH-TRAFFIC LAYER — PART 2
# =====================================================================
async def process_excel_file_async(file_bytes, filename):
    return await cpu_bound_limited(process_excel_file, file_bytes, filename)


async def generate_progress_pdf_async(pdf_data, engineer_name,
                                      project_name, logo_bytes,
                                      ticket_id):
    return await cpu_bound_limited(
        generate_progress_pdf,
        pdf_data, engineer_name, project_name, logo_bytes, ticket_id,
    )


async def generate_autocad_pdf_async(info, boq_df, engineer_name,
                                     project_name, logo_bytes,
                                     ticket_id):
    return await cpu_bound_limited(
        generate_autocad_pdf,
        info, boq_df, engineer_name, project_name, logo_bytes, ticket_id,
    )


async def extract_areas_from_dxf_async(doc_bytes, unit, workflow):
    return await cpu_bound_limited(
        _extract_areas_from_dxf_bytes, doc_bytes, unit, workflow,
    )


async def build_complete_project_async(params):
    return await cpu_bound_limited(build_complete_project, params)


async def plan_architectural_layout_async(plot_data):
    sem = _get_gemini_sem()
    async with sem:
        return await plan_architectural_layout(plot_data)


async def call_gemini_limited(contents, timeout=None, **kwargs):
    t = timeout or _GEMINI_TIMEOUT_S
    return await gemini_limited(
        lambda: call_gemini(contents, timeout=t, **kwargs),
        timeout=t,
    )


async def call_gemini_json_limited(prompt, timeout=None, **kwargs):
    t = timeout or _GEMINI_TIMEOUT_S
    return await gemini_limited(
        lambda: call_gemini_json(prompt, timeout=t, **kwargs),
        timeout=t,
    )


# =====================================================================
# Main page
# =====================================================================
@ui.page('/')
def main_page():
    ui.query('body').style('width: 100vw; height: 100vh; overflow-x: hidden;')

    # ---------------- SIDEBAR ----------------
    sidebar = ui.left_drawer().classes('sidebar-container').style('width: 380px;')
    with sidebar:
        with ui.row().classes('w-full items-center justify-between mb-4 p-2'):
            ui.label('📋 PROJECT METADATA').classes('text-white font-bold text-base tracking-wide')
            ui.button('✕', on_click=sidebar.toggle).classes(
                'bg-transparent text-white text-xl hover:text-[#FF8C00] p-1 min-w-[36px] '
                '!shadow-none !rounded-full !bg-transparent'
            ).style('font-size: 20px; line-height: 1;')

        project_name_input = ui.input(label='Project Name',
                                       value='Highway Expansion Project').classes('w-full mb-3')
        pour_location_input = ui.input(label='Structural Element / Chainage',
                                        value='Highway Section Ch. 12+500').classes('w-full mb-4')

        ui.label('Governing Design Code Basis').classes('text-white font-bold text-sm mb-1')
        ui.markdown('By default every AI output is generated strictly per **ECP 203 / ECP 202 / ECP 104**.'
                    ).classes('text-xs text-[#A9B6D0] mb-2')
        code_basis_select = ui.select(label='Code Type (applies app-wide)',
                                       options=CODE_BASIS_OPTIONS,
                                       value=CODE_BASIS_OPTIONS[0]).classes('w-full mb-4')

        fcu_input = ui.number(label='Specified 28-Day Grade f_cu (N/mm2)',
                              value=30.0, step=5.0).classes('w-full mb-4')

        ui.label('Batch Plant & Site Logs').classes('text-white font-bold text-sm mb-2')
        truck_input  = ui.input(label='Mixer Truck No.',  value='TRK-104').classes('w-full mb-2')
        ticket_input = ui.input(label='Batch Ticket ID', value='BT-99482').classes('w-full mb-4')

        ui.label('Mix Design Parameters').classes('text-white font-bold text-sm mb-2')
        cement_input = ui.input(label='Cement Content (kg/m3)', value='350.0').classes('w-full mb-2')
        water_input  = ui.input(label='Free Water Content (kg/m3)', value='150.0').classes('w-full mb-4')

        engineer_input = ui.input(label='Engineer Name',
                                  value='Eng. Mohamed Abd Al Aty').classes('w-full mb-2')

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

        ui.upload(label='Upload Company Logo', auto_upload=True,
                  on_upload=handle_logo_upload).props('flat dark').classes('w-full mb-2')

    ui.button('☰', on_click=sidebar.toggle).classes(
        'fixed top-4 left-4 z-50 bg-[#10203f] text-white border border-[#FF8C00] '
        'p-3 rounded-full shadow-lg hover:bg-[#1a2a4a]'
    ).style('font-size: 20px; min-width: 48px; min-height: 48px;')

    def current_meta(uid_prefix):
        return {
            'uid':      f"{uid_prefix}-{uuid.uuid4().hex[:8].upper()}",
            'project':  project_name_input.value,
            'location': pour_location_input.value,
            'engineer': engineer_input.value,
            'date':     datetime.date.today().strftime('%Y-%m-%d'),
            'ticket':   ticket_input.value,
        }

    # ---------------- MAIN ----------------
    with ui.column().classes('w-full min-h-screen p-4 bg-[#031338]'):
        with ui.column().classes('w-full bg-[#0d1a35] px-6 py-3 rounded-xl border '
                                  'border-[#FF8C00] shadow-lg mb-3'):
            ui.label('SMART EGY-CIVIL AI AUDITOR').classes(
                'text-white font-bold leading-tight'
            ).style('font-size: 26px; letter-spacing: 0.02em;')
            ui.label('Intelligent General Civil, Geotechnical & Structural Compliance Engine'
                     ).classes('text-sm font-medium text-[#A9B6D0] mt-0')
            ui.label('Lead Technical Auditor: Eng. Mohamed Abd Al Aty'
                     ).classes('text-xs text-[#A9B6D0] font-semibold mt-0')
            ui.label('Next-generation automated civil engineering and quality intelligence, '
                     'precision-calibrated for the Egyptian Code of Practice.'
                     ).classes('text-xs text-[#A9B6D0] mt-0 italic')

        ui.add_head_html('''<style>@keyframes marquee{0%{transform:translate(0,0);}100%{transform:translate(-100%,0);}}</style>''')
        ui.html('''
        <div style="width:100%;overflow:hidden;white-space:nowrap;background-color:rgba(13,26,53,0.6);backdrop-filter:blur(8px);color:#FFF;padding:10px 0;font-weight:600;font-size:13px;margin-bottom:15px;border-radius:8px;border:1px solid rgba(255,140,0,0.3);">
          <div style="display:inline-block;padding-left:100%;animation:marquee 28s linear infinite;">
            <span style="color:#FF8C00;">[CORE ACTIVE]</span> ECP 203 &middot; ECP 202 &middot; ECP 104 &middot; ASTM &middot; AASHTO &middot; BS EN &middot; ISO
          </div>
        </div>
        ''')

        with ui.tabs().classes('w-full text-white bg-[#060f22] rounded-lg') as tabs:
            t_dash       = ui.tab('Concrete Cube Verifier').classes('text-white font-bold')
            t_audit      = ui.tab('AI Multi-Standard Auditor').classes('text-white font-bold')
            t_defect     = ui.tab('Defect Diagnostic').classes('text-white font-bold')
            t_chat       = ui.tab('AI Chatbot').classes('text-white font-bold')
            t_handwriting= ui.tab('Handwriting OCR').classes('text-white font-bold')
            t_jobs       = ui.tab('Job Board').classes('text-white font-bold')
            t_progress   = ui.tab('Progress Tracker').classes('text-white font-bold')
            t_dxf        = ui.tab('DXF / PDF → BOQ').classes('text-white font-bold')

        with ui.tab_panels(tabs, value=t_dash).classes('w-full bg-transparent mt-4'):

            # ============ TAB 1: CONCRETE CUBE VERIFIER ============
            with ui.tab_panel(t_dash):
                ui.label('Concrete Cube Calculation Sheet & Statistical Verifier'
                         ).classes('text-2xl font-bold text-white mb-4')
                with ui.row().classes('w-full gap-4 mb-4'):
                    with ui.column().classes('input-card flex-1'):
                        ui.label('7-Day Cubes (N/mm2)').classes('font-bold text-white text-sm')
                        c7_input = ui.input(value='21.0, 22.5, 20.5').classes('w-full')
                    with ui.column().classes('input-card flex-1'):
                        ui.label('14-Day Cubes (N/mm2)').classes('font-bold text-white text-sm')
                        c14_input = ui.input(value='26.0, 27.2, 25.8').classes('w-full')
                    with ui.column().classes('input-card flex-1'):
                        ui.label('28-Day Cubes (N/mm2)').classes('font-bold text-white text-sm')
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
                    return {'n': len(arr), 'mean': mean, 'std': std,
                            'min': float(arr.min()), 'max': float(arr.max()),
                            'cov': (std / mean * 100.0) if mean > 0 else 0.0}

                def get_selected_stages(stage_filter):
                    all_stages = [
                        ('7-Day',  c7_input,  parse_vals(c7_input.value)),
                        ('14-Day', c14_input, parse_vals(c14_input.value)),
                        ('28-Day', c28_input, parse_vals(c28_input.value)),
                    ]
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
                        ui.label('Running AI statistical evaluation...').classes('self-center text-sm')
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
                            f"- {label} Values (N/mm2): "
                            f"{', '.join(str(v) for v in values) if values else 'No data'} "
                            f"(n={s['n'] if s else 0})"
                            for label, values, s in stage_stats
                        )
                        prompt = f"""
You are an elite Senior Concrete QA/QC and Structural Engineering Expert.
Perform a statistical evaluation and code-compliance verification.
{get_code_directive(basis)}
{NO_LATEX_RULE}
Filter: {stage_filter}
f_cu target: {target_fcu} N/mm2
{stage_data_text}
Cement = {cement_input.value} kg/m3 | Water = {water_input.value} kg/m3
Truck: {truck_input.value} | Ticket: {ticket_input.value}
Provide: per-stage table, statistical commentary, final PASS/FAIL with ECP clause.
"""
                        res_text = await call_gemini_limited(prompt)
                        ai_cube_result_holder['text'] = res_text
                        result_output_area.clear()
                        with result_output_area:
                            with ui.column().classes('output-card w-full'):
                                ui.label('AI Evaluation & Compliance Verdict'
                                         ).classes('text-xl font-bold text-white mb-2')
                                ui.markdown(res_text).classes('markdown-body')
                        with chart_area:
                            labels = [l for l, _v, _s in stage_stats] + ['Target']
                            means  = [(s['mean'] if s else 0) for _l, _v, s in stage_stats] + [target_fcu]
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=labels, y=means,
                                                     mode='lines+markers+text',
                                                     text=[f"{v:.1f}" for v in means],
                                                     textposition="top center",
                                                     line=dict(color='#4FC3F7', width=3),
                                                     marker=dict(size=10, color='#FF8C00')))
                            fig.add_hline(y=target_fcu, line_dash="dash", line_color="#22C55E")
                            fig.update_layout(title=f'Compressive Strength — {stage_filter}',
                                              template='plotly_dark',
                                              paper_bgcolor='#0d1a35', plot_bgcolor='#0d1a35',
                                              margin=dict(t=40, b=20, l=40, r=20), height=340)
                            ui.plotly(fig).classes('w-full mt-2')
                        with export_buttons_area:
                            def download_pdf_report():
                                try:
                                    meta = current_meta('ECP-AI')
                                    pdf_bytes = build_report_pdf(
                                        "AI CONCRETE CUBE CALCULATION & VERIFICATION REPORT",
                                        f"Standard: {basis} | Filter: {stage_filter}",
                                        ai_cube_result_holder['text'], meta,
                                        logo_bytes_holder['bytes'])
                                    ui.download(pdf_bytes,
                                                filename=f"AI_Concrete_Sheet_{ticket_input.value}.pdf")
                                    ui.notify('PDF downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Error: {str(ex)}', type='negative')
                            ui.button('Download PDF', on_click=download_pdf_report
                                      ).classes('primary-btn flex-1')
                    except Exception as ex:
                        result_output_area.clear()
                        with result_output_area:
                            ui.notify(f'Error: {str(ex)}', type='negative')

                stage_selector = ui.select(
                    label='Stage Display Filter',
                    options=['All Stages', '7-Day Stage', '14-Day Stage', '28-Day Stage'],
                    value='All Stages', on_change=run_verification
                ).classes('w-full md:w-1/3 mb-4')
                stats_area        = ui.column().classes('w-full')
                result_output_area = ui.column().classes('w-full')
                chart_area        = ui.column().classes('w-full')
                export_buttons_area = ui.row().classes('w-full gap-4 mt-4')
                ui.button('Run AI Verification', on_click=run_verification
                          ).classes('primary-btn q-my-md')
                with result_output_area:
                    ui.markdown('*Click "Run AI Verification" to generate the report.*'
                                ).classes('text-sm text-[#A9B6D0]')

            # ============ TAB 2: AI MULTI-STANDARD AUDITOR ============
            with ui.tab_panel(t_audit):
                ui.label('AI Multi-Standard Engineering Auditor'
                         ).classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload a specification or report to audit.'
                            ).classes('markdown-body mb-2')
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
                audit_status_label = ui.label('Status: No file uploaded yet'
                                              ).classes('text-xs text-amber-400 font-semibold mb-2')
                uploaded_file_data = {'bytes': None, 'name': None, 'type': None}

                async def handle_audit_upload(e):
                    try:
                        data = await e.file.read()
                        uploaded_file_data['bytes'] = data
                        uploaded_file_data['name']  = e.file.name
                        uploaded_file_data['type']  = detect_mime_type(e.file.name, data)
                        audit_status_label.set_text(f'File Ready: {e.file.name}')
                        audit_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                        ui.notify(f'Loaded: {e.file.name}', type='positive')
                    except Exception as ex:
                        ui.notify(f'Error: {str(ex)}', type='negative')

                ui.upload(label='Select PDF or Image File', auto_upload=True,
                          on_upload=handle_audit_upload).props('flat dark').classes('w-full mb-4')
                audit_output_container = ui.column().classes('w-full')

                async def run_ai_audit():
                    if not client or not uploaded_file_data['bytes']:
                        ui.notify('API key missing or no file uploaded!', type='negative')
                        return
                    audit_output_container.clear()
                    with audit_output_container:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Executing audit...').classes('self-center text-sm')
                    try:
                        basis = code_basis_select.value
                        prompt = f"""You are a Principal Civil/Geotechnical/Highway Auditor.
Audit Focus: {audit_focus}
{get_code_directive(basis)}
{NO_LATEX_RULE}
Perform a comprehensive technical audit."""
                        contents = [prompt]
                        if uploaded_file_data['type'] == 'application/pdf':
                            reader = pypdf.PdfReader(io.BytesIO(uploaded_file_data['bytes']))
                            text = "".join([p.extract_text() or "" for p in reader.pages[:10]])
                            contents.append(f"PDF text:\n{text[:10000]}")
                        else:
                            contents.append(types.Part.from_bytes(
                                data=uploaded_file_data['bytes'],
                                mime_type=uploaded_file_data['type']))
                        audit_result_text = await call_gemini_limited(contents, timeout=240)
                        audit_output_container.clear()
                        with audit_output_container:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Audit Findings'
                                         ).classes('text-xl font-bold text-white mb-2')
                                ui.markdown(audit_result_text).classes('markdown-body')
                    except Exception as ex:
                        audit_output_container.clear()
                        with audit_output_container:
                            ui.notify(f'Error: {str(ex)}', type='negative')

                ui.button('Execute AI Audit', on_click=run_ai_audit).classes('primary-btn')

            # ============ TAB 3: DEFECT DIAGNOSTIC ============
            with ui.tab_panel(t_defect):
                ui.label('AI Defect Diagnostic & Repair Protocol'
                         ).classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload site defect photos for forensic analysis.'
                            ).classes('markdown-body mb-2')
                defect_status_label = ui.label('Status: No file uploaded yet'
                                               ).classes('text-xs text-amber-400 font-semibold mb-2')
                defect_file_data = {'bytes': None, 'type': None}
                defect_user_message = ui.input(
                    label='Describe the defect (optional)',
                    placeholder='e.g., "Cracks near column base with spalling concrete"'
                ).classes('w-full mb-3')

                async def handle_defect_upload(e):
                    try:
                        data = await e.file.read()
                        defect_file_data['bytes'] = data
                        defect_file_data['type']  = detect_mime_type(e.file.name, data)
                        defect_status_label.set_text(f'File Ready: {e.file.name}')
                        defect_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    except Exception as ex:
                        ui.notify(f'Error: {str(ex)}', type='negative')

                ui.upload(label='Site Defect Photo or PDF', auto_upload=True,
                          on_upload=handle_defect_upload).props('flat dark').classes('w-full mb-4')
                defect_output = ui.column().classes('w-full')

                async def run_defect_diagnosis():
                    if not client or not defect_file_data['bytes']:
                        ui.notify('API key or file missing!', type='negative')
                        return
                    defect_output.clear()
                    with defect_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Analyzing defect...').classes('self-center text-sm')
                    try:
                        basis = code_basis_select.value
                        desc = defect_user_message.value.strip() or "No description provided."
                        contents = [f"""You are a Senior Forensic Structural Engineer.
Defect description: "{desc}"
{get_code_directive(basis)}
{NO_LATEX_RULE}
Provide defect type, root cause analysis, repair protocol, product table (Egypt market), and cost summary."""]
                        if defect_file_data['type'] == 'application/pdf':
                            reader = pypdf.PdfReader(io.BytesIO(defect_file_data['bytes']))
                            text = "".join([p.extract_text() or "" for p in reader.pages[:10]])
                            contents.append(f"PDF text:\n{text[:10000]}")
                        else:
                            contents.append(types.Part.from_bytes(
                                data=defect_file_data['bytes'],
                                mime_type=defect_file_data['type']))
                        res_text = await call_gemini_limited(contents, timeout=240)
                        defect_output.clear()
                        with defect_output:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Forensic Diagnosis'
                                         ).classes('text-xl font-bold text-white mb-2')
                                ui.markdown(res_text).classes('markdown-body')
                    except Exception as ex:
                        defect_output.clear()
                        with defect_output:
                            ui.notify(f'Error: {ex}', type='negative')

                ui.button('Diagnose Defect', on_click=run_defect_diagnosis).classes('primary-btn')

            # ============ TAB 4: AI CHATBOT ============
            with ui.tab_panel(t_chat):
                ui.label('Core-Code Intelligent Chatbot'
                         ).classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Ask any engineering question.'
                            ).classes('markdown-body mb-2')
                chat_container = ui.column().classes(
                    'output-card w-full h-[500px] overflow-y-auto mb-4')
                chat_messages = [{"role": "assistant",
                                  "content": "Hello! How can I assist you today?"}]

                def render_chat():
                    chat_container.clear()
                    with chat_container:
                        for msg in chat_messages:
                            is_ai = msg['role'] == 'assistant'
                            with ui.column().classes('chat-message'):
                                cls = 'assistant' if is_ai else 'user'
                                ui.label('Assistant' if is_ai else 'You'
                                         ).classes(f'role-label {cls}')
                                ui.markdown(msg['content']).classes('content markdown-body')

                render_chat()
                user_msg = ui.input(placeholder='Type your question...').classes('w-full mb-2')

                async def send_chat():
                    q = user_msg.value
                    if not q or not q.strip():
                        return
                    chat_messages.append({"role": "user", "content": q})
                    user_msg.value = ''
                    render_chat()
                    if not client:
                        chat_messages.append({"role": "assistant",
                                              "content": "GEMINI_API_KEY not configured."})
                        render_chat()
                        return
                    try:
                        basis = code_basis_select.value
                        system_prompt = (
                            "You are an elite Senior Civil/Geotechnical/Structural Expert.\n"
                            f"{get_code_directive(basis)}\n{NO_LATEX_RULE}\n"
                            "Use strictly METRIC (SI) units."
                        )
                        answer = await call_gemini_limited(
                            q, system_instruction=system_prompt, timeout=240)
                        chat_messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        chat_messages.append({"role": "assistant", "content": f"Error: {e}"})
                    render_chat()

                user_msg.on('keydown.enter', lambda: send_chat())
                with ui.row().classes('w-full gap-4 mt-2'):
                    ui.button('Send', on_click=send_chat).classes('primary-btn flex-1')

                    def download_chat_pdf():
                        try:
                            meta = current_meta('CHAT')
                            styles = build_pdf_styles()
                            flowables = []
                            for m in chat_messages:
                                role = "ASSISTANT" if m['role'] == 'assistant' else "USER"
                                flowables.append(Paragraph(role, styles['h3']))
                                flowables.extend(markdown_to_pdf_flowables(m['content'], styles))
                                flowables.append(Spacer(1, 4))
                            pdf_bytes = build_report_pdf(
                                "AI CHAT TRANSCRIPT", "Q&A Record",
                                "", meta, logo_bytes_holder['bytes'],
                                extra_flowables_before_body=flowables)
                            ui.download(pdf_bytes,
                                        filename=f"AI_Chat_{ticket_input.value}.pdf")
                            ui.notify('Chat PDF downloaded!', type='positive')
                        except Exception as ex:
                            ui.notify(f'PDF Error: {str(ex)}', type='negative')

                    ui.button('Download Transcript PDF', on_click=download_chat_pdf
                              ).classes('primary-btn flex-1')

            # ============ TAB 5: HANDWRITING OCR ============
            with ui.tab_panel(t_handwriting):
                ui.label('Handwriting OCR').classes('text-2xl font-bold text-white mb-2')
                ui.markdown('Upload a handwritten note (PNG / JPG / PDF). '
                            'After transcription you can download the result as PDF or TXT.'
                            ).classes('markdown-body mb-2')
                ocr_file_data = {'bytes': None, 'type': None, 'name': None}
                ocr_status_label = ui.label('Status: No file uploaded yet'
                                            ).classes('text-xs text-amber-400 font-semibold mb-2')

                async def handle_ocr_upload(e):
                    try:
                        data = await e.file.read()
                        ocr_file_data['bytes'] = data
                        ocr_file_data['type']  = detect_mime_type(e.file.name, data)
                        ocr_file_data['name']  = e.file.name
                        ocr_status_label.set_text(
                            f'Ready: {e.file.name} ({len(data)/1024/1024:.1f} MB)')
                        ocr_status_label.classes(replace='text-xs text-emerald-400 font-semibold mb-2')
                    except Exception as ex:
                        ui.notify(f'Error: {str(ex)}', type='negative')

                ui.upload(label='Upload Handwriting', auto_upload=True,
                          on_upload=handle_ocr_upload).props('flat dark').classes('w-full mb-4')
                ocr_output = ui.column().classes('w-full')
                ocr_export = ui.row().classes('w-full gap-4 mt-4')
                text_editor = {'widget': None}
                transcribed_holder = {'text': ''}

                async def run_ocr():
                    if not client or not ocr_file_data['bytes']:
                        ui.notify('API key or file missing!', type='negative')
                        return
                    ocr_output.clear()
                    ocr_export.clear()
                    with ocr_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Transcribing...').classes('self-center text-sm')
                    try:
                        prompt = ("Transcribe the handwritten text. "
                                  "If tabular, use Markdown tables. Return only text.")
                        contents = [prompt]
                        if ocr_file_data['type'] == 'application/pdf':
                            doc = fitz.open(stream=ocr_file_data['bytes'], filetype="pdf")
                            for pnum in range(min(6, len(doc))):
                                pix = doc.load_page(pnum).get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                                contents.append(types.Part.from_bytes(
                                    data=pix.tobytes("png"), mime_type="image/png"))
                            doc.close()
                        else:
                            contents.append(types.Part.from_bytes(
                                data=ocr_file_data['bytes'], mime_type=ocr_file_data['type']))
                        response_text = await call_gemini_limited(
                            contents, temperature=0, timeout=240)
                        transcribed = sanitize_ai_markdown(response_text)
                        transcribed_holder['text'] = transcribed
                        ocr_output.clear()
                        with ocr_output:
                            with ui.column().classes('output-card w-full'):
                                ui.label('Transcribed Text (editable)'
                                         ).classes('text-xl font-bold text-white mb-2')
                                text_editor['widget'] = ui.textarea(value=transcribed).classes(
                                    'w-full markdown-body').style('min-height: 300px;')
                                ui.label('Preview:').classes('text-lg font-bold text-white mt-2')
                                preview = ui.column().classes('w-full')

                                def update_preview():
                                    preview.clear()
                                    with preview:
                                        ui.markdown(text_editor['widget'].value
                                                    ).classes('markdown-body')
                                text_editor['widget'].on('input', update_preview)
                                update_preview()

                        ocr_export.clear()
                        with ocr_export:
                            def download_ocr_pdf():
                                try:
                                    current_text = (text_editor['widget'].value
                                                    if text_editor['widget'] else
                                                    transcribed_holder['text'])
                                    meta = current_meta('OCR')
                                    pdf_bytes = build_report_pdf(
                                        doc_title="Handwriting Transcription",
                                        subtitle=f"Source: {ocr_file_data.get('name', '')}",
                                        body_markdown=current_text,
                                        meta=meta,
                                        logo_bytes=logo_bytes_holder['bytes'],
                                        show_ticket=False,
                                    )
                                    ui.download(
                                        pdf_bytes,
                                        filename=f"Handwriting_Transcription_{ticket_input.value}.pdf")
                                    ui.notify('PDF downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'PDF Error: {str(ex)}', type='negative')

                            def download_ocr_txt():
                                try:
                                    current_text = (text_editor['widget'].value
                                                    if text_editor['widget'] else
                                                    transcribed_holder['text'])
                                    ui.download(
                                        current_text.encode('utf-8'),
                                        filename=f"Handwriting_Transcription_{ticket_input.value}.txt")
                                    ui.notify('TXT downloaded!', type='positive')
                                except Exception as ex:
                                    ui.notify(f'TXT Error: {str(ex)}', type='negative')

                            ui.button('📄 Download PDF Report', on_click=download_ocr_pdf
                                      ).classes('primary-btn flex-1')
                            ui.button('📝 Download TXT', on_click=download_ocr_txt
                                      ).classes('primary-btn flex-1')

                    except Exception as ex:
                        ocr_output.clear()
                        with ocr_output:
                            ui.notify(f'Failed: {str(ex)}', type='negative')

                ui.button('Transcribe Handwriting', on_click=run_ocr).classes('primary-btn')

            # ============ TAB 6: JOB BOARD ============
            with ui.tab_panel(t_jobs):
                ui.label('Engineering Job Board - Egypt'
                         ).classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Search the latest engineering jobs in Egypt.'
                            ).classes('markdown-body mb-2')
                ui.label('🔑 RapidAPI: ' +
                         ('✅ Set' if RAPIDAPI_KEY else '❌ Not set – using Wuzzuf/Bayt fallback.')
                         ).classes('text-sm text-[#A9B6D0] mb-2')
                debug_output = ui.label('Debug: waiting...').classes('text-xs text-[#A9B6D0] mb-2')
                with ui.row().classes('w-full gap-4 mb-4'):
                    search_input = ui.input(label='Search', placeholder='Civil Engineer',
                                            value='Civil Engineer').classes('flex-1')
                    location_input = ui.input(label='Location', placeholder='Cairo').classes('flex-1')
                    ui.button('Search Jobs', on_click=lambda: search_jobs()).classes('primary-btn')
                filter_input = ui.input(label='Filter', placeholder='Type to filter...',
                                        on_change=lambda: filter_jobs()).classes('w-full mb-2')
                results_container = ui.column().classes('w-full')
                jobs_data = []

                def display_jobs(jobs, filter_text=''):
                    results_container.clear()
                    with results_container:
                        if not jobs:
                            ui.label('No jobs found.').classes('text-white')
                            return
                        filtered = jobs
                        if filter_text:
                            fl = filter_text.lower()
                            filtered = [j for j in jobs
                                        if fl in j['title'].lower()
                                        or fl in j['company'].lower()
                                        or fl in j['description'].lower()]
                        for job in filtered:
                            with ui.card().classes('w-full bg-[#0d1a35] border '
                                                    'border-[#2c3f6b] rounded-lg p-3 mb-2'):
                                with ui.row().classes('w-full justify-between'):
                                    ui.label(job['title']).classes('text-lg font-bold text-white')
                                    ui.label(job['company']).classes('text-sm text-[#A9B6D0]')
                                ui.label(job['location']).classes('text-sm text-[#A9B6D0]')
                                desc = job['description'][:200] + \
                                       ('...' if len(job['description']) > 200 else '')
                                ui.label(desc).classes('text-sm text-white mt-1')
                                ui.link('View Job', job['url'], new_tab=True
                                        ).classes('text-[#4FC3F7] hover:text-[#FF8C00]')

                def filter_jobs():
                    display_jobs(jobs_data, filter_input.value.strip())

                async def search_jobs():
                    q = search_input.value.strip()
                    if not q:
                        ui.notify('Enter a search term.', type='warning')
                        return
                    loc = location_input.value.strip()
                    if loc:
                        q += f' {loc}'
                    results_container.clear()
                    with results_container:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Searching...').classes('self-center text-sm')
                    jobs = await run.io_bound(scrape_jobs, q)
                    jobs_data.clear()
                    jobs_data.extend(jobs)
                    debug_output.set_text(f'Found {len(jobs)} jobs')
                    display_jobs(jobs_data)

            # ============ TAB 7: PROGRESS TRACKER ============
            with ui.tab_panel(t_progress):
                ui.label('📊 Project Progress Tracker'
                         ).classes('text-2xl font-bold text-white mb-4')
                ui.markdown('Upload site photos, PDFs, or Excel to track progress.'
                            ).classes('markdown-body mb-2')

                uploaded_files = []
                upload_status = ui.label('No files uploaded yet.'
                                         ).classes('text-xs text-amber-400 mb-2')

                async def handle_progress_upload(e):
                    try:
                        data = await e.file.read()
                        fname = e.file.name
                        ftype = detect_mime_type(fname, data)
                        uploaded_files.append({'bytes': data, 'name': fname, 'type': ftype})
                        upload_status.set_text(f'{len(uploaded_files)} file(s) uploaded.')
                        upload_status.classes(replace='text-xs text-emerald-400 mb-2')
                    except Exception as ex:
                        ui.notify(f'Upload error: {str(ex)}', type='negative')

                ui.upload(auto_upload=True, on_upload=handle_progress_upload,
                          multiple=True).props('flat dark').classes('w-full mb-4')

                with ui.row().classes('w-full gap-4 mb-4 items-center'):
                    ui.label('Start Date').classes('text-white text-sm font-semibold')
                    _start_default = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                    with ui.input(value=_start_default) as start_date:
                        with ui.menu().props('no-parent-event') as _start_menu:
                            ui.date().bind_value(start_date)
                        with start_date.add_slot('append'):
                            ui.icon('edit_calendar').on('click', _start_menu.open).classes('cursor-pointer')
                    ui.label('End Date').classes('text-white text-sm font-semibold')
                    _end_default = datetime.date.today().strftime('%Y-%m-%d')
                    with ui.input(value=_end_default) as end_date:
                        with ui.menu().props('no-parent-event') as _end_menu:
                            ui.date().bind_value(end_date)
                        with end_date.add_slot('append'):
                            ui.icon('edit_calendar').on('click', _end_menu.open).classes('cursor-pointer')

                description_input = ui.input(label='Project Phase',
                                              value='Foundation and Structure').classes('w-full mb-4')

                progress_output    = ui.column().classes('w-full')
                progress_overview  = ui.column().classes('w-full')

                async def run_progress_analysis():
                    if not client or not uploaded_files:
                        ui.notify('API key missing or no files uploaded!', type='negative')
                        return
                    progress_output.clear()
                    progress_overview.clear()
                    with progress_output:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Processing...').classes('self-center text-sm')
                    try:
                        all_rows = []
                        for f in uploaded_files:
                            ext = os.path.splitext(f['name'])[1].lower()
                            if ext in ['.xlsx', '.xls']:
                                df = await process_excel_file_async(f['bytes'], f['name'])
                                if not df.empty:
                                    df['source'] = f['name']
                                    all_rows.append(df)
                            elif f['type'] in ['image/png', 'image/jpeg', 'application/pdf']:
                                data = await extract_progress_from_image(f['bytes'], f['type'])
                                if data:
                                    row = {**data, 'source': f['name']}
                                    all_rows.append(pd.DataFrame([row]))
                        if not all_rows:
                            ui.notify('No data extracted.', type='warning')
                            return
                        combined_df = pd.concat(all_rows, ignore_index=True)
                        if 'date' in combined_df.columns:
                            combined_df['date'] = pd.to_datetime(combined_df['date'],
                                                                  errors='coerce')
                        display_df = combined_df.copy()
                        if 'date' in display_df.columns:
                            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                        display_df = display_df.fillna('N/A')

                        progress_output.clear()
                        with progress_output:
                            ui.label('📋 Progress Summary'
                                     ).classes('text-xl font-bold text-white mb-2')
                            columns = [{'name': c, 'label': c.replace('_', ' ').title(),
                                        'field': c, 'sortable': True}
                                       for c in display_df.columns]
                            _dark_table(columns=columns,
                                        rows=display_df.to_dict('records'),
                                        row_key='index')

                        overview_text = await generate_progress_overview(
                            combined_df,
                            start_date.value or 'N/A',
                            end_date.value or 'N/A',
                            description_input.value)
                        progress_overview.clear()
                        with progress_overview:
                            ui.label('📝 AI Overview'
                                     ).classes('text-xl font-bold text-white mb-2')
                            ui.markdown(overview_text).classes('markdown-body')
                    except Exception as e:
                        progress_output.clear()
                        with progress_output:
                            ui.notify(f'Failed: {str(e)}', type='negative')

                ui.button('Run AI Analysis', on_click=run_progress_analysis
                          ).classes('primary-btn mt-4')

            # ============ TAB 8: DXF / PDF → BOQ ============
            with ui.tab_panel(t_dxf):
                ui.label('📐 DXF / PDF → BOQ').classes('text-2xl font-bold text-white mb-2')
                ui.markdown(
                    'Upload an architectural floor plan. **Quick mode** extracts areas only. '
                    '**Full BOQ mode** runs the complete engine: walls, rooms, wet/dry split, '
                    'finishes, and quantities.'
                ).classes('markdown-body mb-3')

                # ---------------- STATE ----------------
                boq_state = {
                    'bytes': None, 'name': None, 'type': None,
                    'is_dxf': False, 'is_pdf': False, 'is_image': False,
                    'records': None,
                    'walls': None,
                    'rooms': None,
                    'boq': None,
                    'info': None,
                }

                # ---------------- UPLOAD ----------------
                boq_status = ui.label('Status: No file uploaded yet'
                                      ).classes('text-xs text-amber-400 font-semibold mb-2')

                async def handle_boq_upload(e):
                    try:
                        data = await e.file.read()
                        if isinstance(data, str):
                            data = data.encode('utf-8')
                        name = e.file.name
                        ext = os.path.splitext(name)[1].lower()
                        ftype = detect_mime_type(name, data)

                        boq_state['bytes'] = data
                        boq_state['name'] = name
                        boq_state['type'] = ftype
                        boq_state['is_dxf'] = ext == '.dxf'
                        boq_state['is_pdf'] = ftype == 'application/pdf'
                        boq_state['is_image'] = (
                            ftype in ('image/png', 'image/jpeg')
                            and not boq_state['is_dxf']
                        )

                        kind = ('DXF' if boq_state['is_dxf']
                                else 'PDF' if boq_state['is_pdf']
                                else 'Image')
                        boq_status.set_text(f'Ready: {name} ({kind}, {len(data)/1024:.0f} KB)')
                        boq_status.classes(replace='text-xs text-emerald-400 font-semibold mb-2')

                        _update_boq_visibility()

                        # If DXF, probe layers and build the mapping UI
                        if boq_state['is_dxf']:
                            try:
                                rows = await asyncio.to_thread(
                                    _list_layers_dxf, boq_state['bytes'])
                                _render_layer_mapping(rows)
                            except Exception as le:
                                print(f"[boq] layer probe failed: {le!r}")
                                _render_layer_mapping([])
                        else:
                            _render_layer_mapping([])
                    except Exception as ex:
                        ui.notify(f'Upload error: {ex}', type='negative')

                ui.upload(auto_upload=True, on_upload=handle_boq_upload,
                          multiple=False).props('flat dark').classes('w-full mb-3')

                # ---------------- LAYER MAPPING PANEL ----------------
                # Only shown when a DXF is uploaded. Lets the user manually
                # assign a category to any layer the taxonomy couldn't classify.
                layer_map_rows = []   # list of {layer, count, dropdown}
                layer_overrides_holder = {'manual': {}}

                with ui.column().classes('w-full') as boq_layers_panel:
                    ui.label('Layer Mapping').classes('text-sm font-bold text-white mb-1')
                    ui.markdown(
                        'Review how each layer in your DXF was interpreted. '
                        'Fix any wrong or missing classification below, then click Run.'
                    ).classes('text-xs text-[#A9B6D0] mb-2')
                    boq_layers_body = ui.column().classes('w-full')

                LAYER_CATEGORY_OPTIONS = {
                    '': '(ignore this layer)',
                    'wall': 'Wall',
                    'wall_ext': 'Wall — external',
                    'wall_int': 'Wall — internal',
                    'column': 'Column',
                    'beam': 'Beam',
                    'slab': 'Slab',
                    'footing': 'Footing',
                    'door': 'Door',
                    'window': 'Window',
                    'stair': 'Stair',
                    'balcony': 'Balcony',
                    'void': 'Void / shaft',
                    'room': 'Room (floor outline)',
                    'furniture': 'Furniture',
                    'hatch': 'Hatch',
                    'dim': 'Dimension',
                    'text': 'Text',
                    'grid': 'Grid / axis',
                    'section': 'Section mark',
                    'level': 'Level tag',
                    'boundary': 'Plot boundary',
                    'title_block': 'Title block',
                }

                def _render_layer_mapping(layer_rows):
                    boq_layers_body.clear()
                    layer_map_rows.clear()
                    layer_overrides_holder['manual'] = {}
                    if not layer_rows:
                        with boq_layers_body:
                            ui.label('No layers found.').classes('text-xs text-amber-300')
                        return

                    with boq_layers_body:
                        for r in layer_rows:
                            layer = r['layer']
                            count = r['count']
                            cat = r.get('category') or ''
                            # Normalise wall_ext/wall_int into dropdown keys
                            default = cat
                            if cat == 'wall' and r.get('subtype') == 'external':
                                default = 'wall_ext'
                            elif cat == 'wall' and r.get('subtype') == 'internal':
                                default = 'wall_int'

                            with ui.row().classes('w-full items-center gap-2 mb-1'):
                                ui.label(f'{layer}').classes(
                                    'text-white text-xs font-mono flex-1 truncate'
                                ).style('min-width: 180px;')
                                ui.label(f'×{count}').classes(
                                    'text-[#A9B6D0] text-xs w-14 text-right'
                                )
                                dd = ui.select(
                                    options=LAYER_CATEGORY_OPTIONS,
                                    value=default,
                                    with_input=False,
                                ).classes('text-xs').style('width: 220px;')

                                def _on_change_factory(layer_name):
                                    def _h(ev):
                                        v = ev.value or ''
                                        if v == '':
                                            layer_overrides_holder['manual'].pop(
                                                layer_name, None)
                                        else:
                                            # Split wall_ext / wall_int into
                                            # category + subtype
                                            if v == 'wall_ext':
                                                layer_overrides_holder['manual'][layer_name] = {
                                                    'category': 'wall',
                                                    'subtype':  'external',
                                                    'confidence': 'manual',
                                                }
                                            elif v == 'wall_int':
                                                layer_overrides_holder['manual'][layer_name] = {
                                                    'category': 'wall',
                                                    'subtype':  'internal',
                                                    'confidence': 'manual',
                                                }
                                            else:
                                                layer_overrides_holder['manual'][layer_name] = {
                                                    'category': v,
                                                    'subtype':  None,
                                                    'confidence': 'manual',
                                                }
                                    return _h

                                dd.on('update:model-value', _on_change_factory(layer))
                                layer_map_rows.append({
                                    'layer': layer, 'count': count, 'dropdown': dd,
                                })

                # ---------------- MODE ----------------
                boq_mode = ui.radio(
                    {'quick': 'Quick areas', 'full': 'Full BOQ'},
                    value='quick',
                ).props('inline').classes('mb-2 text-white')
                boq_mode.on('update:model-value', lambda _: _update_boq_visibility())

                # ---------------- PARAMETERS PANEL ----------------
                with ui.column().classes('w-full') as boq_params_panel:
                    ui.label('BOQ Parameters').classes('text-sm font-bold text-white mb-2')
                    with ui.row().classes('w-full gap-3 flex-wrap'):
                        with ui.column().classes('input-card'):
                            boq_units = ui.radio(['mm', 'cm', 'm'], value='mm').props('inline')
                            ui.label('DXF drawing units').classes('text-xs text-[#A9B6D0]')
                        with ui.column().classes('input-card'):
                            boq_ext_t = ui.number(label='Ext. wall thickness (mm)',
                                                   value=250, min=80, max=600, step=10).classes('w-40')
                        with ui.column().classes('input-card'):
                            boq_int_t = ui.number(label='Int. wall thickness (mm)',
                                                   value=120, min=60, max=400, step=10).classes('w-40')
                        with ui.column().classes('input-card'):
                            boq_floor_h = ui.number(label='Floor-to-floor height (m)',
                                                     value=3.00, min=2.4, max=6.0, step=0.05).classes('w-40')
                        with ui.column().classes('input-card'):
                            boq_door_h = ui.number(label='Door height (m)',
                                                    value=2.10, min=1.8, max=3.0, step=0.05).classes('w-40')
                        with ui.column().classes('input-card'):
                            boq_win_h = ui.number(label='Window height (m)',
                                                   value=1.20, min=0.6, max=2.4, step=0.05).classes('w-40')
                        with ui.column().classes('input-card'):
                            boq_wet_tile = ui.number(label='Wet-area tile height (m)',
                                                      value=2.10, min=1.0, max=4.0, step=0.05).classes('w-40')

                # ---------------- SCALE PANEL (PDF/PNG only) ----------------
                with ui.column().classes('w-full') as boq_scale_panel:
                    ui.label('Scale Reference (PDF / Image only)').classes('text-sm font-bold text-white mb-2')
                    ui.markdown(
                        'If blank, all AI measurements are approximate. '
                        'For a real scale, measure a known distance in pixels using any '
                        'image viewer and enter the two points below.'
                    ).classes('text-xs text-[#A9B6D0] mb-1')
                    with ui.row().classes('w-full gap-3 flex-wrap'):
                        with ui.column().classes('input-card'):
                            boq_scale_mm_per_px = ui.number(
                                label='mm per pixel',
                                value=None, step=0.1,
                            ).classes('w-40')
                        with ui.column().classes('input-card'):
                            boq_ref_px_x1 = ui.number(label='Point 1 X (px)', value=None).classes('w-32')
                        with ui.column().classes('input-card'):
                            boq_ref_px_y1 = ui.number(label='Point 1 Y (px)', value=None).classes('w-32')
                        with ui.column().classes('input-card'):
                            boq_ref_px_x2 = ui.number(label='Point 2 X (px)', value=None).classes('w-32')
                        with ui.column().classes('input-card'):
                            boq_ref_px_y2 = ui.number(label='Point 2 Y (px)', value=None).classes('w-32')
                        with ui.column().classes('input-card'):
                            boq_ref_mm = ui.number(label='Real distance (mm)',
                                                    value=None, step=100).classes('w-40')

                def _update_boq_visibility():
                    is_full = boq_mode.value == 'full'
                    is_dxf = boq_state.get('is_dxf', False)
                    is_img = boq_state.get('is_pdf', False) or boq_state.get('is_image', False)
                    boq_params_panel.set_visibility(is_full)
                    boq_scale_panel.set_visibility(is_full and is_img)

                _update_boq_visibility()

                # ---------------- RESULTS / EXPORT ----------------
                boq_results = ui.column().classes('w-full mt-3')
                boq_exports = ui.row().classes('w-full gap-4 mt-3')

                # ---------------- RUN ----------------
                async def run_boq():
                    boq_results.clear()
                    boq_exports.clear()
                    if not boq_state['bytes']:
                        ui.notify('Upload a file first.', type='warning')
                        return

                    mode = boq_mode.value
                    is_dxf = boq_state['is_dxf']
                    is_pdf = boq_state['is_pdf']
                    is_img = boq_state['is_image']

                    with boq_results:
                        ui.spinner('ios', size='lg').classes('self-center text-[#4FC3F7]')
                        ui.label('Processing…').classes('self-center text-sm text-white')

                    # ==================================================
                    # QUICK MODE — DXF only, areas table
                    # ==================================================
                    if mode == 'quick':
                        if is_dxf:
                            try:
                                areas = await extract_areas_from_dxf_async(
                                    boq_state['bytes'],
                                    boq_units.value,
                                    'architectural',
                                )
                                df = pd.DataFrame(areas).sort_values(
                                    'area_m2', ascending=False)
                                total = df['area_m2'].sum()
                                boq_results.clear()
                                with boq_results:
                                    ui.label('Extracted Areas'
                                             ).classes('text-xl font-bold text-white mb-2')
                                    cols = [
                                        {'name': 'layer', 'label': 'Layer',
                                         'field': 'layer', 'sortable': True},
                                        {'name': 'label', 'label': 'Label',
                                         'field': 'label', 'sortable': True},
                                        {'name': 'area_m2', 'label': 'Area (m²)',
                                         'field': 'area_m2', 'sortable': True},
                                    ]
                                    _dark_table(columns=cols,
                                                rows=df.to_dict('records'),
                                                row_key='index')
                                    ui.label(f'Total: {total:.4f} m²'
                                             ).classes('text-lg font-bold text-[#FF8C00] mt-2')
                            except Exception as e:
                                boq_results.clear()
                                with boq_results:
                                    ui.notify(f'DXF error: {e}', type='negative')
                        else:
                            boq_results.clear()
                            with boq_results:
                                ui.notify(
                                    'Quick mode is only for DXF. '
                                    'Use Full BOQ mode for PDF / images.',
                                    type='warning',
                                )
                        return

                    # ==================================================
                    # FULL BOQ MODE
                    # ==================================================
                    try:
                        ext_t_mm   = float(boq_ext_t.value or 250)
                        int_t_mm   = float(boq_int_t.value or 120)
                        floor_h_m  = float(boq_floor_h.value or 3.0)
                        door_h_m   = float(boq_door_h.value or 2.1)
                        win_h_m    = float(boq_win_h.value or 1.2)
                        wet_tile_m = float(boq_wet_tile.value or 2.1)

                        if is_dxf:
                            units = boq_units.value or 'mm'

                            # --- Pass 1: AI classifies layers the taxonomy missed ---
                            layer_overrides = {}
                            try:
                                unknown = await asyncio.to_thread(
                                    _probe_unknown_layers_dxf,
                                    boq_state['bytes'],
                                )
                                if unknown:
                                    print(f"[boq] unknown layers: {unknown}")
                                    with boq_results:
                                        ui.label(
                                            f'🔎 Asking AI to classify '
                                            f'{len(unknown)} unknown layer(s)…'
                                        ).classes('self-center text-xs text-[#A9B6D0]')
                                    from services.boq import classify_unknown_layers as _cu
                                    ai_overrides = await gemini_limited(
                                        lambda: _cu(unknown, call_gemini_json_limited),
                                        timeout=60,
                                    )
                                    _ok = sum(1 for v in (ai_overrides or {}).values()
                                              if v.get('category'))
                                    print(f"[boq] AI classified {_ok}/{len(unknown)} layers")
                                    if ai_overrides:
                                        layer_overrides.update(ai_overrides)
                            except Exception as _ce:
                                print(f"[boq] AI layer classifier failed "
                                      f"(continuing): {_ce!r}")

                            # --- Manual overrides always win ---
                            manual = layer_overrides_holder.get('manual') or {}
                            if manual:
                                print(f"[boq] applying {len(manual)} manual "
                                      f"layer override(s)")
                                layer_overrides.update(manual)

                            # --- Pass 2: full extraction ---
                            payload = await asyncio.to_thread(
                                _boq_run_dxf,
                                boq_state['bytes'], units,
                                ext_t_mm, int_t_mm,
                                floor_h_m, door_h_m, win_h_m, wet_tile_m,
                                layer_overrides or None,
                            )
                        else:
                            # ---------- PDF / PNG branch ----------
                            if is_pdf:
                                png = _pdf_first_page_to_png_bytes(boq_state['bytes'])
                                if not png:
                                    raise Exception('Could not read PDF page 1')
                                img_bytes = png
                                img_mime = 'image/png'
                            else:
                                img_bytes = boq_state['bytes']
                                img_mime = boq_state['type']

                            scale_info = None
                            mm_per_px = boq_scale_mm_per_px.value
                            if mm_per_px:
                                scale_info = {'mode': 'pixel_ratio',
                                              'mm_per_pixel': float(mm_per_px)}
                            elif (boq_ref_px_x1.value is not None
                                  and boq_ref_px_x2.value is not None
                                  and boq_ref_mm.value):
                                scale_info = {
                                    'mode': 'two_points',
                                    'p1': [float(boq_ref_px_x1.value),
                                           float(boq_ref_px_y1.value or 0)],
                                    'p2': [float(boq_ref_px_x2.value),
                                           float(boq_ref_px_y2.value or 0)],
                                    'real_distance_mm': float(boq_ref_mm.value),
                                }

                            records, _img_stats = await gemini_limited(
                                lambda: boq_extract_from_image(
                                    img_bytes, img_mime,
                                    call_gemini_json_fn=call_gemini_json_limited,
                                    scale_info=scale_info,
                                ),
                                timeout=300,
                            )

                            walls = boq_process_walls(
                                records,
                                ext_thick=ext_t_mm,
                                int_thick=int_t_mm,
                            )
                            rooms = boq_process_rooms(records)
                            params = {
                                'ext_wall_thickness_m': ext_t_mm / 1000.0,
                                'int_wall_thickness_m': int_t_mm / 1000.0,
                                'floor_height_m':       floor_h_m,
                                'door_height_m':        door_h_m,
                                'window_height_m':      win_h_m,
                                'wet_tile_height_m':    wet_tile_m,
                            }
                            boq_result = boq_compute_boq(walls, rooms, records,
                                                          params=params)
                            payload = {
                                'records': records,
                                'walls':   walls,
                                'rooms':   rooms,
                                'boq':     boq_result,
                                'info': {
                                    'kind':    'image',
                                    'ext_len': walls['stats']['external_len_m'],
                                    'int_len': walls['stats']['internal_len_m'],
                                    'n_rooms': rooms['stats']['total'],
                                },
                            }

                        # --- stash + render ---
                        boq_state['records'] = payload['records']
                        boq_state['walls']   = payload['walls']
                        boq_state['rooms']   = payload['rooms']
                        boq_state['boq']     = payload['boq']
                        boq_state['info']    = payload['info']

                        _render_boq_results(payload)

                    except Exception as e:
                        boq_results.clear()
                        with boq_results:
                            ui.label(f'❌ Failed: {e}'
                                     ).classes('text-red-400 font-bold')
                            traceback.print_exc()

                # ---------------- RENDER ----------------
                def _render_boq_results(payload):
                    boq_results.clear()
                    boq_exports.clear()

                    boq_data = payload['boq']
                    summary = boq_data['summary']
                    boq_items = boq_data['boq']
                    rooms = payload['rooms']['rooms']
                    walls = payload['walls']['walls']

                    # --- confidence scan ---
                    all_records = payload.get('records') or []
                    n_low = sum(1 for r in all_records if r.get('confidence') == 'low')
                    n_total = len(all_records)
                    n_unknown_layer = sum(1 for r in all_records
                                          if not r.get('category'))

                    with boq_results:
                        ui.label('✅ BOQ generated'
                                 ).classes('text-xl font-bold text-green-400 mb-2')

                        if n_low > 0 or n_unknown_layer > 0:
                            with ui.column().classes(
                                'w-full bg-yellow-900/30 border border-yellow-500 '
                                'rounded-lg p-3 mb-3'
                            ):
                                ui.label('⚠️ Review recommended'
                                         ).classes('text-yellow-300 font-bold text-sm')
                                if n_unknown_layer > 0:
                                    ui.label(
                                        f'• {n_unknown_layer} element(s) had no '
                                        f'recognisable category (unknown layer).'
                                    ).classes('text-yellow-200 text-xs')
                                if n_low > 0:
                                    ui.label(
                                        f'• {n_low}/{n_total} element(s) matched with '
                                        f'low confidence — verify in the Walls / Rooms tables.'
                                    ).classes('text-yellow-200 text-xs')

                        with ui.row().classes('w-full gap-3 flex-wrap mb-3'):
                            for label, val, unit in [
                                ('Envelope', summary['envelope_area_m2'], 'm²'),
                                ('Net floor', summary['net_floor_area_m2'], 'm²'),
                                ('Ext walls', summary['ext_wall_len_m'], 'm'),
                                ('Int walls', summary['int_wall_len_m'], 'm'),
                                ('Wet area', summary['wet_area_m2'], 'm²'),
                                ('Dry area', summary['dry_area_m2'], 'm²'),
                                ('Doors', summary['door_count'], ''),
                                ('Windows', summary['window_count'], ''),
                            ]:
                                with ui.column().classes('stat-chip'):
                                    ui.label(f'{val}').classes('val')
                                    ui.label(f'{label} ({unit})' if unit else label).classes('lbl')

                        ui.label('📋 BOQ Quantities'
                                 ).classes('text-lg font-bold text-white mt-3 mb-2')
                        boq_cols = [
                            {'name': 'category', 'label': 'Category',
                             'field': 'category', 'sortable': True},
                            {'name': 'item', 'label': 'Item',
                             'field': 'item', 'sortable': True},
                            {'name': 'qty', 'label': 'Qty',
                             'field': 'qty', 'sortable': True},
                            {'name': 'unit', 'label': 'Unit',
                             'field': 'unit', 'sortable': True},
                            {'name': 'notes', 'label': 'Notes',
                             'field': 'notes', 'sortable': False},
                        ]
                        _dark_table(columns=boq_cols, rows=boq_items,
                                    row_key='item')

                        if rooms:
                            ui.label('🚪 Rooms'
                                     ).classes('text-lg font-bold text-white mt-4 mb-2')
                            room_rows = []
                            for r in rooms:
                                room_rows.append({
                                    'id': r['id'],
                                    'type': r['subtype'],
                                    'area_m2': round(r['geometry']['area_mm2'] / 1e6, 2),
                                    'confidence': r['confidence'],
                                    'reasons': ', '.join(r['meta'].get('reasons', [])) or '—',
                                })
                            _dark_table(
                                columns=[
                                    {'name': 'id', 'label': 'ID', 'field': 'id'},
                                    {'name': 'type', 'label': 'Type', 'field': 'type'},
                                    {'name': 'area_m2', 'label': 'Area (m²)',
                                     'field': 'area_m2', 'sortable': True},
                                    {'name': 'confidence', 'label': 'Confidence',
                                     'field': 'confidence'},
                                    {'name': 'reasons', 'label': 'Detected by',
                                     'field': 'reasons'},
                                ],
                                rows=room_rows, row_key='id')

                        if walls:
                            ui.label('🧱 Walls'
                                     ).classes('text-lg font-bold text-white mt-4 mb-2')
                            wall_rows = []
                            for w in walls:
                                g = w['geometry']
                                wall_rows.append({
                                    'id': w['id'],
                                    'subtype': w['subtype'],
                                    'length_m': round(g['length_mm'] / 1000.0, 3),
                                    'thickness_mm': round(g['width_mm'], 0),
                                    'method': w['meta'].get('method'),
                                    'confidence': w['confidence'],
                                })
                            _dark_table(
                                columns=[
                                    {'name': 'id', 'label': 'ID', 'field': 'id'},
                                    {'name': 'subtype', 'label': 'Type',
                                     'field': 'subtype'},
                                    {'name': 'length_m', 'label': 'Length (m)',
                                     'field': 'length_m', 'sortable': True},
                                    {'name': 'thickness_mm', 'label': 't (mm)',
                                     'field': 'thickness_mm', 'sortable': True},
                                    {'name': 'method', 'label': 'Method',
                                     'field': 'method'},
                                    {'name': 'confidence', 'label': 'Conf.',
                                     'field': 'confidence'},
                                ],
                                rows=wall_rows, row_key='id')

                    boq_exports.clear()
                    with boq_exports:
                        def export_excel():
                            try:
                                # Sheet 1: BOQ
                                items_df = pd.DataFrame(boq_items)

                                # Sheet 2: Summary
                                summary_df = pd.DataFrame(
                                    [{'metric': k, 'value': v}
                                     for k, v in summary.items()])

                                # Sheet 3: Elements (all raw records)
                                elem_rows = []
                                for r in (payload.get('records') or []):
                                    g = r.get('geometry') or {}
                                    elem_rows.append({
                                        'id':         r.get('id'),
                                        'category':   r.get('category'),
                                        'subtype':    r.get('subtype'),
                                        'layer':      r.get('layer'),
                                        'source':     r.get('source'),
                                        'confidence': r.get('confidence'),
                                        'geom_type':  g.get('type'),
                                        'length_mm':  g.get('length_mm'),
                                        'width_mm':   g.get('width_mm'),
                                        'area_mm2':   g.get('area_mm2'),
                                        'text':       r.get('text'),
                                    })
                                elem_df = pd.DataFrame(elem_rows)

                                # Sheet 4: Walls
                                wall_rows = []
                                for w in (payload.get('walls') or {}).get('walls', []):
                                    g = w['geometry']
                                    wall_rows.append({
                                        'id':           w['id'],
                                        'type':         w['subtype'],
                                        'length_m':     round(g['length_mm'] / 1000.0, 3),
                                        'thickness_mm': round(g['width_mm'], 0),
                                        'method':       w['meta'].get('method'),
                                        'confidence':   w['confidence'],
                                    })
                                walls_df = pd.DataFrame(wall_rows)

                                # Sheet 5: Rooms
                                room_rows = []
                                for r in (payload.get('rooms') or {}).get('rooms', []):
                                    room_rows.append({
                                        'id':         r['id'],
                                        'type':       r['subtype'],
                                        'area_m2':    round(r['geometry']['area_mm2'] / 1e6, 3),
                                        'perimeter_m':round(r['geometry']['length_mm'] / 1000.0, 3),
                                        'confidence': r['confidence'],
                                        'detected_by': ', '.join(r['meta'].get('reasons', [])) or '—',
                                        'labels':     ', '.join(r['meta'].get('labels', [])) or '',
                                    })
                                rooms_df = pd.DataFrame(room_rows)

                                bio = io.BytesIO()
                                with pd.ExcelWriter(bio, engine='openpyxl') as writer:
                                    items_df.to_excel(writer, sheet_name='BOQ', index=False)
                                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    if not walls_df.empty:
                                        walls_df.to_excel(writer, sheet_name='Walls', index=False)
                                    if not rooms_df.empty:
                                        rooms_df.to_excel(writer, sheet_name='Rooms', index=False)
                                    if not elem_df.empty:
                                        elem_df.to_excel(writer, sheet_name='Elements', index=False)
                                bio.seek(0)
                                ui.download(bio.read(),
                                            filename=f"BOQ_{boq_state['name']}.xlsx")
                                ui.notify('Excel downloaded! (5 sheets)', type='positive')
                            except Exception as ex:
                                ui.notify(f'Excel error: {ex}', type='negative')

                        def export_pdf():
                            try:
                                from reportlab.platypus import (
                                    SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle,
                                )
                                from reportlab.lib import colors
                                from reportlab.lib.styles import getSampleStyleSheet

                                bio = io.BytesIO()
                                doc = SimpleDocTemplate(bio, pagesize=PAGE_WIDTH,
                                                         rightMargin=MARGIN,
                                                         leftMargin=MARGIN,
                                                         topMargin=MARGIN,
                                                         bottomMargin=MARGIN)
                                styles = getSampleStyleSheet()
                                story = []
                                story.append(Paragraph("BILL OF QUANTITIES",
                                                        styles['Title']))
                                story.append(Spacer(1, 12))
                                story.append(Paragraph(
                                    f"Source: {boq_state['name']}",
                                    styles['Normal']))
                                story.append(Spacer(1, 8))

                                story.append(Paragraph("Summary", styles['Heading2']))
                                for k, v in summary.items():
                                    story.append(Paragraph(f"{k}: {v}",
                                                            styles['Normal']))
                                story.append(Spacer(1, 12))

                                story.append(Paragraph("BOQ Items",
                                                        styles['Heading2']))
                                data = [['Category', 'Item', 'Qty', 'Unit', 'Notes']]
                                for b in boq_items:
                                    data.append([
                                        str(b['category']),
                                        str(b['item']),
                                        f"{b['qty']:.3f}",
                                        str(b['unit']),
                                        str(b.get('notes', ''))[:60],
                                    ])
                                tbl = Table(data, repeatRows=1,
                                            colWidths=[70, 200, 60, 45, 180])
                                tbl.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0),
                                     colors.HexColor('#1B2A4A')),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                                    ('GRID', (0, 0), (-1, -1), 0.4,
                                     colors.HexColor('#94A3B8')),
                                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                ]))
                                story.append(tbl)
                                doc.build(story)
                                bio.seek(0)
                                ui.download(bio.read(),
                                            filename=f"BOQ_{boq_state['name']}.pdf")
                                ui.notify('PDF downloaded!', type='positive')
                            except Exception as ex:
                                ui.notify(f'PDF error: {ex}', type='negative')

                        ui.button('📊 Download Excel', on_click=export_excel
                                  ).classes('primary-btn')
                        ui.button('📄 Download PDF', on_click=export_pdf
                                  ).classes('primary-btn')

                ui.button('▶ Run', on_click=run_boq).classes('primary-btn mt-4')
                with boq_results:
                    ui.markdown('*Upload a file, choose mode, click Run.*'
                                ).classes('text-sm text-[#A9B6D0]')

        # ---------------- FOOTER ----------------
        ui.html('''
        <div class="app-footer">
            <b>Multi-Standard Engineering QA Portal</b> &nbsp;|&nbsp;
            ECP 203 &middot; ECP 202 &middot; ECP 104 &middot; ASTM &middot; AASHTO &middot; BS EN &middot; ISO<br>
            LinkedIn: <a href="https://www.linkedin.com/in/mohamed-abd-al-aty-a326a1214/" target="_blank">Mohamed Abd Al Aty</a>
            &nbsp;|&nbsp; Email: <a href="mailto:mohamedabdalaty63@gmail.com">mohamedabdalaty63@gmail.com</a><br>
            <i>Specialized in QA/QC & Automated Compliance.</i> &copy; 2026<br>
            <span style="color:#FFF;font-weight:600;">Disclaimer:</span>
            These AI modules must be rechecked by a qualified engineer before any decision-making.
        </div>
        ''')
