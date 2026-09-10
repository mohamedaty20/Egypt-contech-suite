import asyncio
import json
import re
import io
import pypdf
import fitz
from nicegui import run
from google.genai import types
from config import (
    client, GEMINI_MODEL, get_code_directive, NO_LATEX_RULE,
    cpu_bound_limited, _get_gemini_sem,
)
from utils.boq import normalize_keys
from services.scraper_service import detect_mime_type


# =====================================================================
# ADDITIVE: process-pool workers for PDF work
# ---------------------------------------------------------------------
# Module-level so NiceGUI's run.cpu_bound can pickle them.
# They take plain bytes and return plain bytes/str — nothing crosses
# the process boundary that isn't picklable.
# =====================================================================

def _pdf_first_page_to_png_bytes(pdf_bytes, zoom=2.0):
    """Worker: PDF bytes -> PNG bytes of page 1 (or None)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if len(doc) == 0:
                return None
            page = doc.load_page(0)
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            return pix.tobytes("png")
        finally:
            doc.close()
    except Exception:
        return None


def _pdf_extract_text(pdf_bytes, max_pages=3, max_chars=6000):
    """Worker: PDF bytes -> concatenated text from first N pages."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for i in range(min(max_pages, len(reader.pages))):
            try:
                txt = reader.pages[i].extract_text() or ""
                parts.append(txt)
            except Exception:
                pass
        return "".join(parts)[:max_chars]
    except Exception:
        return ""


# =====================================================================
# Gemini API calls
# =====================================================================
async def call_gemini(contents, system_instruction=None, temperature=0.1, timeout=240):
    cfg_kwargs = {"temperature": temperature}
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    config = types.GenerateContentConfig(**cfg_kwargs)
    sem = _get_gemini_sem()
    try:
        async def _do():
            return await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
        async with sem:
            response = await asyncio.wait_for(_do(), timeout=timeout)
        from services.pdf_service import sanitize_ai_markdown
        return sanitize_ai_markdown(response.text)
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")


async def call_gemini_json(contents, temperature=0.1, timeout=240):
    cfg_kwargs = {"temperature": temperature}
    config = types.GenerateContentConfig(**cfg_kwargs)
    sem = _get_gemini_sem()
    try:
        async def _do():
            return await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
        async with sem:
            response = await asyncio.wait_for(_do(), timeout=timeout)
        return response.text
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")


# =====================================================================
# Architectural extraction
# =====================================================================
async def extract_architectural_with_ai(element_type, file_bytes, file_type, user_params, code_basis, retry=True):
    contents = []
    from config import ARCH_SCHEMAS
    schema_info = ARCH_SCHEMAS.get(element_type)
    if not schema_info:
        raise ValueError(f"Unsupported architectural element: {element_type}")

    if element_type == 'flooring':
        prompt = f"""
You are a Quantity Surveyor. Extract the building dimensions from the architectural plan.
From the drawing, determine:
- total_length_m: the overall length of the building in meters
- total_width_m: the overall width of the building in meters
- area_m2: the total floor area in square meters (if not given, compute from length x width)

If a dimension is not clearly visible, set it to null.
Return ONLY a JSON object with these fields, no extra text.

Example:
{{"total_length_m": 20.0, "total_width_m": 15.0, "area_m2": 300.0}}
"""
    elif element_type == 'wall_finishing':
        prompt = f"""
You are a Quantity Surveyor. Extract the total wall finishing area from the architectural plan.
Determine the total area of walls that need finishing (paint, plaster, etc.) in square meters.
This is often given as a total wall area or can be computed from perimeter and height.
Return ONLY a JSON object with field "total_area_m2", no extra text.
Example: {{"total_area_m2": 250.0}}
"""
    elif element_type == 'ceilings':
        prompt = f"""
You are a Quantity Surveyor. Extract the total ceiling area from the architectural plan.
This is usually the same as the floor area (or given separately).
Return ONLY a JSON object with field "total_area_m2", no extra text.
Example: {{"total_area_m2": 300.0}}
"""
    elif element_type == 'doors_windows':
        prompt = f"""
You are a Quantity Surveyor. Count the number of doors and windows from the architectural plan.
Return ONLY a JSON object with fields "door_count" and "window_count", no extra text.
Example: {{"door_count": 10, "window_count": 15}}
"""
    else:
        raise ValueError(f"Unsupported architectural element: {element_type}")

    contents.append(prompt)

    if file_type == 'application/pdf':
        try:
            img_bytes = await cpu_bound_limited(_pdf_first_page_to_png_bytes, file_bytes)
            if img_bytes:
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            else:
                contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
        except Exception:
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        img_part = types.Part.from_bytes(data=file_bytes, mime_type=file_type)
        contents.append(img_part)

    raw_response = None
    try:
        response_text = await call_gemini_json(contents, temperature=0, timeout=240)
        raw_response = response_text
        json_str = response_text.strip()
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        start = json_str.find('{')
        end = json_str.rfind('}')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        data = json.loads(json_str)
        aliases = schema_info.get('field_aliases', {})
        norm_data = normalize_keys(data, aliases)
        return norm_data, raw_response
    except Exception as e:
        if retry:
            prompt2 = f"""
Return a JSON object with the fields: {', '.join(schema_info['required'])}.
If unclear, set values to null.
"""
            contents2 = [prompt2]
            try:
                response_text2 = await call_gemini_json(contents2, temperature=0, timeout=240)
                raw_response = response_text2
                json_str2 = response_text2.strip()
                json_str2 = re.sub(r'^```json\s*', '', json_str2)
                json_str2 = re.sub(r'\s*```$', '', json_str2)
                start = json_str2.find('{')
                end = json_str2.rfind('}')
                if start != -1 and end != -1:
                    json_str2 = json_str2[start:end+1]
                data2 = json.loads(json_str2)
                aliases = schema_info.get('field_aliases', {})
                norm_data2 = normalize_keys(data2, aliases)
                return norm_data2, raw_response
            except:
                return {}, raw_response
        else:
            return {}, raw_response


# =====================================================================
# Mass (structural) extraction
# =====================================================================
async def extract_mass_with_ai(element_type, file_bytes, file_type, user_params, code_basis, retry=True):
    contents = []
    from config import MASS_SCHEMAS
    schema_info = MASS_SCHEMAS.get(element_type)
    if not schema_info:
        raise ValueError(f"Unsupported element type: {element_type}")

    extra_instruction = ""
    if element_type == 'columns':
        extra_instruction = " IMPORTANT: Only count columns that are part of the structural grid/plan. Ignore any columns shown in a separate schedule, detail sheet, or table. "

    prompt = f"""
You are an expert Quantity Surveyor. Your task is to EXTRACT raw data from the provided drawing(s) and return ONLY a JSON array of objects.

Extract the following fields for each group:
{', '.join(schema_info['required'])}

If a dimension is not clearly visible, set it to null.
{extra_instruction}
Return ONLY the JSON array, no extra text, no explanations, no markdown.

Example for columns:
[{{"label":"C1","count":6,"width_mm":300,"depth_mm":300,"height_mm":3000}}]

Now extract from the drawing.
"""
    contents.append(prompt)

    if file_type == 'application/pdf':
        try:
            full_text = await cpu_bound_limited(_pdf_extract_text, file_bytes, 3, 6000)
            if full_text.strip():
                contents.append(f"Extracted text from PDF:\n{full_text}")
            img_bytes = await cpu_bound_limited(_pdf_first_page_to_png_bytes, file_bytes)
            if img_bytes:
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            else:
                contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
        except Exception as e:
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        img_part = types.Part.from_bytes(data=file_bytes, mime_type=file_type)
        contents.append(img_part)

    try:
        response_text = await call_gemini_json(contents, temperature=0, timeout=240)
        json_str = response_text.strip()
        json_str = re.sub(r'^```json\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)
        start = json_str.find('[')
        end = json_str.rfind(']')
        if start != -1 and end != -1:
            json_str = json_str[start:end+1]
        data = json.loads(json_str)
        return data
    except Exception as e:
        if retry:
            prompt2 = f"""
Return a JSON array of objects with fields: {', '.join(schema_info['required'])}.
{extra_instruction}
If the drawing is unclear, return an empty array [].
"""
            contents2 = [prompt2]
            if file_type == 'application/pdf':
                try:
                    txt = await cpu_bound_limited(_pdf_extract_text, file_bytes, 3, 6000)
                    if txt.strip():
                        contents2.append(f"Extracted text from PDF:\n{txt}")
                except:
                    pass
            try:
                response_text2 = await call_gemini_json(contents2, temperature=0, timeout=240)
                json_str2 = response_text2.strip()
                json_str2 = re.sub(r'^```json\s*', '', json_str2)
                json_str2 = re.sub(r'\s*```$', '', json_str2)
                start = json_str2.find('[')
                end = json_str2.rfind(']')
                if start != -1 and end != -1:
                    json_str2 = json_str2[start:end+1]
                data2 = json.loads(json_str2)
                return data2
            except:
                return []
        else:
            return []


# =====================================================================
# Progress extraction
# =====================================================================
def parse_progress_from_gemini_response(raw_text):
    raw_text = raw_text.strip()
    raw_text = re.sub(r'^```json\s*', '', raw_text)
    raw_text = re.sub(r'\s*```$', '', raw_text)
    start = raw_text.find('{')
    end = raw_text.rfind('}')
    if start != -1 and end != -1:
        raw_text = raw_text[start:end+1]
    try:
        return json.loads(raw_text)
    except:
        return {}


async def extract_progress_from_image(file_bytes, file_type):
    if not client:
        raise Exception("GEMINI_API_KEY missing.")
    prompt = """
You are a construction project progress reporter. Analyze the provided image (which may show site photos, documents, or progress reports).
Extract the following information and return a JSON object with these fields (if visible):
- "date": date in YYYY-MM-DD format (if any text states a date)
- "description": a brief description of the work shown (e.g., "Concrete pour for foundation", "Steel fixing for column")
- "progress_percent": estimated progress percentage (0-100) for the activity shown
- "category": category of work (e.g., "Concrete", "Steel", "Finishing", "Excavation", "MEP")
- "location": specific location if mentioned (e.g., "Block A", "Floor 2")

If any field is not visible, set it to null.
Return ONLY the JSON, no extra text.
Example: {"date": "2026-03-15", "description": "Formwork installation for slab", "progress_percent": 60, "category": "Formwork", "location": "Block B"}
"""
    contents = [prompt]
    if file_type == 'application/pdf':
        try:
            img_bytes = await cpu_bound_limited(_pdf_first_page_to_png_bytes, file_bytes)
            if img_bytes:
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            else:
                contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
        except:
            contents.append(types.Part.from_bytes(data=file_bytes, mime_type='application/pdf'))
    else:
        contents.append(types.Part.from_bytes(data=file_bytes, mime_type=file_type))
    raw_response = await call_gemini_json(contents, temperature=0, timeout=120)
    data = parse_progress_from_gemini_response(raw_response)
    return data


async def generate_progress_overview(df, start_date, end_date, description):
    if df.empty or not client:
        return "No data available for overview."
    csv_data = df.to_csv(index=False)
    prompt = f"""
You are a project management analyst. Given the following progress data for a construction project from {start_date} to {end_date}, 
with description: "{description}", provide a concise executive summary (2-3 paragraphs) highlighting:
- Overall progress status
- Key activities and their progress
- Any potential risks or delays
- Recommendations for next steps

Data (CSV format):
{csv_data}
"""
    try:
        overview = await call_gemini(prompt, temperature=0.3, timeout=120)
        return overview
    except Exception as e:
        return f"Error generating overview: {str(e)}"


# =====================================================================
# AI Architectural Plan (fallback room program)
# =====================================================================
async def plan_architectural_layout(plot_data):
    """Use Gemini to produce a room program that follows Egyptian building code."""
    prompt = f"""
You are a senior Egyptian architect with 30 years of experience.
Design a complete residential room program for the following plot, strictly following:
- Egyptian Building Law 119/2008 (setbacks, heights, manwer)
- ECP 203 for structural spans (max 5 m clear span for typical RC)
- New Cairo / Giza municipal setback rules

PLOT:
- Area: {plot_data['plot_area_m2']} m2
- Dimensions: {plot_data.get('plot_width', '?')} m x {plot_data.get('plot_length', '?')} m
- Street width: {plot_data['street_width_m']} m
- Location: {plot_data['location']}
- Floors: {plot_data['num_floors']}
- Units per floor: {plot_data.get('num_units_per_floor', 1)}
- Bedrooms wanted: {plot_data.get('num_bedrooms', 3)}
- Bathrooms wanted: {plot_data.get('num_bathrooms', 2)}
- User wish: {plot_data.get('user_description', 'Standard Egyptian family home')}

ROOM RULES (enforce strictly):
- Master bedroom >= 14 m2; other bedrooms >= 10 m2
- Living/Reception >= 20 m2
- Kitchen >= 7 m2 (must have exterior wall for window)
- Bathroom >= 3.5 m2
- Corridor width >= 1.1 m
- Stair width >= 1.1 m
- Manwer (light well 2x2 m min) if plot < 175 m2 AND floors >= 2

Return ONLY valid JSON (no markdown, no code fences). Structure:

{{
  "rooms": [
    {{"name": "Living Room", "type": "living", "area_m2": 28, "priority": 1, "zone": "public", "needs_window": true}},
    {{"name": "Kitchen", "type": "kitchen", "area_m2": 9, "priority": 2, "zone": "public", "needs_window": true}},
    {{"name": "Master Bedroom", "type": "bedroom_master", "area_m2": 16, "priority": 3, "zone": "private", "needs_window": true}},
    {{"name": "Bedroom 2", "type": "bedroom", "area_m2": 12, "priority": 4, "zone": "private", "needs_window": true}},
    {{"name": "Bathroom", "type": "bathroom", "area_m2": 4, "priority": 5, "zone": "private", "needs_window": false}}
  ],
  "core": {{"type": "staircase", "position": "center"}},
  "manwer_required": false,
  "manwer_size_m": 0,
  "compliance_notes": "Front setback 3 m (street > 12 m). Max height 4 floors."
}}
"""
    try:
        text = await call_gemini_json([prompt], temperature=0.3, timeout=180)
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1:
            text = text[s:e+1]
        return json.loads(text)
    except Exception as ex:
        print(f"[AI plan] fallback: {ex}")
        return {
            "rooms": [
                {"name": "Living Room", "type": "living", "area_m2": 25, "priority": 1, "zone": "public", "needs_window": True},
                {"name": "Kitchen", "type": "kitchen", "area_m2": 9, "priority": 2, "zone": "public", "needs_window": True},
                {"name": "Master Bedroom", "type": "bedroom_master", "area_m2": 16, "priority": 3, "zone": "private", "needs_window": True},
                {"name": "Bedroom 2", "type": "bedroom", "area_m2": 12, "priority": 4, "zone": "private", "needs_window": True},
                {"name": "Bathroom", "type": "bathroom", "area_m2": 4, "priority": 5, "zone": "private", "needs_window": False},
            ],
            "core": {"type": "staircase", "position": "center"},
            "manwer_required": False,
            "manwer_size_m": 0,
            "compliance_notes": "Fallback layout."
        }


# =====================================================================
# AI FULL LAYOUT DESIGNER (draws real positioned rectangles)
# =====================================================================
async def design_layout_with_ai(plot_data):
    """AI designs the actual positioned rectangles. Returns full layout JSON."""
    pw_mm = int(plot_data.get('plot_width', 12) * 1000)
    pl_mm = int(plot_data.get('plot_length', 16) * 1000)
    street = plot_data.get('street_side', 'S')

    prompt = f"""You are a master Egyptian architect. Design a REAL residential floor plan as POSITIONED rectangles in millimetres.

PLOT: {pw_mm} x {pl_mm} mm. Street faces {street} side.
Floors: {plot_data.get('num_floors', 2)}. Bedrooms: {plot_data.get('num_bedrooms', 3)}. Bathrooms: {plot_data.get('num_bathrooms', 2)}.
User wish: {plot_data.get('user_description', 'Standard Egyptian family home')}

EGYPTIAN CODE RULES (must obey):
- Building footprint must be inside plot with setbacks: front 2500mm, rear 2000mm, each side 1500mm (from plot edge)
- Corridor width >= 1100 mm
- Stair core >= 2200 x 3200 mm, must touch corridor
- Master bedroom >= 14 m2, other bedrooms >= 10 m2, kitchen >= 7 m2, bathroom >= 3.5 m2, living >= 20 m2, dining >= 10 m2
- Every room must have a door on the corridor side
- Living, kitchen, all bedrooms must have a window on an EXTERIOR wall
- No two rooms may overlap

COORDINATE SYSTEM: (0,0) at bottom-left of plot. X to the right, Y up. Room (x,y) is bottom-left corner, (w,h) is width/height. All in mm.

Return ONLY valid JSON, no markdown, no fences:

{{
  "building": {{"x": 1500, "y": 2500, "w": {pw_mm-3000}, "h": {pl_mm-4500}}},
  "corridor": {{"x": 2000, "y": 9000, "w": {pw_mm-4000}, "h": 1300}},
  "core": {{"x": 9000, "y": 7000, "w": 2500, "h": 3500}},
  "rooms": [
    {{"name": "Living Room", "type": "living", "x": 2000, "y": 2700, "w": 5000, "h": 6000,
      "door_wall": "N", "door_pos": 3000, "window_walls": ["S", "W"]}},
    {{"name": "Kitchen", "type": "kitchen", "x": 7200, "y": 2700, "w": 3500, "h": 4000,
      "door_wall": "N", "door_pos": 2000, "window_walls": ["S"]}},
    {{"name": "Master Bedroom", "type": "bedroom_master", "x": 2000, "y": 10500, "w": 4500, "h": 4200,
      "door_wall": "S", "door_pos": 2500, "window_walls": ["W", "N"]}}
  ]
}}

Design something BEAUTIFUL and FUNCTIONAL for real Egyptian families. Vary the layout based on the inputs. Position rooms to minimize wasted corridor space. Group wet rooms (kitchen, bathrooms) together for plumbing efficiency. Place bedrooms away from street noise."""

    try:
        text = await call_gemini_json([prompt], temperature=0.7, timeout=300)
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1:
            text = text[s:e+1]
        return json.loads(text)
    except Exception as ex:
        print(f"[AI design] failed: {ex}")
        return None


async def refine_layout_with_ai(previous_layout, violations, plot_data):
    """Send violations back to AI to fix the layout."""
    prompt = f"""You are a master Egyptian architect. Your previous design has code violations.
Fix ALL of them and return the corrected full layout JSON.

PREVIOUS LAYOUT:
{json.dumps(previous_layout, indent=2)}

VIOLATIONS TO FIX:
{chr(10).join('- ' + v for v in violations)}

PLOT: {int(plot_data.get('plot_width',12)*1000)} x {int(plot_data.get('plot_length',16)*1000)} mm.
Street on {plot_data.get('street_side','S')} side.

Egyptian code minimums:
- Corridor >= 1100 mm, core >= 2200x3200 mm
- Master BR >= 14 m2, BR >= 10 m2, kitchen >= 7 m2, bath >= 3.5 m2, living >= 20 m2
- Every room needs a door on the corridor-facing wall
- Living/kitchen/bedrooms need a window on exterior wall
- No overlaps, all inside building bounds

Return ONLY valid JSON (same structure as before), no markdown, no explanation."""

    try:
        text = await call_gemini_json([prompt], temperature=0.4, timeout=300)
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1:
            text = text[s:e+1]
        return json.loads(text)
    except Exception as ex:
        print(f"[AI refine] failed: {ex}")
        return None


# =====================================================================
# ADDITIVE: sophisticated, varied AI layout designer
# ---------------------------------------------------------------------
# Nothing above this line was changed. Everything below is new:
#   - _LAYOUT_STYLES : curated design vocabulary
#   - design_layout_with_ai_enhanced() : full positioned floor plan,
#     high-temperature, style-hinted, different output every call
#   - validate_layout() : checks each design against ECP minimums and
#     returns a list of violations (empty = clean)
# =====================================================================

import random as _random


_LAYOUT_STYLES = [
    ("Contemporary open-plan villa",
     "wide central living-dining axis, minimal corridors, sliding glass to garden, "
     "guest powder room near entry, master suite on quiet rear corner"),
    ("Traditional Egyptian mandara layout",
     "formal guest reception (mandara) at entry with separate WC, family living "
     "deeper in plan, kitchens with service yard access, bedrooms clustered rear"),
    ("Compact urban efficiency",
     "minimal circulation, stacked wet walls, bedrooms side by side, entry foyer "
     "opening directly to living, no wasted corridor area"),
    ("Luxury family residence",
     "walk-in closets, en-suite master, family lounge separate from formal living, "
     "large kitchen with island and pantry, laundry room, guest suite with own bath"),
    ("Multi-generational home",
     "two separate bedroom wings sharing a communal living core, ground floor "
     "bedroom with accessible bathroom, secondary entrance"),
    ("Courtyard-orientation home",
     "central internal light well (manwar), all major rooms open onto courtyard, "
     "L-shaped footprint around the void"),
    ("Long narrow plot optimiser",
     "single-loaded corridor along one side, service rooms stacked, bedrooms with "
     "cross ventilation front-to-back"),
    ("Garden-facing residence",
     "living, dining, and master all on the rear façade, service rooms front, "
     "guest wing separate"),
]


async def design_layout_with_ai_enhanced(plot_data, style_hint=None, temperature=0.95):
    """Ask Gemini for a FULL positioned floor plan. Returns dict with
    building/corridor/rooms, or None on failure. Uses a random style hint
    and high temperature so every call produces a different design."""
    pw_mm = int((plot_data.get('plot_width') or 12) * 1000)
    pl_mm = int((plot_data.get('plot_length') or 16) * 1000)
    street = plot_data.get('street_side', 'S')
    if style_hint is None:
        style_hint = _random.choice(_LAYOUT_STYLES)

    if isinstance(style_hint, tuple):
        style_name, style_desc = style_hint
    else:
        style_name, style_desc = "Custom", style_hint

    prompt = f"""You are a master Egyptian architect. Design a REAL residential floor plan as POSITIONED rectangles in millimetres.

STYLE FOR THIS DESIGN (obey strictly): {style_name}
Style details: {style_desc}

PLOT: {pw_mm} x {pl_mm} mm. Street faces {street} side.
Floors: {plot_data.get('num_floors', 2)}.
Bedrooms: {plot_data.get('num_bedrooms', 3)}.
Bathrooms: {plot_data.get('num_bathrooms', 2)}.
User wish: {plot_data.get('user_description', 'Standard Egyptian family home')}

EGYPTIAN CODE RULES (MUST OBEY, ECP 203 + Law 119/2008):
- Setbacks: front 2500mm, rear 2000mm, each side 1500mm from plot edge
- Corridor width >= 1100 mm
- Stair core >= 2200 x 3200 mm, must touch corridor
- Master bedroom >= 14 m2, other bedrooms >= 10 m2
- Living >= 20 m2
- Kitchen >= 7 m2 (must have exterior wall for window)
- Bathroom >= 3.5 m2
- Dining (if present) >= 10 m2
- Every room must have a door on the corridor side
- Living, kitchen, all bedrooms MUST have a window on an EXTERIOR wall
- NO TWO ROOMS MAY OVERLAP
- Every room must be INSIDE the building bounds

COORDINATE SYSTEM: (0,0) at bottom-left of the PLOT (not the building).
X increases right, Y increases up. Room (x,y,w,h) is bottom-left corner + size.
All values in MILLIMETRES.

Return ONLY a JSON object, no prose, no markdown fences:

{{
  "building": {{"x": 1500, "y": 2500, "w": {pw_mm-3000}, "h": {pl_mm-4500}}},
  "corridor": {{"x": 1600, "y": 8000, "w": {pw_mm-3200}, "h": 1200}},
  "core": {{"x": 8500, "y": 6500, "w": 2400, "h": 3600}},
  "rooms": [
    {{"name": "Living Room", "type": "living", "x": 1600, "y": 2600, "w": 5000, "h": 5200,
      "door_wall": "N", "door_pos": 2500, "window_walls": ["S", "W"]}},
    {{"name": "Kitchen", "type": "kitchen", "x": 6800, "y": 2600, "w": 3200, "h": 4000,
      "door_wall": "N", "door_pos": 1600, "window_walls": ["S"]}}
  ],
  "design_notes": "Brief 1-2 sentence description of the layout concept"
}}

Design a DIFFERENT, rich layout each time. Vary room sizes within code limits.
Place wet rooms (kitchen + baths) together for plumbing efficiency.
Place bedrooms away from street noise. Fill the plot efficiently without waste."""

    try:
        text = await call_gemini_json([prompt], temperature=temperature, timeout=300)
        text = re.sub(r'^```json\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)
        s, e = text.find('{'), text.rfind('}')
        if s != -1 and e != -1:
            text = text[s:e+1]
        data = json.loads(text)
        if not data.get('rooms'):
            return None
        data['_style'] = style_name
        return data
    except Exception as ex:
        print(f"[design_layout_with_ai_enhanced] failed: {ex}")
        return None


def validate_layout(layout, plot_data):
    """Return list of violation strings; empty list = valid."""
    violations = []
    rooms = layout.get('rooms', [])
    if not rooms:
        return ["No rooms produced"]

    pw_mm = int((plot_data.get('plot_width') or 12) * 1000)
    pl_mm = int((plot_data.get('plot_length') or 16) * 1000)

    for r in rooms:
        name = r.get('name', '?')
        try:
            x = float(r.get('x', 0)); y = float(r.get('y', 0))
            w = float(r.get('w', 0)); h = float(r.get('h', 0))
        except Exception:
            violations.append(f"{name}: non-numeric x/y/w/h")
            continue
        if w <= 0 or h <= 0:
            violations.append(f"{name}: zero or negative size")
            continue
        area_m2 = (w * h) / 1_000_000
        nm = name.lower()
        if 'master' in nm and area_m2 < 12:
            violations.append(f"{name}: {area_m2:.1f} m2 < 12 m2 min")
        elif 'bedroom' in nm and area_m2 < 9:
            violations.append(f"{name}: {area_m2:.1f} m2 < 9 m2 min")
        elif 'living' in nm and area_m2 < 18:
            violations.append(f"{name}: {area_m2:.1f} m2 < 18 m2 min")
        elif 'kitchen' in nm and area_m2 < 6:
            violations.append(f"{name}: {area_m2:.1f} m2 < 6 m2 min")
        elif 'bath' in nm and area_m2 < 3:
            violations.append(f"{name}: {area_m2:.1f} m2 < 3 m2 min")
        if x < 0 or y < 0 or (x + w) > pw_mm or (y + h) > pl_mm:
            violations.append(f"{name}: outside plot bounds")

    # Overlap check
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            r1, r2 = rooms[i], rooms[j]
            try:
                x1, y1 = float(r1['x']), float(r1['y'])
                w1, h1 = float(r1['w']), float(r1['h'])
                x2, y2 = float(r2['x']), float(r2['y'])
                w2, h2 = float(r2['w']), float(r2['h'])
            except Exception:
                continue
            if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
                ov = min(x1 + w1, x2 + w2) - max(x1, x2)
                ov *= min(y1 + h1, y2 + h2) - max(y1, y2)
                if ov > 100_000:  # more than 0.1 m2 overlap
                    violations.append(
                        f"{r1.get('name')} overlaps {r2.get('name')} by {ov/1e6:.2f} m2")
    return violations
