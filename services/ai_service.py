import asyncio
import json
import re
import io
import pypdf
import fitz
from nicegui import run
from google.genai import types
from config import client, GEMINI_MODEL, get_code_directive, NO_LATEX_RULE
from utils.boq import normalize_keys
from services.scraper_service import detect_mime_type

# --- Gemini API calls ---
async def call_gemini(contents, system_instruction=None, temperature=0.1, timeout=240):
    cfg_kwargs = {"temperature": temperature}
    if system_instruction:
        cfg_kwargs["system_instruction"] = system_instruction
    config = types.GenerateContentConfig(**cfg_kwargs)
    try:
        response = await asyncio.wait_for(
            run.io_bound(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            ),
            timeout=timeout
        )
        from services.pdf_service import sanitize_ai_markdown
        return sanitize_ai_markdown(response.text)
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")

async def call_gemini_json(contents, temperature=0.1, timeout=240):
    cfg_kwargs = {"temperature": temperature}
    config = types.GenerateContentConfig(**cfg_kwargs)
    try:
        response = await asyncio.wait_for(
            run.io_bound(
                client.models.generate_content,
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            ),
            timeout=timeout
        )
        return response.text
    except asyncio.TimeoutError:
        raise Exception("AI request timed out after 240 seconds.")
    except Exception as e:
        raise Exception(f"AI request failed: {str(e)}")

# --- Architectural extraction ---
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
- area_m2: the total floor area in square meters (if not given, compute from length × width)

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
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            doc.close()
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

# --- Mass (structural) extraction ---
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
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages_text = []
            for i in range(min(3, len(reader.pages))):
                try:
                    txt = reader.pages[i].extract_text() or ""
                    pages_text.append(txt)
                except:
                    pass
            full_text = "".join(pages_text)
            if full_text.strip():
                contents.append(f"Extracted text from PDF:\n{full_text[:6000]}")
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                contents.append(img_part)
            doc.close()
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
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    txt = "".join([p.extract_text() or "" for p in reader.pages[:3]])
                    if txt.strip():
                        contents2.append(f"Extracted text from PDF:\n{txt[:6000]}")
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

# --- Progress extraction from image ---
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
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                page = doc.load_page(0)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            doc.close()
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
