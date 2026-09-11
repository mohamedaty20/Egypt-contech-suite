"""
services/boq/extractor_image.py

Extract elements from PDF / PNG / JPG floor plans via Gemini vision.

Key differences vs DXF extraction:
  - Everything is approximate. Every record gets confidence="low" or "medium".
  - Requires a scale reference to convert pixels to mm:
        * User-supplied: two points + real distance, OR
        * User-supplied: "1 pixel = X mm", OR
        * AI reads a scale bar in the drawing (least reliable).
  - Output record format is IDENTICAL to extractor_dxf so the rest of the
    pipeline (wall_processor, room_processor, engine) works unchanged.
"""

import json
import re
import base64


# =====================================================================
# PROMPT
# =====================================================================
def _build_prompt(scale_hint=None):
    scale_block = ""
    if scale_hint:
        scale_block = (
            f"\nSCALE REFERENCE: {scale_hint}\n"
            f"Use this to convert all pixel measurements to millimetres."
        )

    return f"""You are a CAD-aware architectural drawing reader.
Analyze the provided floor plan and extract every element you can see.

Return ONE JSON object with this EXACT structure:

{{
  "scale_bar_found": true | false,
  "scale_bar_text": "1:100" or null,
  "elements": [
    {{
      "category": "wall" | "wall_ext" | "wall_int" | "column" | "beam" |
                  "slab" | "footing" | "door" | "window" | "stair" |
                  "balcony" | "void" | "room" | "text",
      "subtype": "external" | "internal" | null,
      "label": "Master Bedroom" or null,
      "shape": "line" | "rect" | "circle" | "polygon",
      "pixel_points": [[x1,y1], [x2,y2], ...],
      "pixel_length": 1234.5,
      "pixel_width":  250.0,
      "pixel_area":   500000.0,
      "estimated_real_length_mm": 5000,
      "estimated_real_width_mm": 250,
      "notes": "any text or dimension visible near this element"
    }}
  ]
}}

RULES:
- Do NOT invent elements. Only report what you can see.
- A wall is drawn as TWO PARALLEL LINES: report it as ONE wall.
- Every room must be listed as a separate element.
- Dimensions written as text ("4000", "4.00") should be captured in "notes"
  AND in "estimated_real_*" if the AI can compute them.
- Use pixel coordinates from the top-left of the image.
- If you cannot confidently classify something, omit it.
- Do NOT output markdown, prose, or explanations. Only the JSON object.
{scale_block}
"""


# =====================================================================
# RESPONSE PARSING
# =====================================================================
def _strip_fences(txt):
    t = txt.strip()
    t = re.sub(r'^```json\s*', '', t)
    t = re.sub(r'^```\s*', '', t)
    t = re.sub(r'\s*```$', '', t)
    return t


def _parse_ai_json(raw):
    if not raw:
        return None
    txt = _strip_fences(raw)
    start = txt.find('{')
    end = txt.rfind('}')
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(txt[start:end + 1])
    except Exception as e:
        print(f"[image] JSON parse failed: {e!r}")
        return None


# =====================================================================
# SCALE RESOLUTION
# =====================================================================
def _pixels_to_mm_factor(scale_info):
    """
    scale_info = {
      "mode": "pixel_ratio" | "two_points" | "scale_bar_text" | "unknown",
      "mm_per_pixel": float or None,
      "p1": [x,y] or None,
      "p2": [x,y] or None,
      "real_distance_mm": float or None,
    }

    Returns mm_per_pixel (float) or None if it can't be determined.
    """
    if not scale_info:
        return None

    mode = scale_info.get("mode")

    if mode == "pixel_ratio":
        return float(scale_info.get("mm_per_pixel") or 0) or None

    if mode == "two_points":
        p1 = scale_info.get("p1")
        p2 = scale_info.get("p2")
        real = scale_info.get("real_distance_mm")
        if p1 and p2 and real:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            pixel_dist = (dx * dx + dy * dy) ** 0.5
            if pixel_dist > 1:
                return real / pixel_dist
        return None

    # scale_bar_text and unknown → cannot resolve reliably
    return None


# =====================================================================
# MAIN
# =====================================================================
async def extract_elements_from_image(
        image_bytes,
        mime_type,
        call_gemini_json_fn,
        scale_info=None,
):
    """
    Extract elements from a PDF/PNG/JPG via Gemini vision.

    Args:
      image_bytes: bytes of the image OR first page of PDF as PNG
      mime_type:   'image/png' | 'image/jpeg'
      call_gemini_json_fn: async fn(contents, temperature, timeout) -> str
      scale_info:  dict (see _pixels_to_mm_factor)

    Returns: (records, stats)
    """
    from google.genai import types

    prompt = _build_prompt(scale_hint=_describe_scale(scale_info))

    img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    try:
        raw = await call_gemini_json_fn(
            [prompt, img_part],
            temperature=0.0,
            timeout=240,
        )
    except Exception as e:
        print(f"[image] Gemini call failed: {e!r}")
        return [], {"by_category": {}, "error": str(e)}

    data = _parse_ai_json(raw)
    if not data:
        return [], {"by_category": {}, "error": "unparseable AI response"}

    mm_per_pixel = _pixels_to_mm_factor(scale_info)

    elements = data.get("elements") or []
    records = []
    stats = {"by_category": {}, "unknown_scale": mm_per_pixel is None}

    for idx, el in enumerate(elements):
        cat = el.get("category")
        if not cat:
            continue

        # ---- geometry in mm ----
        pts_px = el.get("pixel_points") or []
        pts_mm = []
        if mm_per_pixel:
            pts_mm = [[float(p[0]) * mm_per_pixel, float(p[1]) * mm_per_pixel]
                      for p in pts_px if len(p) >= 2]

        length_mm = el.get("estimated_real_length_mm")
        if not length_mm and el.get("pixel_length") and mm_per_pixel:
            length_mm = float(el["pixel_length"]) * mm_per_pixel

        width_mm = el.get("estimated_real_width_mm")
        if not width_mm and el.get("pixel_width") and mm_per_pixel:
            width_mm = float(el["pixel_width"]) * mm_per_pixel

        area_mm2 = el.get("pixel_area")
        if area_mm2 and mm_per_pixel:
            area_mm2 = float(area_mm2) * (mm_per_pixel ** 2)
        elif not area_mm2:
            area_mm2 = None

        rec = {
            "id":         f"img_{cat}_{idx:05d}",
            "category":   cat,
            "subtype":    el.get("subtype"),
            "layer":      None,
            "source":     "ai_vision",
            "units":      "mm",
            "geometry": {
                "type":       _geom_type_from_shape(el.get("shape")),
                "points":     pts_mm,
                "length_mm":  float(length_mm) if length_mm else None,
                "width_mm":   float(width_mm) if width_mm else None,
                "area_mm2":   float(area_mm2) if area_mm2 else None,
                "count":      1,
            },
            "text":       el.get("label") or el.get("notes"),
            "confidence": "medium" if mm_per_pixel else "low",
            "meta": {
                "pixel_points": pts_px,
                "mm_per_pixel": mm_per_pixel,
                "notes":        el.get("notes"),
            },
        }
        records.append(rec)
        stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

    print(f"[image] extracted {len(records)} elements  "
          f"scale={'known' if mm_per_pixel else 'UNKNOWN (all mm approximate)'}")
    print(f"[image] by_category: {stats['by_category']}")

    return records, stats


# =====================================================================
# HELPERS
# =====================================================================
def _geom_type_from_shape(shape):
    s = (shape or "").lower()
    if s == "line":     return "line"
    if s == "rect":     return "polyline"
    if s == "polygon":  return "polyline"
    if s == "circle":   return "circle"
    return "polyline"


def _describe_scale(scale_info):
    if not scale_info:
        return None
    mode = scale_info.get("mode")
    if mode == "pixel_ratio":
        return f"1 pixel = {scale_info.get('mm_per_pixel')} mm"
    if mode == "two_points":
        p1 = scale_info.get("p1")
        p2 = scale_info.get("p2")
        real = scale_info.get("real_distance_mm")
        return (f"Two reference points were clicked at pixel "
                f"{p1} and {p2}, real distance = {real} mm. "
                f"Compute mm/pixel from these.")
    return None
