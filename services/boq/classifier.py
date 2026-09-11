"""
services/boq/classifier.py

Fallback classifier. Only used when taxonomy.classify_layer() returns
category=None. Sends the UNKNOWN layer names (no geometry) to Gemini and
asks it to assign each to one of the known categories.

Cheap call — no images, no geometry, ~200 tokens per file.
"""

import json
import re
from .taxonomy import LAYER_TAXONOMY


# =====================================================================
# PROMPT
# =====================================================================
def _build_prompt(unknown_layers):
    """
    Build a strict JSON-only prompt listing the unknown layers and
    the allowed category keys.
    """
    allowed = sorted(LAYER_TAXONOMY.keys())
    layers_txt = "\n".join(f'  - "{n}"' for n in unknown_layers)

    return f"""You are a CAD layer classifier for Egyptian architectural drawings.

The following layer names could not be matched to any known category.
Assign each one to EXACTLY ONE of these categories (or null if truly unknown):

{", ".join(allowed)}

Layer names:
{layers_txt}

Rules:
- "WLL", "wal", "wallx", "حائط" -> "wall"
- Anything containing "ext", "external", "خارجي" -> "wall_ext"
- Anything containing "int", "internal", "داخلي" -> "wall_int"
- Layer names starting with "S-" or "STR-" are structural.
- Fixture layers (toilet, sink, sanitary) are NOT walls — use "furniture".
- If unsure, return null rather than guessing.

Return ONLY this JSON (no markdown, no prose):
{{"assignments": [{{"layer": "...", "category": "..."}}, ...]}}
"""


# =====================================================================
# MAIN
# =====================================================================
async def classify_unknown_layers(unknown_layers, call_gemini_json_fn):
    """
    Classify unmatched layer names via Gemini.

    Args:
      unknown_layers: list of layer name strings (raw, as they appear in the file)
      call_gemini_json_fn: async fn(prompt, temperature, timeout) -> str

    Returns:
      dict: {layer_name: {"category": str|None, "confidence": "medium"|"low"}}
    """
    if not unknown_layers:
        return {}

    # Cap at 80 names — beyond that it's likely a broken file
    unknown_layers = list(dict.fromkeys(unknown_layers))[:80]

    prompt = _build_prompt(unknown_layers)

    try:
        raw = await call_gemini_json_fn(prompt, temperature=0.0, timeout=60)
    except Exception as e:
        print(f"[classifier] AI call failed: {e!r}")
        return {n: {"category": None, "confidence": "low"} for n in unknown_layers}

    # Strip fences
    txt = raw.strip()
    txt = re.sub(r'^```json\s*', '', txt)
    txt = re.sub(r'^```\s*', '', txt)
    txt = re.sub(r'\s*```$', '', txt)

    start = txt.find('{')
    end = txt.rfind('}')
    if start == -1 or end == -1:
        print("[classifier] no JSON object in response")
        return {n: {"category": None, "confidence": "low"} for n in unknown_layers}

    try:
        data = json.loads(txt[start:end + 1])
    except Exception as e:
        print(f"[classifier] JSON parse failed: {e!r}")
        return {n: {"category": None, "confidence": "low"} for n in unknown_layers}

    result = {}
    allowed = set(LAYER_TAXONOMY.keys())

    for entry in data.get("assignments", []):
        name = entry.get("layer", "").strip()
        cat = entry.get("category")
        if not name:
            continue
        if cat not in allowed:
            cat = None
        result[name] = {
            "category": cat,
            "confidence": "medium" if cat else "low",
        }

    # Any layer not mentioned by the AI → null
    for n in unknown_layers:
        result.setdefault(n, {"category": None, "confidence": "low"})

    matched = sum(1 for v in result.values() if v["category"])
    print(f"[classifier] AI classified {matched}/{len(unknown_layers)} unknown layers")
    return result
