"""
services/boq/taxonomy.py

Layer-name classification with fuzzy matching.

The whole BOQ pipeline depends on knowing what each DXF layer *means*.
Users name layers differently ("WLL", "wall", "A-WALL-EXT", "wal").
This module resolves them to canonical categories.
"""

# =====================================================================
# THE TAXONOMY
# ---------------------------------------------------------------------
# Each key is a canonical category. The list contains every alias we
# have observed in real Egyptian AutoCAD files. Order within the list
# does not matter — matching tries ALL aliases and picks the best score.
# =====================================================================
LAYER_TAXONOMY = {
    "wall": [
        "wall", "walls", "wal", "wll", "wl",
        "a-wall", "a-wall-ext", "a-wall-int",
        "arc-wall", "arch-wall", "architectural-wall",
        "partition", "part", "masonry", "brick",
        "block", "blockwork", "brickwall", "brick-wall",
        "حائط", "جدار", "طوب",
    ],
    "wall_ext": [
        "ext-wall", "wall-ext", "external-wall", "outer-wall",
        "a-wall-ext", "wall-exterior", "ext_wall", "extwall",
        "externalwall", "outerwall", "ext-wall-250", "ext-wall-200",
        "external", "exterior",
        "حائط-خارجي", "جدار-خارجي",
    ],
    "wall_int": [
        "int-wall", "wall-int", "internal-wall", "inner-wall",
        "partition-wall", "a-wall-int", "wall-interior", "int_wall",
        "intwall", "internalwall", "innerwall", "int-wall-120",
        "int-wall-150", "internal", "interior", "partition",
        "حائط-داخلي", "جدار-داخلي",
    ],
    "column": [
        "column", "columns", "col", "cols",
        "s-col", "s-column", "str-col", "structural-column",
        "concrete-column", "r.c-col", "rc-col", "col-grid",
        "c", "c1", "c2", "c3",
        "عمود", "أعمدة",
    ],
    "beam": [
        "beam", "beams", "bm", "bms",
        "s-beam", "s-bm", "str-beam", "structural-beam",
        "girder", "girders", "r.c-beam", "rc-beam",
        "كمرة", "كمرات",
    ],
    "slab": [
        "slab", "slabs", "slb",
        "s-slab", "floor", "floor-slab", "susp-slab",
        "solid-slab", "deck", "suspended-slab",
        "بلاطة", "سقف",
    ],
    "footing": [
        "footing", "footings", "ftg",
        "fnd", "fnd-foot", "s-footing", "s-fnd",
        "raft", "isolated-footing", "pad-footing",
        "قاعدة", "أساس",
    ],
    "door": [
        "door", "doors", "dr", "drs",
        "a-door", "door-leaf", "gate", "gates",
        "opening-door", "single-door", "double-door",
        "باب", "أبواب",
    ],
    "window": [
        "window", "windows", "win", "wnd", "wndw",
        "glz", "glazing", "a-window", "a-glaz",
        "curtain-wall", "sliding-window", "fixed-window",
        "شباك", "نافذة", "نوافذ",
    ],
    "stair": [
        "stair", "stairs", "str", "staircase",
        "st", "stair-core", "stair-flight",
        "a-stair", "core-stair", "stairs-core",
        "سلم", "سلالم",
    ],
    "balcony": [
        "balcony", "balconies", "bal", "blcny",
        "terrace", "a-balcony", "loggia", "veranda",
        "بلكونة", "بلكونات", "شرفة",
    ],
    "void": [
        "void", "voids", "opening", "openings",
        "shaft", "manwar", "light-well", "lightwell",
        "well", "hole", "duct", "elevator-shaft",
        "منور", "فتحة", "بئر",
    ],
    "room": [
        "room", "rooms", "space", "space-name",
        "area", "area-label", "room-name", "room-tag",
        "zone", "region",
        "غرفة", "غرف",
    ],
    "hatch": [
        "hatch", "hatching", "solid", "fill", "poche",
    ],
    "dim": [
        "dim", "dims", "dimension", "dimensions",
        "anno-dim", "arc-dim", "a-anno-dim", "structural-dim",
        "أبعاد", "قياسات",
    ],
    "text": [
        "text", "txt", "mtext", "label", "labels",
        "anno-text", "a-anno-text", "note", "notes",
        "annotation", "a-anno",
        "نص", "ملاحظات",
    ],
    "furniture": [
        "furn", "furniture", "furnishing",
        "a-furn", "furniture-plan", "ff&e",
        "أثاث",
    ],
    "grid": [
        "grid", "axis", "axes", "grid-line",
        "s-grid", "column-grid", "grid-bubble",
        "محاور", "شبكة",
    ],
    "section": [
        "section", "sections", "sect",
        "sec-mark", "section-mark", "detail", "details",
        "قطع", "تفاصيل",
    ],
    "level": [
        "level", "levels", "elevation", "elev",
        "spot-level", "ffl", "ssl",
        "منسوب", "مناسيب",
    ],
    "beam_schedule": [
        "beam-sched", "beam-schedule", "beam-reinf",
        "b-sched", "b-schedule",
    ],
    "column_schedule": [
        "column-sched", "column-schedule", "col-sched",
        "col-schedule", "c-sched",
    ],
    "footing_schedule": [
        "footing-sched", "footing-schedule", "fnd-sched",
    ],
    "title_block": [
        "title", "title-block", "border", "frame",
        "sheet", "sheet-frame",
    ],
    "boundary": [
        "boundary", "plot", "plot-boundary", "plot-line",
        "site", "site-boundary", "property-line",
        "حد", "حدود",
    ],
}


# =====================================================================
# CATEGORY GROUPING
# ---------------------------------------------------------------------
# Some tasks only care whether the layer is a "primary architectural
# element" vs "annotation" vs "structural". This lookup tells the
# engine which categories belong to which group.
# =====================================================================
CATEGORY_GROUPS = {
    "architectural_primary": {
        "wall", "wall_ext", "wall_int",
        "column", "beam", "slab", "footing",
        "door", "window", "stair", "balcony", "void", "room",
    },
    "architectural_secondary": {
        "furniture", "level", "boundary",
    },
    "annotation": {
        "dim", "text", "section", "grid",
        "beam_schedule", "column_schedule", "footing_schedule",
        "title_block", "hatch",
    },
    "masonry_finish": {
        "wall", "wall_ext", "wall_int",
    },
    "opening": {
        "door", "window",
    },
    "wet_area_candidate": {
        "room",
    },
}


# =====================================================================
# NORMALISATION
# =====================================================================
def normalize_layer_name(name: str) -> str:
    """
    Lowercase, strip whitespace, collapse runs of separators (- _ / . space)
    into single hyphens. Removes leading/trailing hyphens.

    Examples:
        "A-WALL-EXT"        -> "a-wall-ext"
        "  Wall_Int  "      -> "wall-int"
        "WLL"               -> "wll"
        "wall.ext"          -> "wall-ext"
        "Wall  External"    -> "wall-external"
    """
    if not name:
        return ""
    s = str(name).strip().lower()
    # Collapse any run of separator characters into a single hyphen
    out = []
    prev_sep = False
    for ch in s:
        if ch in " \t-_.·/\\|":
            if not prev_sep and out:
                out.append("-")
                prev_sep = True
        else:
            out.append(ch)
            prev_sep = False
    # strip trailing separator
    while out and out[-1] == "-":
        out.pop()
    return "".join(out)


# =====================================================================
# LEVENSHTEIN DISTANCE
# ---------------------------------------------------------------------
# Hand-rolled (no external dep). Used for fuzzy layer matching.
# =====================================================================
def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein distance. Case-sensitive."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[j - 1] + 1,      # insertion
                prev[j] + 1,          # deletion
                prev[j - 1] + cost,   # substitution
            ))
        prev = curr
    return prev[-1]


# =====================================================================
# CLASSIFY LAYER
# ---------------------------------------------------------------------
# Priority order:
#   1. Exact match to a specific subtype (wall_ext, wall_int)
#   2. Exact match to a parent category (wall, door, etc.)
#   3. Substring match (layer contains the alias)
#   4. Levenshtein distance <= threshold
# Returns (category, subtype, confidence, matched_alias)
# =====================================================================
def _specific_subtypes_first():
    """
    Subtypes must be checked before their parents, otherwise
    'wall_ext' would match 'wall' first.
    """
    return ["wall_ext", "wall_int", "beam_schedule", "column_schedule",
            "footing_schedule", "title_block"]


def _score_exact(normalized_name, alias):
    return 100 if normalized_name == alias else 0


def _score_substring(normalized_name, alias):
    if not alias or not normalized_name:
        return 0
    if alias in normalized_name:
        # Prefer longer aliases → more specific match
        return 60 + min(len(alias), 20)
    return 0


def _score_fuzzy(normalized_name, alias, max_distance=2):
    if abs(len(normalized_name) - len(alias)) > max_distance:
        return 0
    d = levenshtein(normalized_name, alias)
    if d > max_distance:
        return 0
    # Higher score for smaller distance and shorter alias
    return 40 - d * 10 + min(len(alias), 10)


def classify_layer(layer_name: str):
    """
    Classify a layer name into a canonical category.

    Returns:
        {
            "category":   str | None,
            "subtype":    str | None,      # 'external' / 'internal' for walls
            "confidence": 'high' | 'medium' | 'low' | None,
            "matched":    str | None,      # the alias that matched
            "raw":        str,             # original layer name
            "normalized": str,
        }
    """
    raw = layer_name or ""
    norm = normalize_layer_name(raw)

    if not norm:
        return {
            "category": None, "subtype": None, "confidence": None,
            "matched": None, "raw": raw, "normalized": norm,
        }

    best = {
        "category": None, "subtype": None, "confidence": None,
        "matched": None, "score": 0,
    }

    # Pass 1: check subtypes first (specific before general)
    for cat in _specific_subtypes_first():
        for alias in LAYER_TAXONOMY.get(cat, []):
            a = normalize_layer_name(alias)
            score = _score_exact(norm, a)
            if score > best["score"]:
                subtype = None
                parent = cat
                if cat == "wall_ext":
                    subtype = "external"; parent = "wall"
                elif cat == "wall_int":
                    subtype = "internal"; parent = "wall"
                best = {
                    "category": parent, "subtype": subtype,
                    "confidence": "high", "matched": alias, "score": score,
                }

    # Pass 2: general categories
    for cat, aliases in LAYER_TAXONOMY.items():
        if cat in ("wall_ext", "wall_int"):
            continue  # already handled
        for alias in aliases:
            a = normalize_layer_name(alias)
            # Exact
            s = _score_exact(norm, a)
            if s > best["score"]:
                best = {
                    "category": cat, "subtype": None,
                    "confidence": "high", "matched": alias, "score": s,
                }
                continue
            # Substring
            s = _score_substring(norm, a)
            if s > best["score"]:
                best = {
                    "category": cat, "subtype": None,
                    "confidence": "medium", "matched": alias, "score": s,
                }
                continue
            # Fuzzy
            s = _score_fuzzy(norm, a, max_distance=2)
            if s > best["score"]:
                best = {
                    "category": cat, "subtype": None,
                    "confidence": "low", "matched": alias, "score": s,
                }

    return {
        "category":   best["category"],
        "subtype":    best["subtype"],
        "confidence": best["confidence"],
        "matched":    best["matched"],
        "raw":        raw,
        "normalized": norm,
    }


# =====================================================================
# CATEGORY GROUP HELPERS
# =====================================================================
def in_group(category: str, group: str) -> bool:
    """True if `category` belongs to the named group."""
    return category in CATEGORY_GROUPS.get(group, set())


def is_primary_architectural(category: str) -> bool:
    return in_group(category, "architectural_primary")


def is_annotation(category: str) -> bool:
    return in_group(category, "annotation")
