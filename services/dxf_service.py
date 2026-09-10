# services/dxf_service.py — Professional Architectural + Structural generator
import io
import math
import random
import numpy as np
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, box
from config import BOQ_RATES

_VALID_LW = [0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60,
             70, 80, 90, 100, 106, 120, 140, 158, 200, 211]

def _snap_lw(lw):
    try:
        lw = int(lw)
    except Exception:
        return 25
    return lw if lw in _VALID_LW else min(_VALID_LW, key=lambda v: abs(v - lw))

def _ascii(s):
    return (str(s).replace('²', '2').replace('³', '3').replace('Ø', 'D')
            .replace('·', '-').replace('×', 'x').replace('°', 'deg'))


# ======================================================================
# DXF reader helpers
# ======================================================================
def detect_dxf_layers(doc):
    layers = {}
    for entity in doc.modelspace():
        l = entity.dxf.layer
        layers.setdefault(l, {'count': 0, 'types': set(), 'keywords': []})
        layers[l]['count'] += 1
        layers[l]['types'].add(entity.dxftype())
    for layer in layers:
        lc = layer.lower()
        layers[layer]['keywords'] = [k for k in
            ['wall','column','beam','slab','footing','room','door','window','area'] if k in lc]
    return layers


def _polyline_is_closed(entity):
    """Check if a LWPOLYLINE or POLYLINE is effectively closed.
    Accepts: flag bit 1, entity.closed property, OR first≈last point."""
    try:
        flags = entity.dxf.get('flags', 0) if hasattr(entity.dxf, 'get') else entity.dxf.flags
        if flags & 1:
            return True
    except Exception:
        pass
    try:
        if entity.closed:
            return True
    except Exception:
        pass
    # Fallback: check if first ≈ last point (tolerance scales with size)
    try:
        if entity.dxftype() == 'LWPOLYLINE':
            pts = [(p[0], p[1]) for p in entity.get_points()]
        else:
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
        if len(pts) < 3:
            return False
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        span = max(abs(x0), abs(y0), abs(x1), abs(y1), 1.0)
        return math.hypot(x1 - x0, y1 - y0) < max(1.0, span * 0.001)
    except Exception:
        return False


def _extract_polyline_points(entity):
    if entity.dxftype() == 'LWPOLYLINE':
        return [(p[0], p[1]) for p in entity.get_points()]
    return [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]


def _shoelace_area(pts):
    if len(pts) < 3:
        return 0.0
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def extract_areas_from_dxf(doc, unit='mm', workflow='architectural'):
    """Robust area extractor. Accepts closed polylines (flag OR coincident
    endpoints) AND hatch boundary paths. Prints a diagnostic census."""
    scale = {'mm': 1e-6, 'cm': 1e-4, 'm': 1.0}.get(unit, 1e-6)
    results = []
    msp = doc.modelspace()

    # --- Diagnostics: entity type census ---
    counts = {}
    for e in msp:
        counts[e.dxftype()] = counts.get(e.dxftype(), 0) + 1
    print(f"[dxf-extract] entity census: {counts}")

    # --- Pre-collect text positions for label association ---
    text_positions = []
    try:
        for t in msp.query('TEXT MTEXT'):
            try:
                insert = t.dxf.insert
                if t.dxftype() == 'TEXT':
                    txt = (t.dxf.text or '').strip()
                else:
                    txt = (getattr(t, 'text', '') or '').strip()
                text_positions.append((float(insert.x), float(insert.y), txt))
            except Exception:
                pass
    except Exception:
        pass

    def find_label(cx, cy, radius):
        best, best_d = '', radius
        for tx, ty, txt in text_positions:
            d = math.hypot(tx - cx, ty - cy)
            if d < best_d and txt:
                best, best_d = txt, d
        return best

    # --- Polylines ---
    poly_closed = 0
    for entity in msp:
        if entity.dxftype() not in ('LWPOLYLINE', 'POLYLINE'):
            continue
        try:
            if not _polyline_is_closed(entity):
                continue
            poly_closed += 1
            pts = _extract_polyline_points(entity)
            if len(pts) < 3:
                continue
            a = _shoelace_area(pts)
            if a < 1e-6:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            minx = min(p[0] for p in pts); maxx = max(p[0] for p in pts)
            miny = min(p[1] for p in pts); maxy = max(p[1] for p in pts)
            radius = max(200.0, max(maxx - minx, maxy - miny) * 0.6)
            label = find_label(cx, cy, radius) or find_label(cx, cy, 500)
            results.append({
                'layer': entity.dxf.layer,
                'area_m2': round(a * scale, 4),
                'label': label,
                'vertices': len(pts),
                'entity_type': entity.dxftype(),
            })
        except Exception as e:
            print(f"[dxf-extract] polyline failed: {e!r}")
            continue

    # --- HATCH boundaries ---
    hatch_count = 0
    try:
        for hatch in msp.query('HATCH'):
            try:
                for path in hatch.paths:
                    pts = []
                    if hasattr(path, 'vertices') and path.vertices:
                        pts = [(float(v[0]), float(v[1])) for v in path.vertices]
                    if len(pts) < 3:
                        continue
                    a = _shoelace_area(pts)
                    if a < 1e-6:
                        continue
                    cx = sum(p[0] for p in pts) / len(pts)
                    cy = sum(p[1] for p in pts) / len(pts)
                    label = find_label(cx, cy, 500)
                    results.append({
                        'layer': hatch.dxf.layer,
                        'area_m2': round(a * scale, 4),
                        'label': label,
                        'vertices': len(pts),
                        'entity_type': 'HATCH',
                    })
                    hatch_count += 1
            except Exception as e:
                print(f"[dxf-extract] hatch failed: {e!r}")
                continue
    except Exception:
        pass

    print(f"[dxf-extract] closed polylines={poly_closed} hatch_paths={hatch_count} "
          f"→ {len(results)} areas returned")
    return results


def create_dxf_layer(doc, name, color, lineweight=25, linetype='CONTINUOUS'):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=linetype, lineweight=_snap_lw(lineweight))


# ======================================================================
# Drawing primitives
# ======================================================================
def _wall_rect(p1, p2, thickness):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1:
        return None
    nx, ny = -dy/L * thickness/2, dx/L * thickness/2
    return [(x1+nx, y1+ny), (x2+nx, y2+ny), (x2-nx, y2-ny),
            (x1-nx, y1-ny), (x1+nx, y1+ny)]


def _wall_rect_with_gap(p1, p2, thickness, gap_center, gap_width):
    """Wall rectangle broken around a gap (for doors/windows)."""
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1:
        return []
    ux, uy = dx/L, dy/L
    nx, ny = -uy * thickness/2, ux * thickness/2
    g0 = max(0, gap_center - gap_width/2)
    g1 = min(L, gap_center + gap_width/2)
    parts = []
    if g0 > 20:
        a = (x1, y1); b = (x1 + ux*g0, y1 + uy*g0)
        parts.append([(a[0]+nx, a[1]+ny), (b[0]+nx, b[1]+ny),
                      (b[0]-nx, b[1]-ny), (a[0]-nx, a[1]-ny), (a[0]+nx, a[1]+ny)])
    if g1 < L - 20:
        a = (x1 + ux*g1, y1 + uy*g1); b = (x2, y2)
        parts.append([(a[0]+nx, a[1]+ny), (b[0]+nx, b[1]+ny),
                      (b[0]-nx, b[1]-ny), (a[0]-nx, a[1]-ny), (a[0]+nx, a[1]+ny)])
    return parts


def draw_wall_seg(msp, p1, p2, t, layer='A-WALL-EXT', color=7, gaps=None):
    """Draw wall with optional list of (center_along, width) openings."""
    if not gaps:
        r = _wall_rect(p1, p2, t)
        if r:
            msp.add_lwpolyline(r, dxfattribs={'layer': layer, 'color': color})
        return
    x1, y1 = p1; x2, y2 = p2
    L = math.hypot(x2-x1, y2-y1)
    if L < 1:
        return
    # Sort gaps
    gaps_sorted = sorted(gaps, key=lambda g: g[0])
    cursor = 0.0
    for gc, gw in gaps_sorted:
        seg_start = cursor
        seg_end = max(cursor, gc - gw/2)
        if seg_end - seg_start > 20:
            a = (x1 + (x2-x1)*seg_start/L, y1 + (y2-y1)*seg_start/L)
            b = (x1 + (x2-x1)*seg_end/L, y1 + (y2-y1)*seg_end/L)
            r = _wall_rect(a, b, t)
            if r:
                msp.add_lwpolyline(r, dxfattribs={'layer': layer, 'color': color})
        cursor = max(cursor, gc + gw/2)
    if cursor < L - 20:
        a = (x1 + (x2-x1)*cursor/L, y1 + (y2-y1)*cursor/L)
        r = _wall_rect(a, (x2, y2), t)
        if r:
            msp.add_lwpolyline(r, dxfattribs={'layer': layer, 'color': color})


def draw_door(msp, cx, cy, width, wall_thickness, is_horizontal=True, flip=False):
    if is_horizontal:
        msp.add_line((cx-width/2, cy-wall_thickness/2), (cx-width/2, cy+wall_thickness/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        msp.add_line((cx+width/2, cy-wall_thickness/2), (cx+width/2, cy+wall_thickness/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        if flip:
            msp.add_line((cx-width/2, cy), (cx-width/2, cy-width),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx-width/2, cy), radius=width, start_angle=270, end_angle=360,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})
        else:
            msp.add_line((cx-width/2, cy), (cx-width/2, cy+width),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx-width/2, cy), radius=width, start_angle=0, end_angle=90,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})
    else:
        msp.add_line((cx-wall_thickness/2, cy-width/2), (cx+wall_thickness/2, cy-width/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        msp.add_line((cx-wall_thickness/2, cy+width/2), (cx+wall_thickness/2, cy+width/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        if flip:
            msp.add_line((cx, cy-width/2), (cx-width, cy-width/2),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx, cy-width/2), radius=width, start_angle=180, end_angle=270,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})
        else:
            msp.add_line((cx, cy-width/2), (cx+width, cy-width/2),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx, cy-width/2), radius=width, start_angle=0, end_angle=90,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})


def draw_window(msp, cx, cy, width, wall_thickness, is_horizontal=True):
    if is_horizontal:
        for off in (-wall_thickness/2, 0, wall_thickness/2):
            msp.add_line((cx-width/2, cy+off), (cx+width/2, cy+off),
                         dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        for ex in (-width/2, width/2):
            msp.add_line((cx+ex, cy-wall_thickness/2), (cx+ex, cy+wall_thickness/2),
                         dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        msp.add_line((cx, cy-wall_thickness/2), (cx, cy+wall_thickness/2),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})
    else:
        for off in (-wall_thickness/2, 0, wall_thickness/2):
            msp.add_line((cx+off, cy-width/2), (cx+off, cy+width/2),
                         dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        for ey in (-width/2, width/2):
            msp.add_line((cx-wall_thickness/2, cy+ey), (cx+wall_thickness/2, cy+ey),
                         dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        msp.add_line((cx-wall_thickness/2, cy), (cx+wall_thickness/2, cy),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})


def place_furniture(msp, x, y, w, h, room_type):
    c, col = 'A-FURN', 6
    if room_type in ('bedroom_master', 'bedroom'):
        bw, bh = w*0.55, h*0.65
        bx, by = x + (w-bw)/2, y + (h-bh)/2
        msp.add_lwpolyline([(bx,by),(bx+bw,by),(bx+bw,by+bh),(bx,by+bh)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
        msp.add_lwpolyline([(bx+50,by+bh-250),(bx+bw-50,by+bh-250),
                            (bx+bw-50,by+bh-50),(bx+50,by+bh-50)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
    elif room_type == 'living':
        msp.add_lwpolyline([(x+200,y+200),(x+200+w*0.35,y+200),
                            (x+200+w*0.35,y+200+h*0.3),(x+200,y+200+h*0.3)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
        msp.add_lwpolyline([(x+w*0.55,y+h*0.35),(x+w*0.85,y+h*0.35),
                            (x+w*0.85,y+h*0.65),(x+w*0.55,y+h*0.65)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
    elif room_type == 'kitchen':
        msp.add_lwpolyline([(x+100,y+100),(x+w-100,y+100),(x+w-100,y+700),(x+100,y+700)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
    elif room_type == 'bathroom':
        msp.add_circle((x+400,y+400), radius=200, dxfattribs={'layer': c, 'color': col})
        msp.add_lwpolyline([(x+w-600,y+100),(x+w-100,y+100),(x+w-100,y+500),(x+w-600,y+500)],
                           dxfattribs={'layer': c, 'color': col}, close=True)


def draw_label(msp, x, y, text, layer='A-ROOM-TEXT', height=180, color=4, align='C'):
    t = msp.add_text(_ascii(text), dxfattribs={'layer': layer, 'height': height, 'color': color})
    a = {'C': TextEntityAlignment.MIDDLE_CENTER,
         'L': TextEntityAlignment.LEFT,
         'R': TextEntityAlignment.RIGHT}[align]
    t.set_placement((x, y), align=a)


def draw_table(msp, x, y, col_widths, rows, row_h=350, header=True,
               layer='ANNO-TABLE', color=7):
    """Draw a rectangular table with text rows."""
    ncols = len(col_widths)
    total_w = sum(col_widths)
    # header row + data rows
    nrows = len(rows)
    total_h = nrows * row_h
    # Outer rectangle
    msp.add_lwpolyline([(x, y), (x+total_w, y), (x+total_w, y-total_h),
                        (x, y-total_h)], dxfattribs={'layer': layer, 'color': color}, close=True)
    # Column separators
    cx = x
    for cw in col_widths[:-1]:
        cx += cw
        msp.add_line((cx, y), (cx, y-total_h),
                     dxfattribs={'layer': layer, 'color': color})
    # Row separators
    for r in range(1, nrows):
        yy = y - r*row_h
        msp.add_line((x, yy), (x+total_w, yy),
                     dxfattribs={'layer': layer, 'color': color})
    # Text
    for r, row in enumerate(rows):
        yy = y - r*row_h - row_h/2
        cx = x
        for c, cell in enumerate(row):
            cx_cell = cx + col_widths[c]/2
            col = 7 if (r == 0 and header) else color
            draw_label(msp, cx_cell, yy, str(cell)[:28], layer, height=200,
                       color=col, align='C')
            cx += col_widths[c]


def draw_column_grid_bubbles(msp, x0, y0, L, W, xs, ys, col_size, layer='S-GRID'):
    """Draw grid lines with bubble labels."""
    # Vertical grid lines (numbered 1, 2, ...)
    for i, cx in enumerate(xs):
        msp.add_line((cx, y0 - 1500), (cx, y0 + W + 1500),
                     dxfattribs={'layer': layer, 'color': 2, 'linetype': 'DASHED'})
        # Bubble at bottom
        msp.add_circle((cx, y0 - 1800), radius=400, dxfattribs={'layer': layer, 'color': 2})
        draw_label(msp, cx, y0 - 1800, str(i+1), layer, 250, 2)
    # Horizontal grid lines (lettered A, B, ...)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    for j, cy in enumerate(ys):
        msp.add_line((x0 - 1500, cy), (x0 + L + 1500, cy),
                     dxfattribs={'layer': layer, 'color': 2, 'linetype': 'DASHED'})
        msp.add_circle((x0 - 1800, cy), radius=400, dxfattribs={'layer': layer, 'color': 2})
        draw_label(msp, x0 - 1800, cy, letters[j % len(letters)], layer, 250, 2)


# ======================================================================
# Layout engine — snap cuts to column grid lines
# ======================================================================
def _partition_rect_grid(rect, rooms, grid_xs, grid_ys):
    if not rooms:
        return []
    if len(rooms) == 1:
        return [(rooms[0], rect)]

    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    total = sum(r.get('area_m2', 10) for r in rooms) or 1
    frac = max(0.22, min(0.78, rooms[0].get('area_m2', 10) / total))

    if w >= h:
        ideal = x0 + w * frac
        # candidate grid x's that are > 3m from either end
        candidates = [g for g in grid_xs if x0 + 3000 < g < x1 - 3000]
        cut = min(candidates, key=lambda g: abs(g - ideal)) if candidates else ideal
        return ([(rooms[0], (x0, y0, cut, y1))] +
                _partition_rect_grid((cut, y0, x1, y1), rooms[1:], grid_xs, grid_ys))
    else:
        ideal = y0 + h * frac
        candidates = [g for g in grid_ys if y0 + 3000 < g < y1 - 3000]
        cut = min(candidates, key=lambda g: abs(g - ideal)) if candidates else ideal
        return ([(rooms[0], (x0, y0, x1, cut))] +
                _partition_rect_grid((x0, cut, x1, y1), rooms[1:], grid_xs, grid_ys))


# ======================================================================
# Free span detection for openings
# ======================================================================
def _columns_along_wall(p1, p2, columns, tolerance=200):
    """Return list of distances t (0..L) where a column lies on the wall."""
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1:
        return []
    ux, uy = dx/L, dy/L
    hits = []
    for cx, cy in columns:
        # Project column onto wall line
        t = (cx - x1) * ux + (cy - y1) * uy
        if t < -50 or t > L + 50:
            continue
        proj_x, proj_y = x1 + ux*t, y1 + uy*t
        if math.hypot(cx - proj_x, cy - proj_y) < tolerance:
            hits.append(t)
    return hits


def _free_spans(L, col_positions, col_block=400, clearance=250):
    """Given wall length L and column positions, return list of (t0, t1) free spans."""
    blocks = []
    for t in col_positions:
        blocks.append((t - col_block/2 - clearance, t + col_block/2 + clearance))
    blocks.sort()
    merged = []
    for b in blocks:
        if merged and b[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b[1]))
        else:
            merged.append(list(b))
    spans = []
    cursor = 0.0
    for b in merged:
        if b[0] > cursor + 100:
            spans.append((cursor, b[0]))
        cursor = max(cursor, b[1])
    if cursor < L - 100:
        spans.append((cursor, L))
    return spans


def best_opening_position(p1, p2, columns, opening_width, col_size=300):
    """Return distance t along wall for the best opening center, or None."""
    x1, y1 = p1; x2, y2 = p2
    L = math.hypot(x2-x1, y2-y1)
    if L < opening_width + 400:
        return None
    col_positions = _columns_along_wall(p1, p2, columns)
    spans = _free_spans(L, col_positions, col_block=col_size)
    # Pick longest span that fits
    valid = [(a, b) for a, b in spans if b - a >= opening_width + 200]
    if not valid:
        return None
    longest = max(valid, key=lambda s: s[1] - s[0])
    return (longest[0] + longest[1]) / 2


# ======================================================================
# MAIN GENERATOR
# ======================================================================
def build_complete_project(params):
        seed = params.get('variation_seed') or hash((
        params.get('plot_area_m2', 200),
        params.get('street_width_m', 10),
        params.get('num_floors', 2),
        params.get('num_bedrooms', 3),
    ))
    random.seed(seed)

    # ----- 1. Plot -----
    if params.get('plot_polygon') is not None:
        coords = params['plot_polygon']
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
        plot_poly = poly
    else:
        area = params['plot_area_m2']
        pw = params.get('plot_width') or math.sqrt(area / 1.4)
        pl = params.get('plot_length') or (area / pw)
        plot_poly = box(0, 0, pl, pw)

    # ----- 2. Egyptian Law 119/2008 -----
    sw = params.get('street_width_m', 10)
    if sw >= 12:   front_sb, max_floors = 3.0, 4
    elif sw >= 8:  front_sb, max_floors = 2.5, 3
    elif sw >= 6:  front_sb, max_floors = 2.0, 3
    else:          front_sb, max_floors = 1.5, 2
    rear_sb = 2.0
    side_sb = 1.5

    plot_area = plot_poly.area
    coverage = 0.65 if plot_area < 200 else (0.60 if plot_area < 500 else 0.50)

    min_x, min_y, max_x, max_y = plot_poly.bounds
    avail_w = max_x - min_x - 2 * side_sb
    avail_l = max_y - min_y - front_sb - rear_sb
    max_footprint = plot_area * coverage
    if avail_w * avail_l > max_footprint:
        s = math.sqrt(max_footprint / (avail_w * avail_l))
        avail_w *= s; avail_l *= s

    building_width = max(8.0, avail_w)
    building_length = max(10.0, avail_l)
    num_floors = min(params.get('num_floors', max_floors), max_floors)
    floor_h_m = params.get('floor_height_m', 3.0)

    # ----- 3. Room program -----
    layout_plan = params.get('layout_plan') or {}
    ai_rooms = layout_plan.get('rooms', [])
    nb = params.get('num_bedrooms', 3)
    nba = params.get('num_bathrooms', 2)

    if not ai_rooms:
        ai_rooms = [
            {"name": "Living Room", "type": "living", "area_m2": 26, "priority": 1, "zone": "public", "needs_window": True},
            {"name": "Kitchen", "type": "kitchen", "area_m2": 10, "priority": 2, "zone": "public", "needs_window": True},
            {"name": "Dining", "type": "dining", "area_m2": 12, "priority": 3, "zone": "public", "needs_window": True},
        ]
        for i in range(nb):
            nm = "Master Bedroom" if i == 0 else f"Bedroom {i+1}"
            rt = "bedroom_master" if i == 0 else "bedroom"
            ai_rooms.append({"name": nm, "type": rt, "area_m2": 16 if i==0 else 12,
                             "priority": 4+i, "zone": "private", "needs_window": True})
        for i in range(nba):
            ai_rooms.append({"name": "Bathroom" if i==0 else f"Bathroom {i+1}",
                             "type": "bathroom", "area_m2": 4.5,
                             "priority": 10+i, "zone": "private", "needs_window": False})

    # ----- 3b. Detect pre-positioned rooms -----
    # `_position_rooms` (in ui/pages.py) returns rooms with x/y/w/h but no
    # `zone` key. `_partition_rect_grid` below expects a raw room program
    # with `zone`. When the caller passes pre-positioned rooms, use them
    # directly instead of partitioning again.
    _pre_positioned = bool(ai_rooms) and all(
        ('x' in r and 'y' in r and 'w' in r and 'h' in r) for r in ai_rooms
    )

    # ----- 4. Sizes in mm -----
    L = building_length * 1000
    W = building_width * 1000
    x0, y0 = 0.0, 0.0
    x1, y1 = L, W

    # ----- 5. Structural grid FIRST (so rooms snap to it) -----
    span_max = 5000
    nx = max(3, math.ceil(L / span_max) + 1)
    ny = max(2, math.ceil(W / span_max) + 1)
    xs = np.linspace(x0, x1, nx).tolist()
    ys = np.linspace(y0, y1, ny).tolist()
    cols = [(cx, cy) for cx in xs for cy in ys]
    col_size = 300 if num_floors <= 2 else 350
    beam_b, beam_d = 250, 600
    slab_t = 140
    foot = 1200 if num_floors <= 2 else 1500

    # ----- 6. DXF setup -----
    doc = ezdxf.new(dxfversion='R2000', setup=True)
    msp = doc.modelspace()
    doc.header['$INSUNITS'] = 4

    layers_def = {
        'A-WALL-EXT': {'color': 7, 'lineweight': 50},
        'A-WALL-INT': {'color': 8, 'lineweight': 25},
        'A-DOOR':     {'color': 3, 'lineweight': 18},
        'A-WINDOW':   {'color': 5, 'lineweight': 13},
        'A-FURN':     {'color': 6, 'lineweight': 9},
        'A-ROOM-TEXT':{'color': 4, 'lineweight': 13},
        'A-CORE':     {'color': 4, 'lineweight': 30},
        'S-COLUMN':   {'color': 1, 'lineweight': 50},
        'S-BEAM':     {'color': 5, 'lineweight': 25},
        'S-FOOTING':  {'color': 9, 'lineweight': 40},
        'S-GRID':     {'color': 2, 'lineweight': 13},
        'ANNO-DIM':   {'color': 2, 'lineweight': 13},
        'ANNO-TEXT':  {'color': 7, 'lineweight': 13},
        'ANNO-TITLE': {'color': 7, 'lineweight': 25},
        'ANNO-TABLE': {'color': 7, 'lineweight': 18},
    }
    for n, p in layers_def.items():
        create_dxf_layer(doc, n, p['color'], lineweight=p['lineweight'])

    # ==================================================================
    # SHEET 1 — ARCHITECTURAL PLAN (at origin)
    # ==================================================================
    wall_ext_t = 250
    int_t = 150
    corridor_h = 1300
    mid_y = (y0 + y1) / 2

    public_zone = (x0, y0, x1, mid_y - corridor_h/2)
    private_zone = (x0, mid_y + corridor_h/2, x1, y1)

    if _pre_positioned:
        # Caller already positioned the rooms — use them as-is.
        placements = []
        for r in ai_rooms:
            placements.append((
                {
                    'name': r.get('name', 'Room'),
                    'type': r.get('type', 'bedroom'),
                    'needs_window': bool(r.get('window_walls')),
                },
                (float(r['x']), float(r['y']),
                 float(r['x']) + float(r['w']),
                 float(r['y']) + float(r['h'])),
            ))
    else:
        public_rooms = sorted([r for r in ai_rooms if r.get('zone') == 'public'],
                              key=lambda r: r.get('priority', 99))
        private_rooms = sorted([r for r in ai_rooms if r.get('zone') == 'private'],
                               key=lambda r: r.get('priority', 99))
        if random.random() > 0.5:
            private_rooms = list(reversed(private_rooms))

        pub_pl = _partition_rect_grid(public_zone, public_rooms, xs, ys) if public_rooms else []
        priv_pl = _partition_rect_grid(private_zone, private_rooms, xs, ys) if private_rooms else []
        placements = pub_pl + priv_pl

    # Outer walls with openings
    # Collect exterior openings first
    ext_openings = {  # side: list of (center_x or y, width)
        'bottom': [], 'top': [], 'left': [], 'right': []
    }
    door_marks = []  # (mark, type, w, h, qty)
    win_marks = []
    d_idx = 0; w_idx = 0
    door_specs = {}  # for schedule
    win_specs = {}

    for room, (rx0, ry0, rx1, ry1) in placements:
        if not room.get('needs_window'):
            continue
        # window on exterior side
        if abs(ry0 - y0) < wall_ext_t:
            t = best_opening_position((x0, y0), (x1, y0), cols, 1400, col_size)
            # Instead of scanning whole wall, place per-room window
            wx = (rx0 + rx1) / 2
            ext_openings['bottom'].append((wx, 1400))
        if abs(ry1 - y1) < wall_ext_t:
            wx = (rx0 + rx1) / 2
            ext_openings['top'].append((wx, 1400))
        if abs(rx0 - x0) < wall_ext_t:
            wy = (ry0 + ry1) / 2
            ext_openings['left'].append((wy, 1200))
        if abs(rx1 - x1) < wall_ext_t:
            wy = (ry0 + ry1) / 2
            ext_openings['right'].append((wy, 1200))

    # Draw external walls (bottom-up, right-up, top-back, left-down)
    draw_wall_seg(msp, (x0, y0), (x1, y0), wall_ext_t, 'A-WALL-EXT', 7,
                  gaps=ext_openings['bottom'])
    draw_wall_seg(msp, (x1, y0), (x1, y1), wall_ext_t, 'A-WALL-EXT', 7,
                  gaps=ext_openings['right'])
    draw_wall_seg(msp, (x1, y1), (x0, y1), wall_ext_t, 'A-WALL-EXT', 7,
                  gaps=ext_openings['top'])
    draw_wall_seg(msp, (x0, y1), (x0, y0), wall_ext_t, 'A-WALL-EXT', 7,
                  gaps=ext_openings['left'])

    # Draw windows
    for wx, ww in ext_openings['bottom']:
        draw_window(msp, wx, y0, ww, wall_ext_t, is_horizontal=True)
    for wx, ww in ext_openings['top']:
        draw_window(msp, wx, y1, ww, wall_ext_t, is_horizontal=True)
    for wy, ww in ext_openings['left']:
        draw_window(msp, x0, wy, ww, wall_ext_t, is_horizontal=False)
    for wy, ww in ext_openings['right']:
        draw_window(msp, x1, wy, ww, wall_ext_t, is_horizontal=False)

    # Interior walls — collect door gaps first
    door_gaps_per_seg = []
    for room, (rx0, ry0, rx1, ry1) in placements:
        segs = []
        if rx0 > x0 + wall_ext_t:
            segs.append(((rx0, ry0), (rx0, ry1), 'V'))
        if rx1 < x1 - wall_ext_t:
            segs.append(((rx1, ry0), (rx1, ry1), 'V'))
        if ry0 > y0 + wall_ext_t:
            segs.append(((rx0, ry0), (rx1, ry0), 'H'))
        if ry1 < y1 - wall_ext_t:
            segs.append(((rx0, ry1), (rx1, ry1), 'H'))

        # Door on the wall facing corridor
        room_cy = (ry0 + ry1) / 2
        if room_cy < mid_y:
            wall_p1, wall_p2 = (rx0, ry1), (rx1, ry1)
            door_flip = False
        else:
            wall_p1, wall_p2 = (rx0, ry0), (rx1, ry0)
            door_flip = True

        t_best = best_opening_position(wall_p1, wall_p2, cols, 900, col_size)
        if t_best is not None:
            wall_L = math.hypot(wall_p2[0]-wall_p1[0], wall_p2[1]-wall_p1[1])
            ux = (wall_p2[0]-wall_p1[0]) / wall_L
            uy = (wall_p2[1]-wall_p1[1]) / wall_L
            door_cx = wall_p1[0] + ux*t_best
            door_cy = wall_p1[1] + uy*t_best
            draw_door(msp, door_cx, door_cy, 900, int_t, is_horizontal=True, flip=door_flip)
            d_idx += 1
            mark = f"D{d_idx}"
            door_marks.append((mark, "Single Leaf", 900, 2100, 1, room['name']))

    # Now draw interior walls as full rectangles (no gaps — doors already drawn)
    wall_segments = []
    for room, (rx0, ry0, rx1, ry1) in placements:
        if rx0 > x0 + wall_ext_t:
            wall_segments.append(((rx0, ry0), (rx0, ry1)))
        if rx1 < x1 - wall_ext_t:
            wall_segments.append(((rx1, ry0), (rx1, ry1)))
        if ry0 > y0 + wall_ext_t:
            wall_segments.append(((rx0, ry0), (rx1, ry0)))
        if ry1 < y1 - wall_ext_t:
            wall_segments.append(((rx0, ry1), (rx1, ry1)))
    # De-duplicate walls that are shared
    seen = set()
    for p1, p2 in wall_segments:
        key = tuple(sorted([(round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1]))]))
        if key in seen:
            continue
        seen.add(key)
        r = _wall_rect(p1, p2, int_t)
        if r:
            msp.add_lwpolyline(r, dxfattribs={'layer': 'A-WALL-INT', 'color': 8})

    # Furniture + room labels
    room_schedule = []
    for i, (room, (rx0, ry0, rx1, ry1)) in enumerate(placements):
        w = rx1 - rx0
        h = ry1 - ry0
        area_m2 = (w * h) / 1_000_000
        place_furniture(msp, rx0, ry0, w, h, room.get('type', 'bedroom'))
        draw_label(msp, (rx0+rx1)/2, (ry0+ry1)/2 + 100,
                   room['name'], 'A-ROOM-TEXT', 180, 4)
        draw_label(msp, (rx0+rx1)/2, (ry0+ry1)/2 - 200,
                   f"{area_m2:.1f} m2", 'A-ROOM-TEXT', 150, 4)
        perimeter = 2*((rx1-rx0)+(ry1-ry0)) / 1000
        room_schedule.append((i+1, room['name'], "Ground", f"{area_m2:.1f}",
                              f"{perimeter:.1f}", "Tiles", "Paint"))

    # Core (stairs)
    core_w, core_d = 2400, 3600
    core_x = x1 - wall_ext_t - core_w - 400
    core_y = mid_y - core_d/2
    msp.add_lwpolyline([(core_x, core_y), (core_x+core_w, core_y),
                        (core_x+core_w, core_y+core_d), (core_x, core_y+core_d)],
                       dxfattribs={'layer': 'A-CORE', 'color': 4}, close=True)
    for i in range(1, 12):
        ty = core_y + (core_d/12) * i
        msp.add_line((core_x, ty), (core_x+core_w, ty),
                     dxfattribs={'layer': 'A-CORE', 'color': 4})
    draw_label(msp, core_x + core_w/2, core_y - 300, "STAIR", 'A-ROOM-TEXT', 180, 4)

    # Column grid bubbles for architectural plan
    draw_column_grid_bubbles(msp, x0, y0, L, W, xs, ys, col_size)

    # Dimensions
    off = 700
    msp.add_line((x0, y0-off-2000), (x1, y0-off-2000), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for px in [x0, x1]:
        msp.add_line((px, y0-off-2100), (px, y0-off-1900), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    draw_label(msp, (x0+x1)/2, y0-off-2500, f"{L/1000:.2f} m", 'ANNO-TEXT', 220, 2)
    msp.add_line((x0-off-2000, y0), (x0-off-2000, y1), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for py in [y0, y1]:
        msp.add_line((x0-off-2100, py), (x0-off-1900, py), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    draw_label(msp, x0-off-2600, (y0+y1)/2, f"{W/1000:.2f} m", 'ANNO-TEXT', 220, 2)

    # Title above arch plan
    draw_label(msp, x0 + L/2, y1 + 3000, "GROUND FLOOR PLAN  —  SCALE 1:100",
               'ANNO-TITLE', 350, 7)
    # --- Enhanced architectural sheet: dimension chains ---
    off_chain = off + 3500
    _draw_dimension_chain(msp, x0, y0, xs, -off_chain, label=f"Overall {L/1000:.2f} m")
    _draw_dimension_chain(msp, x0, y0, xs, -(off_chain + 900))
    # Vertical chain
    y_dim_x = x0 - off_chain
    msp.add_line((y_dim_x, y0), (y_dim_x, y1),
                 dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for yy in ys:
        msp.add_line((y_dim_x - 100, yy), (y_dim_x + 100, yy),
                     dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for i in range(len(ys) - 1):
        mid = (ys[i] + ys[i + 1]) / 2
        seg = ys[i + 1] - ys[i]
        draw_label(msp, y_dim_x - 400, mid, f"{seg:.0f}", 'ANNO-DIM', 180, 2)
    # Section markers
    _draw_section_marker(msp, x0 + L / 2, y1 + 500, "A-A", 'up')
    _draw_section_marker(msp, x0 + L / 2, y0 - 500, "B-B", 'down')
    # North arrow (top-right of arch plan)
    _draw_north_arrow(msp, x0 + L + 2500, y1 - 1000, size=1400)

    # ==================================================================
    # SHEET 2 — STRUCTURAL PLAN (below)
    # ==================================================================
    sy = -40000  # Offset down
    sx = 0

    # Columns with labels
    col_labels = []
    idx = 0
    for cy in ys:
        for cx in xs:
            idx += 1
            label = f"C{idx}"
            col_labels.append((label, cx, cy))
            msp.add_lwpolyline(
                [(sx+cx-col_size/2, sy+cy-col_size/2),
                 (sx+cx+col_size/2, sy+cy-col_size/2),
                 (sx+cx+col_size/2, sy+cy+col_size/2),
                 (sx+cx-col_size/2, sy+cy+col_size/2)],
                dxfattribs={'layer': 'S-COLUMN', 'color': 1}, close=True)
            draw_label(msp, sx+cx, sy+cy, label, 'S-COLUMN', 160, 1)

    # Grid lines & bubbles
    draw_column_grid_bubbles(msp, sx+x0, sy+y0, L, W, xs, ys, col_size)

    # Beams (double lines)
    beam_labels = []
    bidx = 0
    for j, cy in enumerate(ys):
        for i in range(len(xs)-1):
            bidx += 1
            p1 = (sx+xs[i], sy+cy)
            p2 = (sx+xs[i+1], sy+cy)
            # double line
            for off_y in (-beam_b/2, beam_b/2):
                msp.add_line((p1[0], p1[1]+off_y), (p2[0], p2[1]+off_y),
                             dxfattribs={'layer': 'S-BEAM', 'color': 5})
            beam_labels.append((f"B{bidx}", (p1[0]+p2[0])/2, cy+sy))
    for i, cx in enumerate(xs):
        for j in range(len(ys)-1):
            bidx += 1
            p1 = (sx+cx, sy+ys[j])
            p2 = (sx+cx, sy+ys[j+1])
            for off_x in (-beam_b/2, beam_b/2):
                msp.add_line((p1[0]+off_x, p1[1]), (p2[0]+off_x, p2[1]),
                             dxfattribs={'layer': 'S-BEAM', 'color': 5})
            beam_labels.append((f"B{bidx}", cx+sx, (p1[1]+p2[1])/2))

    # Beam labels (small)
    for label, lx, ly in beam_labels[:40]:
        draw_label(msp, lx, ly, label, 'S-BEAM', 120, 5)

    # Footings (dashed rectangles + labels)
    fidx = 0
    foot_labels = []
    for cy in ys:
        for cx in xs:
            fidx += 1
            label = f"F{fidx}"
            foot_labels.append((label, cx, cy))
            msp.add_lwpolyline(
                [(sx+cx-foot/2, sy+cy-foot/2), (sx+cx+foot/2, sy+cy-foot/2),
                 (sx+cx+foot/2, sy+cy+foot/2), (sx+cx-foot/2, sy+cy+foot/2)],
                dxfattribs={'layer': 'S-FOOTING', 'color': 9, 'linetype': 'DASHED'}, close=True)
            draw_label(msp, sx+cx+foot/2+400, sy+cy, label, 'S-FOOTING', 140, 9)

    # Dimensions on structural plan
    msp.add_line((sx+x0, sy+y0-off-2000), (sx+x1, sy+y0-off-2000),
                 dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for px in [x0, x1]:
        msp.add_line((sx+px, sy+y0-off-2100), (sx+px, sy+y0-off-1900),
                     dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    draw_label(msp, sx+(x0+x1)/2, sy+y0-off-2500, f"{L/1000:.2f} m",
               'ANNO-TEXT', 220, 2)
    msp.add_line((sx+x0-off-2000, sy+y0), (sx+x0-off-2000, sy+y1),
                 dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    draw_label(msp, sx+x0-off-2600, sy+(y0+y1)/2, f"{W/1000:.2f} m",
               'ANNO-TEXT', 220, 2)

    # Structural title
    draw_label(msp, sx + x0 + L/2, sy + y1 + 3000,
               "FOUNDATION & ROOF FRAMING PLAN  —  SCALE 1:100",
               'ANNO-TITLE', 350, 7)

    # ==================================================================
    # SHEET 3 — SCHEDULES (right side of arch plan)
    # ==================================================================
    tb_x = x0 + L + 6000
    tb_y = y1

    draw_label(msp, tb_x + 4000, tb_y + 500, "ROOM SCHEDULE",
               'ANNO-TITLE', 300, 7)
    room_rows = [("No.", "Room Name", "Floor", "Area m2", "Perim", "Floor Fin.", "Wall Fin.")]
    for r in room_schedule:
        room_rows.append(tuple(str(x) for x in r))
    draw_table(msp, tb_x, tb_y, [600, 1800, 900, 900, 800, 1200, 1200], room_rows)

    # Door schedule
    door_y = tb_y - (len(room_rows) + 2) * 350 - 500
    draw_label(msp, tb_x + 3000, door_y + 500, "DOOR SCHEDULE",
               'ANNO-TITLE', 300, 7)
    door_rows = [("Mark", "Type", "W (mm)", "H (mm)", "Qty", "Location")]
    for r in door_marks:
        door_rows.append(tuple(str(x) for x in r))
    draw_table(msp, tb_x, door_y, [700, 1500, 800, 800, 500, 2000], door_rows)

    # Window schedule
    win_y = door_y - (len(door_rows) + 2) * 350 - 500
    draw_label(msp, tb_x + 3000, win_y + 500, "WINDOW SCHEDULE",
               'ANNO-TITLE', 300, 7)
    win_rows = [("Mark", "Type", "W (mm)", "H (mm)", "Qty", "Location")]
    for i, r in enumerate(placements):
        room, _ = r
        if room.get('needs_window'):
            win_rows.append((f"W{i+1}", "Sliding Alum.", "1400", "1200", "1", room['name']))
    draw_table(msp, tb_x, win_y, [700, 1500, 800, 800, 500, 2000], win_rows)

def _draw_dimension_chain(msp, x0, y_ref, grid_positions, offset_mm, label=None,
                           layer='ANNO-DIM', color=2):
    """Draw a chain of dimension segments between grid_positions, offset from y_ref."""
    if len(grid_positions) < 2:
        return
    y_dim = y_ref + offset_mm
    # Main dimension line
    msp.add_line((grid_positions[0], y_dim), (grid_positions[-1], y_dim),
                 dxfattribs={'layer': layer, 'color': color})
    # Ticks + segment labels
    for i, x in enumerate(grid_positions):
        msp.add_line((x, y_dim - 100), (x, y_dim + 100),
                     dxfattribs={'layer': layer, 'color': color})
        if i < len(grid_positions) - 1:
            seg_mm = grid_positions[i + 1] - grid_positions[i]
            mid = (grid_positions[i] + grid_positions[i + 1]) / 2
            draw_label(msp, mid, y_dim + 220, f"{seg_mm:.0f}", layer, height=180, color=color)
    # Extension lines back to the drawing
    for x in (grid_positions[0], grid_positions[-1]):
        msp.add_line((x, y_ref), (x, y_dim + 200),
                     dxfattribs={'layer': layer, 'color': color})
    if label:
        mid = (grid_positions[0] + grid_positions[-1]) / 2
        draw_label(msp, mid, y_dim + 700, label, layer, height=220, color=color)


def _draw_north_arrow(msp, x, y, size=1200, layer='ANNO-SYMBOL'):
    """North arrow: circle, filled triangle, 'N' label."""
    msp.add_circle((x, y), radius=size / 2,
                   dxfattribs={'layer': layer, 'color': 7})
    # Triangle pointing up
    half = size * 0.22
    msp.add_lwpolyline([(x, y + half * 1.4), (x - half, y - half * 0.6),
                        (x + half, y - half * 0.6)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    draw_label(msp, x, y + size * 0.75, "N", layer, height=320, color=7)


def _draw_section_marker(msp, x, y, tag, direction='down', layer='ANNO-SECTION'):
    """Section cut marker: circle with tag + a short cut line."""
    arrow_len = 1200
    if direction == 'down':
        msp.add_line((x, y), (x, y - arrow_len),
                     dxfattribs={'layer': layer, 'color': 1})
        msp.add_lwpolyline([(x, y), (x - 200, y - 400), (x + 200, y - 400)],
                           dxfattribs={'layer': layer, 'color': 1}, close=True)
    else:
        msp.add_line((x, y), (x, y + arrow_len),
                     dxfattribs={'layer': layer, 'color': 1})
        msp.add_lwpolyline([(x, y), (x - 200, y + 400), (x + 200, y + 400)],
                           dxfattribs={'layer': layer, 'color': 1}, close=True)
    # Tag bubble
    by = y - arrow_len - 400 if direction == 'down' else y + arrow_len + 400
    msp.add_circle((x, by), radius=500, dxfattribs={'layer': layer, 'color': 1})
    draw_label(msp, x, by, tag, layer, height=280, color=1)


def _draw_column_section_detail(msp, x, y, col_w, col_h, n_bars, bar_dia,
                                 stirrup_dia, cover=40, layer='S-DETAIL'):
    """Draw a typical column cross-section detail at (x,y) bottom-left."""
    msp.add_lwpolyline([(x, y), (x + col_w, y), (x + col_w, y + col_h),
                        (x, y + col_h)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    # Stirrup (inset by cover)
    c = cover
    msp.add_lwpolyline([(x + c, y + c), (x + col_w - c, y + c),
                        (x + col_w - c, y + col_h - c), (x + c, y + col_h - c)],
                       dxfattribs={'layer': layer, 'color': 1}, close=True)
    # Longitudinal bars at corners + midpoints
    bar_r = max(60, bar_dia * 1.5)
    positions = [
        (x + c, y + c), (x + col_w - c, y + c),
        (x + col_w - c, y + col_h - c), (x + c, y + col_h - c),
        (x + col_w / 2, y + c), (x + col_w / 2, y + col_h - c),
        (x + c, y + col_h / 2), (x + col_w - c, y + col_h / 2),
    ]
    for bx, by in positions[:n_bars]:
        msp.add_circle((bx, by), radius=bar_r,
                       dxfattribs={'layer': layer, 'color': 1})


def _draw_beam_section_detail(msp, x, y, beam_w, beam_d, top_bars, bot_bars,
                               stirrup_dia, layer='S-DETAIL'):
    msp.add_lwpolyline([(x, y), (x + beam_w, y), (x + beam_w, y + beam_d),
                        (x, y + beam_d)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    c = 40
    msp.add_lwpolyline([(x + c, y + c), (x + beam_w - c, y + c),
                        (x + beam_w - c, y + beam_d - c), (x + c, y + beam_d - c)],
                       dxfattribs={'layer': layer, 'color': 1}, close=True)
    bar_r = 80
    # Bottom bars (evenly spaced)
    for i in range(bot_bars):
        bx = x + c + (beam_w - 2 * c) * (i + 1) / (bot_bars + 1)
        msp.add_circle((bx, y + c), radius=bar_r,
                       dxfattribs={'layer': layer, 'color': 1})
    # Top bars
    for i in range(top_bars):
        bx = x + c + (beam_w - 2 * c) * (i + 1) / (top_bars + 1)
        msp.add_circle((bx, y + beam_d - c), radius=bar_r,
                       dxfattribs={'layer': layer, 'color': 1})


def _draw_footing_section_detail(msp, x, y, foot_w, foot_h, col_w,
                                  n_bars_bot, layer='S-DETAIL'):
    msp.add_lwpolyline([(x, y), (x + foot_w, y), (x + foot_w, y + foot_h),
                        (x, y + foot_h)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    # Column stub on top
    col_x = x + (foot_w - col_w) / 2
    msp.add_lwpolyline([(col_x, y + foot_h), (col_x + col_w, y + foot_h),
                        (col_x + col_w, y + foot_h + col_w * 1.5),
                        (col_x, y + foot_h + col_w * 1.5)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    # Bottom reinforcement
    bar_r = 80
    c = 100
    for i in range(n_bars_bot):
        bx = x + c + (foot_w - 2 * c) * (i + 1) / (n_bars_bot + 1)
        msp.add_circle((bx, y + c), radius=bar_r,
                       dxfattribs={'layer': layer, 'color': 1})


def _draw_sheet_border(msp, x, y, w, h, sheet_title, sheet_code,
                        layer='ANNO-BORDER'):
    """Outer border + inner frame + title strip at bottom-right."""
    msp.add_lwpolyline([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    margin = 1000
    msp.add_lwpolyline([(x + margin, y + margin), (x + w - margin, y + margin),
                        (x + w - margin, y + h - margin), (x + margin, y + h - margin)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    draw_label(msp, x + w / 2, y + h - margin - 800,
               sheet_title, layer, height=450, color=7)
    draw_label(msp, x + w - margin - 3000, y + margin + 400,
               f"SHEET: {sheet_code}", layer, height=300, color=7)

    # ==================================================================
    # SHEET 4 — BOQ + NOTES (right side of struct plan)
    # ==================================================================
    bx = sx + x0 + L + 6000
    by = sy + y1

    # --- BOQ ---
    draw_label(msp, bx + 5000, by + 500, "BILL OF QUANTITIES (EGP)",
               'ANNO-TITLE', 300, 7)
    # Compute BOQ quantities
    num_cols = len(cols)
    slab_vol = (L/1000)*(W/1000)*(slab_t/1000)*num_floors
    col_vol = num_cols * (col_size/1000)**2 * floor_h_m * num_floors
    beam_len_total = 0
    for _ in range(nx-1):
        beam_len_total += (ny) * (xs[1]-xs[0])/1000
    for _ in range(ny-1):
        beam_len_total += (nx) * (ys[1]-ys[0])/1000
    beam_vol = beam_len_total * (beam_b/1000) * (beam_d/1000) * num_floors
    foot_vol = num_cols * (foot/1000)**2 * 0.5
    total_concrete = slab_vol + col_vol + beam_vol + foot_vol
    rebar_ton = total_concrete * 0.110
    formwork = beam_len_total * (beam_d/1000) * 2 * num_floors + col_vol * 8
    wall_len = 0
    for _r, (rx0, ry0, rx1, ry1) in placements:
        wall_len += 2*((rx1-rx0)+(ry1-ry0))/1000
    wall_area = wall_len * floor_h_m * num_floors
    brick_count = wall_area * 50
    flooring_area = (L/1000)*(W/1000)*num_floors
    paint_area = wall_area * 2

    boq_rows = [("Item", "Unit", "Qty", "Rate", "Amount EGP")]
    rates = [("Concrete C30/37", "m3", total_concrete, 2500),
             ("Reinforcement Steel", "ton", rebar_ton, 15000),
             ("Formwork", "m2", formwork, 300),
             ("Brick Masonry", "nos", brick_count, 2.5),
             ("Floor Tiling", "m2", flooring_area, 150),
             ("Wall Painting", "m2", paint_area, 30),
             ("Windows Aluminum", "nos", len(win_rows)-1, 2000),
             ("Doors Wood", "nos", len(door_rows)-1, 3000)]
    grand = 0
    for name, unit, qty, rate in rates:
        amount = qty * rate
        grand += amount
        boq_rows.append((name, unit, f"{qty:.1f}", f"{rate:,.0f}", f"{amount:,.0f}"))
    boq_rows.append(("GRAND TOTAL", "", "", "", f"{grand:,.0f}"))
    draw_table(msp, bx, by, [2600, 800, 900, 1000, 1600], boq_rows)

    # --- Structural notes ---
    ny_y = by - (len(boq_rows) + 2) * 350 - 500
    draw_label(msp, bx + 3000, ny_y + 500, "STRUCTURAL NOTES (ECP 203)",
               'ANNO-TITLE', 300, 7)
                              # ==================================================================
    # SHEET 5 — SCHEDULES (columns, beams, footings) + TYPICAL DETAILS
    # ==================================================================
    sch_x = bx
    sch_y = ny_y - (len(notes_rows) + 3) * 350 - 1500

    # --- Column schedule ---
    draw_label(msp, sch_x + 4000, sch_y + 500, "COLUMN SCHEDULE",
               'ANNO-TITLE', 300, 7)
    col_sched = [("Mark", "Size (mm)", "Main Bars", "Stirrups", "Qty")]
    n_bars_col = 8
    for i, (label, cx, cy) in enumerate(col_labels[:20]):
        col_sched.append((
            label, f"{col_size}x{col_size}",
            f"{n_bars_col}D16", "D8@150", "1",
        ))
    draw_table(msp, sch_x, sch_y, [1200, 1600, 1400, 1400, 800], col_sched)

    # --- Beam schedule ---
    beam_y = sch_y - (len(col_sched) + 3) * 350 - 1000
    draw_label(msp, sch_x + 4000, beam_y + 500, "BEAM SCHEDULE",
               'ANNO-TITLE', 300, 7)
    beam_sched = [("Mark", "Size (mm)", "Top Bars", "Bottom Bars", "Stirrups", "Span (m)")]
    for i in range(min(12, len(beam_labels))):
        bsize = f"{beam_b}x{beam_d}"
        beam_sched.append((
            f"B{i+1}", bsize, "2D16", "3D16", "D8@150", "5.00",
        ))
    draw_table(msp, sch_x, beam_y,
               [1000, 1500, 1300, 1300, 1400, 1100], beam_sched)

    # --- Footing schedule ---
    foot_y = beam_y - (len(beam_sched) + 3) * 350 - 1000
    draw_label(msp, sch_x + 4000, foot_y + 500, "FOOTING SCHEDULE",
               'ANNO-TITLE', 300, 7)
    foot_sched = [("Mark", "Size (mm)", "Depth (mm)", "Bottom R/F", "Qty")]
    for i in range(min(12, len(foot_labels))):
        foot_sched.append((
            f"F{i+1}", f"{foot}x{foot}", "500",
            "D12@150 both ways", "1",
        ))
    draw_table(msp, sch_x, foot_y,
               [1200, 1700, 1400, 2600, 800], foot_sched)

    # --- Typical column section detail ---
    det_x = sch_x
    det_y = foot_y - (len(foot_sched) + 3) * 350 - 2500
    draw_label(msp, det_x + 2500, det_y + 3500, "TYPICAL COLUMN SECTION  (1:20)",
               'ANNO-TITLE', 300, 7)
    _draw_column_section_detail(msp, det_x, det_y, col_size * 2.5, col_size * 2.5,
                                 n_bars=8, bar_dia=16, stirrup_dia=8)
    draw_label(msp, det_x + col_size * 1.25, det_y - 700,
               f"{col_size}x{col_size} 8D16  D8@150 c/c", 'ANNO-TEXT', 220, 7)

    # --- Typical beam section detail ---
    bd_x = det_x + 4000
    draw_label(msp, bd_x + 2500, det_y + 3500, "TYPICAL BEAM SECTION  (1:20)",
               'ANNO-TITLE', 300, 7)
    _draw_beam_section_detail(msp, bd_x, det_y, beam_b * 3.5, beam_d * 1.8,
                              top_bars=2, bot_bars=3, stirrup_dia=8)
    draw_label(msp, bd_x + beam_b * 1.75, det_y - 700,
               f"{beam_b}x{beam_d} 2D16 top, 3D16 bot", 'ANNO-TEXT', 220, 7)

    # --- Typical footing section detail ---
    ft_x = bd_x + 4000
    draw_label(msp, ft_x + 2500, det_y + 3500, "TYPICAL FOOTING SECTION  (1:20)",
               'ANNO-TITLE', 300, 7)
    _draw_footing_section_detail(msp, ft_x, det_y, foot * 1.6, 500,
                                  col_size, n_bars_bot=8)
    draw_label(msp, ft_x + foot * 0.8, det_y - 700,
               f"{foot}x{foot}x500  D12@150 B/W", 'ANNO-TEXT', 220, 7)
    notes = [
        "1. All dimensions are in millimetres unless noted otherwise.",
        "2. Concrete grade: C30/37 for columns & beams, C25/30 for slabs.",
        f"3. Column size: {col_size} x {col_size} mm (based on {num_floors} floors).",
        f"4. Beam size: {beam_b} x {beam_d} mm typical.",
        f"5. Slab thickness: {slab_t} mm (solid slab, max span 5.0 m).",
        f"6. Footing size: {foot} x {foot} mm x 500 mm depth typical.",
        "7. Reinforcement: Grade 400/600 per ECP 203.",
        "8. Cover: 25 mm slabs, 40 mm columns, 50 mm footings.",
        "9. Max clear span: 5000 mm (ECP 203 limit for RC solid slab).",
        "10. Setbacks per Egyptian Building Law 119/2008.",
        f"11. Total plot area: {plot_area:.1f} m2. Coverage: {coverage*100:.0f}%.",
        f"12. Max permitted floors: {max_floors} (street width {sw} m).",
    ]
    notes_rows = [("Note",)] + [(n,) for n in notes]
    draw_table(msp, bx, ny_y, [9000], notes_rows, row_h=400)

    # ==================================================================
    # PROJECT TITLE BLOCK (bottom-right of layout)
    # ==================================================================
    tb_title_x = tb_x
    tb_title_y = win_y - (len(win_rows) + 2) * 350 - 1000
    tb_w, tb_h = 9500, 4000
    msp.add_lwpolyline([(tb_title_x, tb_title_y),
                        (tb_title_x+tb_w, tb_title_y),
                        (tb_title_x+tb_w, tb_title_y-tb_h),
                        (tb_title_x, tb_title_y-tb_h)],
                       dxfattribs={'layer': 'ANNO-TITLE', 'color': 7}, close=True)
    msp.add_line((tb_title_x, tb_title_y-800), (tb_title_x+tb_w, tb_title_y-800),
                 dxfattribs={'layer': 'ANNO-TITLE', 'color': 7})
    msp.add_line((tb_title_x, tb_title_y-1800), (tb_title_x+tb_w, tb_title_y-1800),
                 dxfattribs={'layer': 'ANNO-TITLE', 'color': 7})
    msp.add_line((tb_title_x, tb_title_y-2800), (tb_title_x+tb_w, tb_title_y-2800),
                 dxfattribs={'layer': 'ANNO-TITLE', 'color': 7})

    draw_label(msp, tb_title_x + tb_w/2, tb_title_y - 400,
               "SMART EGY-CIVIL AI AUDITOR", 'ANNO-TITLE', 320, 7)
    draw_label(msp, tb_title_x + tb_w/2, tb_title_y - 1300,
               f"PROJECT: {_ascii(params.get('project_name','Residential Building'))[:40]}",
               'ANNO-TEXT', 240, 7)
    draw_label(msp, tb_title_x + tb_w/2, tb_title_y - 2300,
               f"ENGINEER: {_ascii(params.get('engineer','Eng. Mohamed'))[:40]}    "
               f"DATE: {params.get('date','')}",
               'ANNO-TEXT', 240, 7)
    draw_label(msp, tb_title_x + tb_w/2, tb_title_y - 3400,
               f"PLOT {plot_area:.1f} m2 | FLOORS {num_floors} | COVERAGE {coverage*100:.0f}% | SCALE 1:100 (mm)",
               'ANNO-TEXT', 240, 7)

    # ==================================================================
    # WRITE BINARY DXF
    # ==================================================================
    dxf_buf = io.BytesIO()
    doc.write(dxf_buf, fmt='bin')
    dxf_bytes = dxf_buf.getvalue()

    # Build BOQ dataframe for UI
    boq_df = pd.DataFrame([{'Item': n, 'Quantity': round(q, 2), 'Unit': u,
                            'Unit Rate (EGP)': r, 'Total Cost (EGP)': round(q*r, 2)}
                           for n, u, q, r in rates])

    layout_info = {
        'plot_area': round(plot_area, 2),
        'street_width': sw,
        'location': params.get('location', ''),
        'max_floors': max_floors,
        'num_floors': num_floors,
        'footprint_area': round((L/1000)*(W/1000), 2),
        'building_width': round(W/1000, 2),
        'building_length': round(L/1000, 2),
        'front_setback': front_sb,
        'rear_setback': rear_sb,
        'side_setback': side_sb,
        'num_rooms': len(placements),
        'num_columns': len(cols),
        'coverage_ratio': f"{coverage*100:.0f}%",
    }
    return {'dxf': dxf_bytes, 'boq': boq_df.to_dict('records'), 'info': layout_info}


# ======================================================================
# ADDITIVE HIGH-TRAFFIC LAYER
# ---------------------------------------------------------------------
# Nothing above this line was changed.
# This file has NO Gemini calls — the async-Gemini part of the pattern
# does not apply here. Only the cpu_bound + BytesIO parts do.
# ======================================================================

def _strip_thumbnail_section(text: str) -> str:
    """
    Remove the THUMBNAILIMAGE section from an ASCII DXF string.

    AutoCAD embeds a small PNG preview in this section as hex data under
    tag 310. If that hex is truncated by even one character (which happens
    when files pass through upload/download without byte-perfect handling),
    ezdxf's unhexlify() raises 'Odd-length string'. We don't need the
    preview, so drop the whole section before parsing.
    """
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
            # Skip to the matching ENDSEC
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
    """
    Rebuild a fresh ezdxf document from raw DXF bytes.

    ASCII path:
      - decode
      - strip THUMBNAILIMAGE (avoids the odd-length hex crash)
      - ezdxf.read on the cleaned text
      - if that still fails, ezdxf.recover.read (ignores bad entities)

    Binary path:
      - ezdxf.read on BytesIO, ezdxf.recover.read as fallback.
    """
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

    # ASCII path
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


def _extract_areas_from_dxf_worker(doc_bytes, unit, workflow):
    """Runs INSIDE the cpu_bound worker. Plain bytes in, plain list out."""
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    return extract_areas_from_dxf(doc, unit=unit, workflow=workflow)


def _detect_dxf_layers_worker(doc_bytes):
    """Runs INSIDE the cpu_bound worker. Plain bytes in, plain dict out."""
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    result = detect_dxf_layers(doc)
    # `types` is a set — convert to list so the result is picklable back.
    for layer, info in result.items():
        if isinstance(info.get('types'), set):
            info['types'] = sorted(info['types'])
    return result


async def detect_dxf_layers_async(doc_bytes):
    """Async / process-pool version of detect_dxf_layers()."""
    from config import cpu_bound_limited
    return await cpu_bound_limited(_detect_dxf_layers_worker, doc_bytes)


async def extract_areas_from_dxf_async(doc_bytes, unit='mm', workflow='architectural'):
    """Async / process-pool version of extract_areas_from_dxf()."""
    from config import cpu_bound_limited
    return await cpu_bound_limited(_extract_areas_from_dxf_worker,
                                   doc_bytes, unit, workflow)


async def build_complete_project_async(params):
    """
    Async / process-pool version of build_complete_project().
    `params` must be a plain dict (it already is in every call site).
    """
    from config import cpu_bound_limited
    return await cpu_bound_limited(build_complete_project, params)
