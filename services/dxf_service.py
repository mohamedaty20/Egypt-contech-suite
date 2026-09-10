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


def extract_areas_from_dxf(doc, unit='mm', workflow='architectural'):
    scale = {'mm': 1e-6, 'cm': 1e-4, 'm': 1.0}.get(unit, 1e-6)
    results = []
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE') and entity.closed:
            try:
                pts = ([(p.x, p.y) for p in entity.get_points()]
                       if entity.dxftype() == 'LWPOLYLINE'
                       else [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices])
                a = 0.0
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i+1) % len(pts)]
                    a += x1*y2 - x2*y1
                a = abs(a) / 2.0
                label = ""
                cx = sum(p[0] for p in pts)/len(pts)
                cy = sum(p[1] for p in pts)/len(pts)
                for t in msp.query('TEXT MTEXT'):
                    if abs(t.dxf.insert.x - cx) < 10 and abs(t.dxf.insert.y - cy) < 10:
                        label = t.dxf.text
                        break
                results.append({'layer': entity.dxf.layer, 'area_m2': round(a*scale, 4),
                                'label': label.strip() if label else '', 'vertices': len(pts)})
            except Exception:
                continue
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
    seed = hash((params.get('plot_area_m2', 200),
                 params.get('street_width_m', 10),
                 params.get('num_floors', 2),
                 params.get('num_bedrooms', 3)))
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
#
# What is added:
#   - _extract_areas_from_dxf_worker(bytes, unit, workflow)
#         Picklable worker that rebuilds the ezdxf doc inside the pool.
#   - detect_dxf_layers_async(doc_bytes)
#   - extract_areas_from_dxf_async(doc_bytes, unit, workflow)
#   - build_complete_project_async(params)
#
# All four delegate to the UNCHANGED sync functions above via
# config.cpu_bound_limited (which is semaphore-gated).
# ======================================================================

_DXF_BINARY_MAGIC = b"AutoCAD Binary DXF\r\n\x1a\x00"

def _open_dxf_doc_from_bytes(doc_bytes):
    """
    Rebuild a fresh ezdxf document from raw DXF bytes.

    Tries binary first if the header looks binary, otherwise ASCII.
    If the first attempt fails, falls back to the other encoding.
    Prints the first 32 bytes so we can see what the file actually is.
    """
    if isinstance(doc_bytes, str):
        doc_bytes = doc_bytes.encode("utf-8")

    head = doc_bytes[:64]
    print(f"[dxf] first 32 bytes: {head[:32]!r}")
    print(f"[dxf] contains NUL: {b'\\x00' in head}, "
          f"contains SUB: {b'\\x1a' in head}")

    # Strong signal: binary DXF has an ASCII magic header
    looks_binary = (
        head.startswith(b"AutoCAD Binary DXF")
        or (b"\x00" in head[:32])
    )

    if looks_binary:
        try:
            return ezdxf.read(io.BytesIO(doc_bytes))
        except Exception as e:
            print(f"[dxf] binary path failed: {e!r} — falling back to ASCII")

    # ASCII path
    try:
        text = doc_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = doc_bytes.decode("latin-1")
        except Exception:
            text = doc_bytes.decode("utf-8", errors="replace")

    try:
        return ezdxf.read(io.StringIO(text))
    except Exception as e:
        print(f"[dxf] ASCII path failed: {e!r} — falling back to binary")
        return ezdxf.read(io.BytesIO(doc_bytes))


def _extract_areas_from_dxf_worker(doc_bytes, unit, workflow):
    """Runs INSIDE the cpu_bound worker. Plain bytes in, plain list out."""
    doc = _open_doc_from_bytes(doc_bytes)
    return extract_areas_from_dxf(doc, unit=unit, workflow=workflow)


def _detect_dxf_layers_worker(doc_bytes):
    """Runs INSIDE the cpu_bound worker. Plain bytes in, plain dict out."""
    doc = _open_doc_from_bytes(doc_bytes)
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
