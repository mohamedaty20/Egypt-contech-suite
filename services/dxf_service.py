# services/dxf_service.py — Professional Architectural generator
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
    scale = {'mm': 1e-6, 'cm': 1e-4, 'm': 1.0}.get(unit, 1e-6)
    results = []
    msp = doc.modelspace()
    counts = {}
    for e in msp:
        counts[e.dxftype()] = counts.get(e.dxftype(), 0) + 1
    print(f"[dxf-extract] entity census: {counts}")
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


def _draw_balcony(msp, x_left, x_right, outer_y, side,
                  depth=1200, layer='A-BALCONY'):
    if side == 'S':
        y_out = outer_y - depth
    else:
        y_out = outer_y + depth
    pts = [(x_left, outer_y), (x_right, outer_y),
           (x_right, y_out),   (x_left, y_out)]
    msp.add_lwpolyline(pts, close=True,
                       dxfattribs={'layer': layer, 'color': 5})
    msp.add_line((x_left, y_out), (x_right, y_out),
                 dxfattribs={'layer': layer, 'color': 5})


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
    total_w = sum(col_widths)
    nrows = len(rows)
    total_h = nrows * row_h
    msp.add_lwpolyline([(x, y), (x+total_w, y), (x+total_w, y-total_h),
                        (x, y-total_h)], dxfattribs={'layer': layer, 'color': color}, close=True)
    cx = x
    for cw in col_widths[:-1]:
        cx += cw
        msp.add_line((cx, y), (cx, y-total_h),
                     dxfattribs={'layer': layer, 'color': color})
    for r in range(1, nrows):
        yy = y - r*row_h
        msp.add_line((x, yy), (x+total_w, yy),
                     dxfattribs={'layer': layer, 'color': color})
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
    for i, cx in enumerate(xs):
        msp.add_line((cx, y0 - 1500), (cx, y0 + W + 1500),
                     dxfattribs={'layer': layer, 'color': 2, 'linetype': 'DASHED'})
        msp.add_circle((cx, y0 - 1800), radius=400, dxfattribs={'layer': layer, 'color': 2})
        draw_label(msp, cx, y0 - 1800, str(i+1), layer, 250, 2)
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    for j, cy in enumerate(ys):
        msp.add_line((x0 - 1500, cy), (x0 + L + 1500, cy),
                     dxfattribs={'layer': layer, 'color': 2, 'linetype': 'DASHED'})
        msp.add_circle((x0 - 1800, cy), radius=400, dxfattribs={'layer': layer, 'color': 2})
        draw_label(msp, x0 - 1800, cy, letters[j % len(letters)], layer, 250, 2)


# ======================================================================
# Sophisticated drawing helpers
# ======================================================================
def _draw_dimension_chain(msp, x0, y_ref, grid_positions, offset_mm, label=None,
                           layer='ANNO-DIM', color=2):
    if len(grid_positions) < 2:
        return
    y_dim = y_ref + offset_mm
    msp.add_line((grid_positions[0], y_dim), (grid_positions[-1], y_dim),
                 dxfattribs={'layer': layer, 'color': color})
    for i, x in enumerate(grid_positions):
        msp.add_line((x, y_dim - 100), (x, y_dim + 100),
                     dxfattribs={'layer': layer, 'color': color})
        if i < len(grid_positions) - 1:
            seg_mm = grid_positions[i + 1] - grid_positions[i]
            mid = (grid_positions[i] + grid_positions[i + 1]) / 2
            draw_label(msp, mid, y_dim + 220, f"{seg_mm:.0f}", layer, height=180, color=color)
    for x in (grid_positions[0], grid_positions[-1]):
        msp.add_line((x, y_ref), (x, y_dim + 200),
                     dxfattribs={'layer': layer, 'color': color})
    if label:
        mid = (grid_positions[0] + grid_positions[-1]) / 2
        draw_label(msp, mid, y_dim + 700, label, layer, height=220, color=color)


def _draw_native_dims(msp, xs, ys, x0, y0, offset=7500, layer='ANNO-DIM'):
    ok = 0
    fail = 0
    try:
        y_base = y0 - offset
        for i in range(len(xs) - 1):
            try:
                d = msp.add_linear_dim(
                    base=(xs[i], y_base),
                    p1=(xs[i], y0),
                    p2=(xs[i + 1], y0),
                    dimstyle='Standard',
                    dxfattribs={'layer': layer},
                )
                d.render()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[dims-x] seg {i} failed: {e!r}")
    except Exception as e:
        print(f"[dims-x] outer failed: {e!r}")
    try:
        x_base = x0 - offset
        for i in range(len(ys) - 1):
            try:
                d = msp.add_linear_dim(
                    base=(x_base, ys[i]),
                    p1=(x0, ys[i]),
                    p2=(x0, ys[i + 1]),
                    angle=90,
                    dimstyle='Standard',
                    dxfattribs={'layer': layer},
                )
                d.render()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"[dims-y] seg {i} failed: {e!r}")
    except Exception as e:
        print(f"[dims-y] outer failed: {e!r}")
    print(f"[dims-native] created ok={ok} fail={fail}")


def _draw_north_arrow(msp, x, y, size=1200, layer='ANNO-SYMBOL'):
    msp.add_circle((x, y), radius=size / 2,
                   dxfattribs={'layer': layer, 'color': 7})
    half = size * 0.22
    msp.add_lwpolyline([(x, y + half * 1.4), (x - half, y - half * 0.6),
                        (x + half, y - half * 0.6)],
                       dxfattribs={'layer': layer, 'color': 7}, close=True)
    draw_label(msp, x, y + size * 0.75, "N", layer, height=320, color=7)


def _draw_section_marker(msp, x, y, tag, direction='down', layer='ANNO-SECTION'):
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
    by = y - arrow_len - 400 if direction == 'down' else y + arrow_len + 400
    msp.add_circle((x, by), radius=500, dxfattribs={'layer': layer, 'color': 1})
    draw_label(msp, x, by, tag, layer, height=280, color=1)


# ======================================================================
# Layout engine
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


def _columns_along_wall(p1, p2, columns, tolerance=200):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1:
        return []
    ux, uy = dx/L, dy/L
    hits = []
    for cx, cy in columns:
        t = (cx - x1) * ux + (cy - y1) * uy
        if t < -50 or t > L + 50:
            continue
        proj_x, proj_y = x1 + ux*t, y1 + uy*t
        if math.hypot(cx - proj_x, cy - proj_y) < tolerance:
            hits.append(t)
    return hits


def _free_spans(L, col_positions, col_block=400, clearance=250):
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
    x1, y1 = p1; x2, y2 = p2
    L = math.hypot(x2-x1, y2-y1)
    if L < opening_width + 400:
        return None
    col_positions = _columns_along_wall(p1, p2, columns)
    spans = _free_spans(L, col_positions, col_block=col_size)
    valid = [(a, b) for a, b in spans if b - a >= opening_width + 200]
    if not valid:
        return None
    longest = max(valid, key=lambda s: s[1] - s[0])
    return (longest[0] + longest[1]) / 2


def _is_balcony_room(room):
    rt = (room.get('type') or '').lower()
    return rt in ('living', 'bedroom_master')


def _is_bathroom(room):
    rt = (room.get('type') or '').lower()
    rn = (room.get('name') or '').lower()
    return (rt == 'bathroom' or 'bath' in rn or 'wc' in rn or 'toilet' in rn)


# ======================================================================
# MAIN GENERATOR
# ======================================================================
def build_complete_project(params):
    print("[build] === build_complete_project START ===")
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

    layout_plan = params.get('layout_plan') or {}
    ai_rooms = layout_plan.get('rooms', [])
    stair_cell = layout_plan.get('stair_cell')
    nb = params.get('num_bedrooms', 3)
    nba = params.get('num_bathrooms', 2)

    print(f"[build] rooms in layout_plan: {len(ai_rooms)}")

    from shapely.geometry import Polygon as _ShPoly
    from shapely.ops import unary_union as _sh_union
    wall_polys = []

    def _add_wall_rect_from_line(p1, p2, thickness, gaps=None):
        x1, y1 = p1; x2, y2 = p2
        L = math.hypot(x2 - x1, y2 - y1)
        if L < 1:
            return
        if not gaps:
            r = _wall_rect(p1, p2, thickness)
            if r:
                wall_polys.append(_ShPoly(r))
            return
        gaps_sorted = sorted(gaps, key=lambda g: g[0])
        cursor = 0.0
        for gc, gw in gaps_sorted:
            seg_start = cursor
            seg_end = max(cursor, gc - gw / 2)
            if seg_end - seg_start > 20:
                a = (x1 + (x2 - x1) * seg_start / L, y1 + (y2 - y1) * seg_start / L)
                b = (x1 + (x2 - x1) * seg_end / L, y1 + (y2 - y1) * seg_end / L)
                r = _wall_rect(a, b, thickness)
                if r:
                    wall_polys.append(_ShPoly(r))
            cursor = max(cursor, gc + gw / 2)
        if cursor < L - 20:
            a = (x1 + (x2 - x1) * cursor / L, y1 + (y2 - y1) * cursor / L)
            r = _wall_rect(a, (x2, y2), thickness)
            if r:
                wall_polys.append(_ShPoly(r))

    if not ai_rooms:
        ai_rooms = [
            {"name": "Living Room", "type": "living", "area_m2": 26, "priority": 1, "zone": "public", "needs_window": True},
            {"name": "Kitchen", "type": "kitchen", "area_m2": 10, "priority": 2, "zone": "public", "needs_window": True},
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

    _pre_positioned = bool(ai_rooms) and all(
        ('x' in r and 'y' in r and 'w' in r and 'h' in r) for r in ai_rooms
    )

    if layout_plan.get('building'):
        b = layout_plan['building']
        x0 = float(b['x'])
        y0 = float(b['y'])
        x1 = x0 + float(b['w'])
        y1 = y0 + float(b['h'])
        L = x1 - x0
        W = y1 - y0
    else:
        L = building_length * 1000
        W = building_width * 1000
        x0, y0 = 0.0, 0.0
        x1, y1 = L, W

    # ----- Structural grid -----
    span_max = 5000
    nx = max(3, math.ceil(L / span_max) + 1)
    ny = max(2, math.ceil(W / span_max) + 1)
    xs = np.linspace(x0, x1, nx).tolist()
    ys = np.linspace(y0, y1, ny).tolist()
    cols = [(cx, cy) for cx in xs for cy in ys]
    col_size = 300 if num_floors <= 2 else 350

    # ----- DXF setup -----
    doc = ezdxf.new(dxfversion='R2000', setup=True)
    msp = doc.modelspace()

    doc.header['$INSUNITS']    = 4
    doc.header['$MEASUREMENT'] = 1
    doc.header['$LUNITS']      = 2
    doc.header['$LUPREC']      = 0
    doc.header['$AUNITS']      = 0
    doc.header['$AUPREC']      = 0
    doc.header['$DIMSCALE']    = 100
    doc.header['$DIMLFAC']     = 1.0
    doc.header['$DIMTXT']      = 3.5
    doc.header['$DIMASZ']      = 3.5
    doc.header['$DIMDEC']      = 0
    doc.header['$DIMZIN']      = 8

    try:
        _ds = doc.dimstyles.get('Standard')
    except Exception:
        _ds = None
    if _ds is None:
        try:
            _ds = doc.dimstyles.add('Standard')
        except Exception:
            _ds = None
    if _ds is not None:
        try:
            _ds.dxf.dimscale = 100
            _ds.dxf.dimlfac  = 1.0
            _ds.dxf.dimtxt   = 3.5
            _ds.dxf.dimasz   = 3.5
            _ds.dxf.dimexe   = 1.5
            _ds.dxf.dimexo   = 1.0
            _ds.dxf.dimdec   = 0
            _ds.dxf.dimzin   = 8
            _ds.dxf.dimtad   = 1
            _ds.dxf.dimtih   = 1
            _ds.dxf.dimtoh   = 1
            _ds.dxf.dimtix   = 1
        except Exception as _e:
            print(f"[dimstyle] could not configure: {_e!r}")

    print(f"[dxf-units] INSUNITS={doc.header.get('$INSUNITS')} "
          f"MEASUREMENT={doc.header.get('$MEASUREMENT')} "
          f"DIMLFAC={doc.header.get('$DIMLFAC')} DIMSCALE={doc.header.get('$DIMSCALE')}")

    layers_def = {
        'A-WALL-EXT': {'color': 7, 'lineweight': 50},
        'A-WALL-INT': {'color': 8, 'lineweight': 25},
        'A-DOOR':     {'color': 3, 'lineweight': 18},
        'A-WINDOW':   {'color': 5, 'lineweight': 13},
        'A-BALCONY':  {'color': 5, 'lineweight': 25},
        'A-FURN':     {'color': 6, 'lineweight': 9},
        'A-ROOM-TEXT':{'color': 4, 'lineweight': 13},
        'A-CORE':     {'color': 4, 'lineweight': 30},
        'S-GRID':     {'color': 2, 'lineweight': 13},
        'ANNO-DIM':   {'color': 2, 'lineweight': 13},
        'ANNO-TEXT':  {'color': 7, 'lineweight': 13},
        'ANNO-TITLE': {'color': 7, 'lineweight': 25},
        'ANNO-TABLE': {'color': 7, 'lineweight': 18},
        'ANNO-SYMBOL':{'color': 7, 'lineweight': 25},
        'ANNO-SECTION':{'color': 1, 'lineweight': 25},
        'ANNO-BORDER':{'color': 7, 'lineweight': 50},
    }
    for n, p in layers_def.items():
        create_dxf_layer(doc, n, p['color'], lineweight=p['lineweight'])

    msp.add_lwpolyline(
        [(0, 0), (L, 0), (L, W), (0, W)],
        dxfattribs={'layer': 'ANNO-BORDER', 'color': 1, 'linetype': 'DASHED'},
        close=True,
    )

    # ---- Placements ----
    wall_ext_t = 250
    int_t = 150
    corridor_h = 1300
    mid_y = (y0 + y1) / 2

    public_zone = (x0, y0, x1, mid_y - corridor_h/2)
    private_zone = (x0, mid_y + corridor_h/2, x1, y1)

    if _pre_positioned:
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

    # ---- Aggressive bathroom shrink: force target size, give leftover to Store ----
    _expanded = []
    _n_bath = 0
    for _room, (_rx0, _ry0, _rx1, _ry1) in placements:
        if _is_bathroom(_room):
            _w = _rx1 - _rx0
            _h = _ry1 - _ry0
            _bw = min(_w, 2000)
            _bh = min(_h, 2200)
            # Bath occupies top-left of the original cell
            _expanded.append((_room,
                              (_rx0, _ry1 - _bh, _rx0 + _bw, _ry1)))
            # Bottom strip → Store
            if _ry1 - _bh > _ry0 + 100:
                _expanded.append((
                    {'name': 'Store', 'type': 'store', 'needs_window': False},
                    (_rx0, _ry0, _rx1, _ry1 - _bh),
                ))
            # Right strip → Store
            if _rx0 + _bw < _rx1 - 100:
                _expanded.append((
                    {'name': 'Store', 'type': 'store', 'needs_window': False},
                    (_rx0 + _bw, _ry1 - _bh, _rx1, _ry1),
                ))
            _n_bath += 1
        else:
            _expanded.append((_room, (_rx0, _ry0, _rx1, _ry1)))
    placements = _expanded
    print(f"[build] bathrooms shrunk: {_n_bath}, total placements: {len(placements)}")

    # ---- Identify balcony rooms FIRST (before window placement) ----
    balcony_rooms = []
    for room, (rx0, ry0, rx1, ry1) in placements:
        if not _is_balcony_room(room):
            continue
        bw_ok = (rx1 - rx0) - 800 >= 1500
        if not bw_ok:
            continue
        side = None
        if abs(ry0 - y0) < wall_ext_t + 60:
            side = 'S'
        elif abs(ry1 - y1) < wall_ext_t + 60:
            side = 'N'
        if side:
            balcony_rooms.append((room, (rx0, ry0, rx1, ry1), side))

    # ---- Exterior openings ----
    ext_openings = {'bottom': [], 'top': [], 'left': [], 'right': []}
    balcony_openings = []  # (side, along_center, width) — WIDE door opening, no window

    for room, (rx0, ry0, rx1, ry1) in placements:
        is_balcony = any(b[0] is room for b in balcony_rooms)
        if is_balcony:
            # Wide opening instead of window
            for _r, (brx0, bry0, brx1, bry1), bside in balcony_rooms:
                if _r is not room:
                    continue
                op_w = min(brx1 - brx0 - 600, 2400)
                wx = (brx0 + brx1) / 2
                if bside == 'S':
                    balcony_openings.append(('bottom', wx - x0, op_w))
                elif bside == 'N':
                    balcony_openings.append(('top', wx - x0, op_w))
                break
            # Still allow window on the OTHER (non-balcony) exterior wall if any
            # (skip for simplicity — one exterior wall per room in this grid)
            continue
        if not room.get('needs_window'):
            continue
        room_w = rx1 - rx0
        room_h = ry1 - ry0
        win_h = max(900, min(1400, room_w * 0.45))
        win_v = max(900, min(1200, room_h * 0.45))
        if abs(ry0 - y0) < wall_ext_t + 60:
            wx = (rx0 + rx1) / 2
            ext_openings['bottom'].append((wx - x0, win_h))
        if abs(ry1 - y1) < wall_ext_t + 60:
            wx = (rx0 + rx1) / 2
            ext_openings['top'].append((wx - x0, win_h))
        if abs(rx0 - x0) < wall_ext_t + 60:
            wy = (ry0 + ry1) / 2
            ext_openings['left'].append((wy - y0, win_v))
        if abs(rx1 - x1) < wall_ext_t + 60:
            wy = (ry0 + ry1) / 2
            ext_openings['right'].append((wy - y0, win_v))

    # ---- Entry door — always reserve ----
    entry_door_cx = None
    for room, (rx0, ry0, rx1, ry1) in placements:
        nm = (room.get('name') or '').lower()
        if 'entry' in nm or 'hall' in nm or 'foyer' in nm:
            if abs(ry0 - y0) < wall_ext_t + 60:
                entry_door_cx = (rx0 + rx1) / 2
                break
    if entry_door_cx is None and placements:
        lower_centres = [(rx0 + rx1) / 2
                         for _r, (rx0, ry0, rx1, ry1) in placements
                         if (ry0 + ry1) / 2 < mid_y]
        if lower_centres:
            entry_door_cx = lower_centres[len(lower_centres) // 2]
    if entry_door_cx is None:
        entry_door_cx = (x0 + x1) / 2

    _door_c = entry_door_cx - x0
    _door_w = 1100
    _shift_deltas = (0, 800, -800, 1600, -1600, 2400, -2400, 3200, -3200)
    _final_c = _door_c
    for _d in _shift_deltas:
        _c = _door_c + _d
        if _c - _door_w / 2 < 300 or _c + _door_w / 2 > (x1 - x0) - 300:
            continue
        _hit = False
        for _s, _gc, _gw in balcony_openings:
            if _s == 'bottom' and abs(_c - _gc) < (_gw + _door_w) / 2 + 200:
                _hit = True; break
        if _hit: continue
        if not any(abs(_c - gc) < (gw + _door_w) / 2 + 200
                   for gc, gw in ext_openings['bottom']):
            _final_c = _c
            break
    entry_door_cx = x0 + _final_c
    ext_openings['bottom'].append((_final_c, _door_w))
    print(f"[build] entry door at x={entry_door_cx:.0f} w={_door_w}")

    # ---- Combine window gaps + balcony door gaps for wall cutting ----
    bottom_gaps = ext_openings['bottom'][:]
    top_gaps    = ext_openings['top'][:]
    left_gaps   = ext_openings['left'][:]
    right_gaps  = ext_openings['right'][:]
    for _s, _gc, _gw in balcony_openings:
        if _s == 'bottom': bottom_gaps.append((_gc, _gw))
        elif _s == 'top':  top_gaps.append((_gc, _gw))

    half_ext = wall_ext_t / 2
    _add_wall_rect_from_line((x0 + half_ext, y0), (x0 + half_ext, y1),
                             wall_ext_t, gaps=left_gaps)
    _add_wall_rect_from_line((x1 - half_ext, y0), (x1 - half_ext, y1),
                             wall_ext_t, gaps=right_gaps)
    _add_wall_rect_from_line((x0, y0 + half_ext), (x1, y0 + half_ext),
                             wall_ext_t, gaps=bottom_gaps)
    _add_wall_rect_from_line((x0, y1 - half_ext), (x1, y1 - half_ext),
                             wall_ext_t, gaps=top_gaps)

    # ---- Draw windows (skip entry + balcony gaps) ----
    for wx, ww in ext_openings['bottom']:
        if abs(wx - (entry_door_cx - x0)) < 10:
            continue
        draw_window(msp, wx + x0, y0 + half_ext, ww, wall_ext_t, is_horizontal=True)
    for wx, ww in ext_openings['top']:
        draw_window(msp, wx + x0, y1 - half_ext, ww, wall_ext_t, is_horizontal=True)
    for wy, ww in ext_openings['left']:
        draw_window(msp, x0 + half_ext, wy + y0, ww, wall_ext_t, is_horizontal=False)
    for wy, ww in ext_openings['right']:
        draw_window(msp, x1 - half_ext, wy + y0, ww, wall_ext_t, is_horizontal=False)

    # ---- Interior doors ----
    DOOR_W = 900
    placed_doors = []
    interior_door_gaps = []

    def _door_bbox(cx, cy, w, wt, horiz, flip):
        hw = w / 2
        ht = wt / 2
        if horiz:
            if flip:
                return (cx - hw, cy - w, cx + hw, cy + ht)
            return (cx - hw, cy - ht, cx + hw, cy + w)
        else:
            if flip:
                return (cx - w, cy - hw, cx + ht, cy + hw)
            return (cx - ht, cy - hw, cx + w, cy + hw)

    def _boxes_overlap(a, b, margin=50):
        return not (a[2] + margin < b[0] or b[2] + margin < a[0]
                    or a[3] + margin < b[1] or b[3] + margin < a[1])

    def _draw_sliding_door(msp, cx, cy, width, wall_thickness, is_horizontal=True):
        if is_horizontal:
            msp.add_line((cx - width/2, cy - wall_thickness/4),
                         (cx + width/2, cy - wall_thickness/4),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_line((cx - width/2, cy + wall_thickness/4),
                         (cx + width/2, cy + wall_thickness/4),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
        else:
            msp.add_line((cx - wall_thickness/4, cy - width/2),
                         (cx - wall_thickness/4, cy + width/2),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_line((cx + wall_thickness/4, cy - width/2),
                         (cx + wall_thickness/4, cy + width/2),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})

    corridor_bot = mid_y - corridor_h / 2
    corridor_top = mid_y + corridor_h / 2
    DOOR_WALL_TOL = 250.0

    for room, (rx0, ry0, rx1, ry1) in placements:
        if stair_cell:
            scx0 = float(stair_cell['x'])
            scy0 = float(stair_cell['y'])
            scx1 = scx0 + float(stair_cell['w'])
            scy1 = scy0 + float(stair_cell['h'])
            if (abs(rx0 - scx0) < 50 and abs(ry0 - scy0) < 50
                    and abs(rx1 - scx1) < 50 and abs(ry1 - scy1) < 50):
                continue

        room_cy = (ry0 + ry1) / 2
        room_depth = ry1 - ry0

        faces_corridor_bottom = abs(ry1 - corridor_bot) < DOOR_WALL_TOL
        faces_corridor_top    = abs(ry0 - corridor_top) < DOOR_WALL_TOL

        if room_cy < mid_y:
            if not faces_corridor_bottom:
                continue
            wall_p1, wall_p2 = (rx0, ry1), (rx1, ry1)
            door_flip = True
        else:
            if not faces_corridor_top:
                continue
            wall_p1, wall_p2 = (rx0, ry0), (rx1, ry0)
            door_flip = False

        t_best = best_opening_position(wall_p1, wall_p2, cols, DOOR_W, col_size)
        if t_best is None:
            continue

        wall_L = math.hypot(wall_p2[0] - wall_p1[0], wall_p2[1] - wall_p1[1])
        if wall_L < DOOR_W + 100:
            continue

        ux = (wall_p2[0] - wall_p1[0]) / wall_L
        uy = (wall_p2[1] - wall_p1[1]) / wall_L

        half = DOOR_W / 2 + 50
        shifts = (0,
                  DOOR_W + 150, -(DOOR_W + 150),
                  2 * (DOOR_W + 150), -2 * (DOOR_W + 150),
                  3 * (DOOR_W + 150), -3 * (DOOR_W + 150))

        chosen = None
        for delta in shifts:
            t_try = t_best + delta
            if t_try < half or t_try > wall_L - half:
                continue
            cx = wall_p1[0] + ux * t_try
            cy = wall_p1[1] + uy * t_try
            bbox = _door_bbox(cx, cy, DOOR_W, int_t, True, door_flip)
            collision = False
            for prev in placed_doors:
                if _boxes_overlap(bbox, prev):
                    collision = True
                    break
            if not collision:
                chosen = (cx, cy, bbox)
                break

        if chosen is None:
            continue

        door_cx, door_cy, bbox = chosen
        placed_doors.append(bbox)
        interior_door_gaps.append((door_cx, door_cy, DOOR_W))

        if room_depth >= DOOR_W + 200:
            draw_door(msp, door_cx, door_cy, DOOR_W, int_t,
                      is_horizontal=True, flip=door_flip)
            door_type = "Single Leaf"
        else:
            _draw_sliding_door(msp, door_cx, door_cy, DOOR_W, int_t,
                               is_horizontal=True)
            door_type = "Sliding"
        d_idx += 1
        mark = f"D{d_idx}"
        door_marks.append((mark, door_type, DOOR_W, 2100, 1, room['name']))

    # ---- Entry door symbol (INWARD swing + label) ----
    if entry_door_cx is not None:
        draw_door(msp, entry_door_cx, y0 + wall_ext_t / 2, 1100, wall_ext_t,
                  is_horizontal=True, flip=False)
        draw_label(msp, entry_door_cx, y0 - 800, "ENTRY", 'A-DOOR', 260, 3)
        door_marks.append(("D-ENTRY", "Single Leaf", 1100, 2100, 1, "Apartment Entry"))

    # ---- Interior walls ----
    TOL = 400.0

    def _cluster(values):
        if not values:
            return {}
        s = sorted(set(values))
        clusters = [[s[0]]]
        for v in s[1:]:
            if v - clusters[-1][-1] <= TOL:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        out = {}
        for c in clusters:
            center = sum(c) / len(c)
            for v in c:
                out[v] = center
        return out

    all_vx = []
    all_hy = []
    for room, (rx0, ry0, rx1, ry1) in placements:
        if rx0 > x0 + wall_ext_t: all_vx.append(rx0)
        if rx1 < x1 - wall_ext_t: all_vx.append(rx1)
        if ry0 > y0 + wall_ext_t: all_hy.append(ry0)
        if ry1 < y1 - wall_ext_t: all_hy.append(ry1)

    x_cluster = _cluster(all_vx)
    y_cluster = _cluster(all_hy)

    vert_by_x = {}
    horiz_by_y = {}

    for room, (rx0, ry0, rx1, ry1) in placements:
        if rx0 > x0 + wall_ext_t:
            vert_by_x.setdefault(x_cluster[rx0], []).append((ry0, ry1))
        if rx1 < x1 - wall_ext_t:
            vert_by_x.setdefault(x_cluster[rx1], []).append((ry0, ry1))
        if ry0 > y0 + wall_ext_t:
            horiz_by_y.setdefault(y_cluster[ry0], []).append((rx0, rx1))
        if ry1 < y1 - wall_ext_t:
            horiz_by_y.setdefault(y_cluster[ry1], []).append((rx0, rx1))

    def _merge_ranges(ranges, tol=60.0):
        if not ranges:
            return []
        ranges = sorted(ranges)
        merged = [list(ranges[0])]
        for a, b in ranges[1:]:
            if a <= merged[-1][1] + tol:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(a, b) for a, b in merged]

    door_match_tol = TOL + 100.0
    for y, ranges in horiz_by_y.items():
        for xa, xb in _merge_ranges(ranges):
            if xb - xa < 60: continue
            seg_gaps = []
            for dcx, dcy, dw in interior_door_gaps:
                if abs(dcy - y) <= door_match_tol and (xa - 50) <= dcx <= (xb + 50):
                    seg_gaps.append((dcx - xa, dw))
            if seg_gaps:
                _add_wall_rect_from_line((xa, y), (xb, y), int_t, gaps=seg_gaps)
            else:
                r = _wall_rect((xa, y), (xb, y), int_t)
                if r: wall_polys.append(_ShPoly(r))

    for x, ranges in vert_by_x.items():
        for ya, yb in _merge_ranges(ranges):
            if yb - ya < 60: continue
            r = _wall_rect((x, ya), (x, yb), int_t)
            if r: wall_polys.append(_ShPoly(r))

    if wall_polys:
        try:
            merged = _sh_union(wall_polys)
            geoms = list(merged.geoms) if hasattr(merged, 'geoms') else [merged]
            for g in geoms:
                if g.is_empty: continue
                try:
                    outer = [(float(x), float(y)) for x, y in g.exterior.coords]
                    msp.add_lwpolyline(outer, close=True,
                                       dxfattribs={'layer': 'A-WALL-EXT', 'color': 7})
                    for hole in g.interiors:
                        hole_pts = [(float(x), float(y)) for x, y in hole.coords]
                        msp.add_lwpolyline(hole_pts, close=True,
                                           dxfattribs={'layer': 'A-WALL-EXT', 'color': 7})
                except Exception as e:
                    print(f"[walls] draw geom failed: {e!r}")
        except Exception as e:
            print(f"[walls] union failed: {e!r}")

    # ---- Furniture + labels + schedules ----
    room_schedule = []
    for i, (room, (rx0, ry0, rx1, ry1)) in enumerate(placements):
        w = rx1 - rx0
        h = ry1 - ry0
        area_m2 = (w * h) / 1_000_000
        inset = int_t / 2 + 20
        place_furniture(msp, rx0 + inset, ry0 + inset,
                        w - 2 * inset, h - 2 * inset,
                        room.get('type', 'bedroom'))
        draw_label(msp, (rx0+rx1)/2, (ry0+ry1)/2 + 100,
                   room['name'], 'A-ROOM-TEXT', 180, 4)
        draw_label(msp, (rx0+rx1)/2, (ry0+ry1)/2 - 200,
                   f"{area_m2:.1f} m2", 'A-ROOM-TEXT', 150, 4)
        perimeter = 2*((rx1-rx0)+(ry1-ry0)) / 1000
        room_schedule.append((i+1, room['name'], "Ground", f"{area_m2:.1f}",
                              f"{perimeter:.1f}", "Tiles", "Paint"))

    # ---- Balconies ----
    for room, (rx0, ry0, rx1, ry1), bside in balcony_rooms:
        balcony_w = min(rx1 - rx0 - 800, 3000)
        bcx = (rx0 + rx1) / 2
        if bside == 'S':
            _draw_balcony(msp, bcx - balcony_w/2, bcx + balcony_w/2,
                          y0, 'S', depth=1200)
        else:
            _draw_balcony(msp, bcx - balcony_w/2, bcx + balcony_w/2,
                          y1, 'N', depth=1200)
    print(f"[balcony] total drawn: {len(balcony_rooms)}")

    # ---- Stair core ----
    stair_inset = int_t / 2 + 20
    inner_x0 = x0 + wall_ext_t
    inner_y0 = y0 + wall_ext_t
    inner_x1 = x1 - wall_ext_t
    inner_y1 = y1 - wall_ext_t
    if stair_cell:
        scx0 = float(stair_cell['x']) + stair_inset
        scy0 = float(stair_cell['y']) + stair_inset
        scx1 = float(stair_cell['x']) + float(stair_cell['w']) - stair_inset
        scy1 = float(stair_cell['y']) + float(stair_cell['h']) - stair_inset
        scx0 = max(scx0, inner_x0); scy0 = max(scy0, inner_y0)
        scx1 = min(scx1, inner_x1); scy1 = min(scy1, inner_y1)
        core_x, core_y = scx0, scy0
        core_w = max(600.0, scx1 - scx0)
        core_d = max(600.0, scy1 - scy0)
    else:
        avail_w_core = max(600.0, (inner_x1 - inner_x0) - 800.0)
        avail_d_core = max(600.0, (inner_y1 - inner_y0) - 800.0)
        core_w = min(2400.0, avail_w_core)
        core_d = min(3600.0, avail_d_core)
        core_x = inner_x1 - core_w - 400
        core_y = mid_y - core_d / 2

    msp.add_lwpolyline([(core_x, core_y), (core_x+core_w, core_y),
                        (core_x+core_w, core_y+core_d), (core_x, core_y+core_d)],
                       dxfattribs={'layer': 'A-CORE', 'color': 4}, close=True)
    n_steps = 10
    for i in range(1, n_steps):
        ty = core_y + (core_d / n_steps) * i
        msp.add_line((core_x, ty), (core_x+core_w, ty),
                     dxfattribs={'layer': 'A-CORE', 'color': 4})
    draw_label(msp, core_x + core_w/2, core_y - 300, "STAIR", 'A-ROOM-TEXT', 180, 4)

    draw_column_grid_bubbles(msp, x0, y0, L, W, xs, ys, col_size)

    off = 700
    msp.add_line((x0, y0-off-2000), (x1, y0-off-2000), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for px in [x0, x1]:
        msp.add_line((px, y0-off-2100), (px, y0-off-1900), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    draw_label(msp, (x0+x1)/2, y0-off-2500, f"{L/1000:.2f} m", 'ANNO-TEXT', 220, 2)
    msp.add_line((x0-off-2000, y0), (x0-off-2000, y1), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for py in [y0, y1]:
        msp.add_line((x0-off-2100, py), (x0-off-1900, py), dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    draw_label(msp, x0-off-2600, (y0+y1)/2, f"{W/1000:.2f} m", 'ANNO-TEXT', 220, 2)

    draw_label(msp, x0 + L/2, y1 + 3000, "GROUND FLOOR PLAN  —  SCALE 1:100",
               'ANNO-TITLE', 350, 7)

    off_chain = off + 3500
    _draw_dimension_chain(msp, x0, y0, xs, -off_chain, label=f"Overall {L/1000:.2f} m")
    _draw_dimension_chain(msp, x0, y0, xs, -(off_chain + 900))
    _draw_native_dims(msp, xs, ys, x0, y0, offset=off_chain + 3300)
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
    _draw_section_marker(msp, x0 + L / 2, y1 + 500, "A-A", 'up')
    _draw_section_marker(msp, x0 + L / 2, y0 - 500, "B-B", 'down')
    _draw_north_arrow(msp, x0 + L + 2500, y1 - 1000, size=1400)

    # ==================================================================
    # SCHEDULES SHEET (right of plan)
    # ==================================================================
    tb_x = x0 + L + 6000
    tb_y = y1

    draw_label(msp, tb_x + 4000, tb_y + 500, "ROOM SCHEDULE",
               'ANNO-TITLE', 300, 7)
    room_rows = [("No.", "Room Name", "Floor", "Area m2", "Perim", "Floor Fin.", "Wall Fin.")]
    for r in room_schedule:
        room_rows.append(tuple(str(x) for x in r))
    draw_table(msp, tb_x, tb_y, [600, 1800, 900, 900, 800, 1200, 1200], room_rows)

    door_y = tb_y - (len(room_rows) + 2) * 350 - 500
    draw_label(msp, tb_x + 3000, door_y + 500, "DOOR SCHEDULE",
               'ANNO-TITLE', 300, 7)
    door_rows = [("Mark", "Type", "W (mm)", "H (mm)", "Qty", "Location")]
    for r in door_marks:
        door_rows.append(tuple(str(x) for x in r))
    draw_table(msp, tb_x, door_y, [700, 1500, 800, 800, 500, 2000], door_rows)

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
    # BOQ SHEET (below schedule)
    # ==================================================================
    bx = 0
    by = win_y - (len(win_rows) + 2) * 350 - 2000

    draw_label(msp, bx + 5000, by + 500, "BILL OF QUANTITIES (EGP)",
               'ANNO-TITLE', 300, 7)
    num_cols = len(cols)
    slab_vol = (L/1000)*(W/1000)*(0.14)*num_floors
    col_vol = num_cols * (col_size/1000)**2 * floor_h_m * num_floors
    beam_len_total = 0
    for _ in range(nx-1):
        beam_len_total += (ny) * (xs[1]-xs[0])/1000
    for _ in range(ny-1):
        beam_len_total += (nx) * (ys[1]-ys[0])/1000
    beam_vol = beam_len_total * 0.25 * 0.6 * num_floors
    foot_vol = num_cols * 1.2**2 * 0.5
    total_concrete = slab_vol + col_vol + beam_vol + foot_vol
    rebar_ton = total_concrete * 0.110
    formwork = beam_len_total * 0.6 * 2 * num_floors + col_vol * 8
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
             ("Windows Aluminum", "nos", max(0, len(win_rows)-1), 2000),
             ("Doors Wood", "nos", max(0, len(door_rows)-1), 3000)]
    grand = 0
    for name, unit, qty, rate in rates:
        amount = qty * rate
        grand += amount
        boq_rows.append((name, unit, f"{qty:.1f}", f"{rate:,.0f}", f"{amount:,.0f}"))
    boq_rows.append(("GRAND TOTAL", "", "", "", f"{grand:,.0f}"))
    draw_table(msp, bx, by, [2600, 800, 900, 1000, 1600], boq_rows)

    ny_y = by - (len(boq_rows) + 2) * 350 - 500
    draw_label(msp, bx + 3000, ny_y + 500, "GENERAL NOTES",
               'ANNO-TITLE', 300, 7)
    notes = [
        "1. All dimensions are in millimetres unless noted otherwise.",
        "2. Concrete grade: C30/37 for columns & beams, C25/30 for slabs.",
        f"3. Column size: {col_size} x {col_size} mm.",
        "4. Reinforcement: Grade 400/600 per ECP 203.",
        "5. Setbacks per Egyptian Building Law 119/2008.",
        f"6. Total plot area: {plot_area:.1f} m2. Coverage: {coverage*100:.0f}%.",
        f"7. Max permitted floors: {max_floors} (street width {sw} m).",
    ]
    notes_rows = [("Note",)] + [(n,) for n in notes]
    draw_table(msp, bx, ny_y, [9000], notes_rows, row_h=400)

    # ==================================================================
    # TITLE BLOCK
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

    # ---- Write DXF ----
    dxf_buf = io.BytesIO()
    doc.write(dxf_buf, fmt='bin')
    dxf_bytes = dxf_buf.getvalue()
    print(f"[build] DXF written, size={len(dxf_bytes)} bytes, fmt=BIN")

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
    print("[build] === build_complete_project END ===")
    return {'dxf': dxf_bytes, 'boq': boq_df.to_dict('records'), 'info': layout_info}


# ======================================================================
# ADDITIVE HIGH-TRAFFIC LAYER
# ======================================================================
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


def _extract_areas_from_dxf_worker(doc_bytes, unit, workflow):
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    return extract_areas_from_dxf(doc, unit=unit, workflow=workflow)


def _detect_dxf_layers_worker(doc_bytes):
    doc = _open_dxf_doc_from_bytes(doc_bytes)
    result = detect_dxf_layers(doc)
    for layer, info in result.items():
        if isinstance(info.get('types'), set):
            info['types'] = sorted(info['types'])
    return result


async def detect_dxf_layers_async(doc_bytes):
    from config import cpu_bound_limited
    return await cpu_bound_limited(_detect_dxf_layers_worker, doc_bytes)


async def extract_areas_from_dxf_async(doc_bytes, unit='mm', workflow='architectural'):
    from config import cpu_bound_limited
    return await cpu_bound_limited(_extract_areas_from_dxf_worker,
                                   doc_bytes, unit, workflow)


async def build_complete_project_async(params):
    from config import cpu_bound_limited
    return await cpu_bound_limited(build_complete_project, params)
