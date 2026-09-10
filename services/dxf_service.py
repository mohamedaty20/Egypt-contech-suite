# services/dxf_service.py — Complete fixed version
"""
DXF Generator Service — Smart EGY-CIVIL AI
Generates PRELIMINARY architectural + structural DXF (R2010).

Compliance note:
  This generator uses simplified rule-of-thumb rules derived from the
  Egyptian Code of Practice (ECP 203 for concrete, ECP 202 for foundations).
  It is NOT a substitute for a licensed engineer's design. All dimensions
  and reinforcement MUST be verified by a qualified structural engineer
  before construction.

Interface:
  build_complete_project(params) -> {'dxf': bytes, 'boq': list[dict], 'info': dict}
"""
import io
import math
import numpy as np
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, box

# ----------------------------------------------------------------------
# Optional config import — fall back to sensible defaults
# ----------------------------------------------------------------------
try:
    from config import BOQ_RATES
except Exception:
    BOQ_RATES = {
        'concrete_m3':  2500.0,
        'rebar_ton':    45000.0,
        'formwork_m2':  350.0,
        'bricks_nos':   3.0,
        'flooring_m2':  220.0,
        'paint_m2':     45.0,
    }

# ----------------------------------------------------------------------
# Valid DXF lineweights (hundredths of mm)
# ----------------------------------------------------------------------
_VALID_LW = [0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60,
             70, 80, 90, 100, 106, 120, 140, 158, 200, 211]

def _lw(v):
    try:
        v = int(v)
    except Exception:
        return 25
    if v in _VALID_LW:
        return v
    return min(_VALID_LW, key=lambda a: abs(a - v))

# ======================================================================
# Layer helpers
# ======================================================================
def create_dxf_layer(doc, name, color=7, lineweight=25, linetype='Continuous'):
    """Create a layer if it doesn't exist. lineweight is in 1/100 mm."""
    if name in doc.layers:
        return doc.layers.get(name)
    try:
        layer = doc.layers.add(name=name, color=color, linetype=linetype)
    except Exception:
        layer = doc.layers.add(name=name, color=color)
    try:
        layer.dxf.lineweight = _lw(lineweight)
    except Exception:
        pass
    return layer

def detect_dxf_layers(doc):
    """Analyse an uploaded DXF and return a per-layer summary."""
    layers = {}
    try:
        for entity in doc.modelspace():
            lname = entity.dxf.layer
            if lname not in layers:
                layers[lname] = {'count': 0, 'types': set(), 'keywords': []}
            layers[lname]['count'] += 1
            layers[lname]['types'].add(entity.dxftype())
    except Exception as e:
        print(f"[DXF] layer scan error: {e}")
        return {}
    for lname in layers:
        lc = lname.lower()
        kws = []
        for kw in ('wall', 'column', 'beam', 'slab', 'footing',
                   'room', 'door', 'window', 'area'):
            if kw in lc:
                kws.append(kw)
        layers[lname]['keywords'] = kws
    return layers

def extract_areas_from_dxf(doc, unit='mm', workflow='architectural'):
    """Extract closed-polyline areas from an uploaded DXF."""
    if unit == 'mm':
        area_scale = 1e-6
    elif unit == 'cm':
        area_scale = 1e-4
    else:
        area_scale = 1.0

    results = []
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() not in ('LWPOLYLINE', 'POLYLINE'):
            continue
        if not getattr(entity, 'closed', False):
            continue
        try:
            if entity.dxftype() == 'LWPOLYLINE':
                pts = [(p[0], p[1]) for p in entity.get_points()]
            else:
                pts = [(v.dxf.location.x, v.dxf.location.y)
                       for v in entity.vertices]
            if len(pts) < 3:
                continue
            # shoelace
            area = 0.0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2.0
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            label = ''
            for txt in msp.query('TEXT MTEXT'):
                try:
                    pos = txt.dxf.insert
                    if abs(pos.x - cx) < 15 and abs(pos.y - cy) < 15:
                        label = getattr(txt.dxf, 'text', '') or ''
                        break
                except Exception:
                    continue
            results.append({
                'layer':   entity.dxf.layer,
                'area_m2': round(area * area_scale, 4),
                'label':   label.strip(),
                'vertices': len(pts),
            })
        except Exception as e:
            print(f"[DXF] skip entity: {e}")
            continue
    return results

# ======================================================================
# Low-level drawing helpers
# ======================================================================
def _text(msp, s, pos, layer='ANNO-TEXT', h=200, color=None,
          align='center', rotation=0.0):
    attrs = {'layer': layer, 'height': h, 'rotation': rotation}
    if color is not None:
        attrs['color'] = color
    t = msp.add_text(str(s), dxfattribs=attrs)
    amap = {
        'center': TextEntityAlignment.MIDDLE_CENTER,
        'left':   TextEntityAlignment.MIDDLE_LEFT,
        'right':  TextEntityAlignment.MIDDLE_RIGHT,
        'top':    TextEntityAlignment.TOP_CENTER,
        'bottom': TextEntityAlignment.BOTTOM_CENTER,
    }
    t.set_placement(pos, align=amap.get(align, TextEntityAlignment.MIDDLE_CENTER))
    return t

def _line(msp, p1, p2, layer, color=None, lw=None):
    attrs = {'layer': layer}
    if color is not None:
        attrs['color'] = color
    if lw is not None:
        attrs['lineweight'] = _lw(lw)
    msp.add_line(p1, p2, dxfattribs=attrs)

def _rect(msp, x, y, w, h, layer, color=None, lw=None, closed=True):
    attrs = {'layer': layer}
    if color is not None:
        attrs['color'] = color
    if lw is not None:
        attrs['lineweight'] = _lw(lw)
    msp.add_lwpolyline(
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)],
        dxfattribs=attrs, close=closed
    )

def _circle(msp, c, r, layer, color=None):
    attrs = {'layer': layer}
    if color is not None:
        attrs['color'] = color
    msp.add_circle(c, r, dxfattribs=attrs)

def _arc(msp, c, r, a0, a1, layer, color=None):
    attrs = {'layer': layer}
    if color is not None:
        attrs['color'] = color
    msp.add_arc(c, r, a0, a1, dxfattribs=attrs)

def _solid(msp, pts, layer, color=None):
    attrs = {'layer': layer}
    if color is not None:
        attrs['color'] = color
    msp.add_solid(pts, dxfattribs=attrs)

def _dim_h(msp, x1, x2, y, label=None, layer='ANNO-DIMS', h=180):
    _line(msp, (x1, y), (x1, y + 200), layer)
    _line(msp, (x2, y), (x2, y + 200), layer)
    _line(msp, (x1, y), (x2, y), layer)
    _line(msp, (x1 - 50, y - 50), (x1 + 50, y + 50), layer)
    _line(msp, (x2 - 50, y - 50), (x2 + 50, y + 50), layer)
    txt = label if label is not None else f"{abs(x2 - x1)/1000:.2f}"
    _text(msp, txt, ((x1 + x2) / 2, y + 150), layer=layer, h=h)

def _dim_v(msp, y1, y2, x, label=None, layer='ANNO-DIMS', h=180):
    _line(msp, (x, y1), (x + 200, y1), layer)
    _line(msp, (x, y2), (x + 200, y2), layer)
    _line(msp, (x, y1), (x, y2), layer)
    _line(msp, (x - 50, y1 - 50), (x + 50, y1 + 50), layer)
    _line(msp, (x - 50, y2 - 50), (x + 50, y2 + 50), layer)
    txt = label if label is not None else f"{abs(y2 - y1)/1000:.2f}"
    _text(msp, txt, (x + 180, (y1 + y2) / 2), layer=layer, h=h, rotation=90)

# ======================================================================
# Squarified room placement (better aspect ratios than naive slicing)
# ======================================================================
def _worst_ratio(row, w):
    if not row:
        return float('inf')
    s = sum(row)
    if s <= 0 or w <= 0:
        return float('inf')
    return max(max(w * w * r / (s * s), s * s / (w * w * r)) for r in row)

def _layout_row(row, x, y, w, h, horizontal):
    """Return placements for a row of rooms. Returns (placed, remaining_rect)."""
    s = sum(row)
    if horizontal:
        row_h = s / w if w > 0 else 0
        placed, cx = [], x
        for r in row:
            rw = r / row_h if row_h > 0 else 0
            placed.append((cx, y, cx + rw, y + row_h))
            cx += rw
        return placed, (x, y + row_h, x + w, y + h)
    else:
        row_w = s / h if h > 0 else 0
        placed, cy = [], y
        for r in row:
            rh = r / row_w if row_w > 0 else 0
            placed.append((x, cy, x + row_w, cy + rh))
            cy += rh
        return placed, (x + row_w, y, x + w, y + h)

def _squarify_rooms(rect, areas):
    """Return list of sub-rectangles matching areas, using squarified treemap."""
    x, y, x1, y1 = rect
    w, h = x1 - x, y1 - y
    if not areas or w <= 0 or h <= 0:
        return []

    total = sum(areas)
    if total <= 0:
        return []

    # Normalise to rectangle area
    scale = (w * h) / total
    norm = [a * scale for a in areas]

    out, row = [], []
    cx, cy, cw, ch = x, y, w, h

    def orient_horizontal(rect_w, rect_h):
        return rect_w >= rect_h

    while norm:
        horiz = orient_horizontal(cw, ch)
        cand = row + [norm[0]]
        base = ch if horiz else cw
        # worst before / after
        if row:
            r_before = _worst_ratio(row, base)
            r_after  = _worst_ratio(cand, base)
            if r_after > r_before:
                placed, (cx, cy, cw, ch) = _layout_row(row, cx, cy, cw, ch, horiz)
                out.extend(placed)
                row = []
                continue
        row = cand
        norm.pop(0)

    if row:
        horiz = orient_horizontal(cw, ch)
        placed, _ = _layout_row(row, cx, cy, cw, ch, horiz)
        out.extend(placed)
    return out

# ======================================================================
# Architectural plan
# ======================================================================
def _draw_arch_plan(msp, ox, oy, params, plot_w_mm, plot_l_mm,
                    bldg_w_mm, bldg_l_mm, rooms, door_w, win_w,
                    grid_x, grid_y, grid_labels_x, grid_labels_y):
    """Draw full architectural floor plan with origin at (ox, oy) = plot bottom-left."""

    ext_t = 250.0     # external wall thickness mm
    int_t = 120.0     # internal wall thickness mm
    corridor = 1100.0 # corridor width mm

    # ---- plot boundary ----
    _rect(msp, ox, oy, plot_w_mm, plot_l_mm, 'A-PLOT', lw=20)

    # ---- grid bubbles (drawn outside building) ----
    bubble_r = 350.0
    for i, gx in enumerate(grid_x):
        X = ox + gx
        _line(msp, (X, oy - 200), (X, oy + plot_l_mm + 400), 'A-GRID', lw=9)
        _circle(msp, (X, oy + plot_l_mm + bubble_r + 200), bubble_r, 'A-GRID')
        _text(msp, grid_labels_x[i],
              (X, oy + plot_l_mm + bubble_r + 200), layer='A-GRID', h=250)
    for j, gy in enumerate(grid_y):
        Y = oy + gy
        _line(msp, (ox - 200, Y), (ox + plot_w_mm + 400, Y), 'A-GRID', lw=9)
        _circle(msp, (ox + plot_w_mm + bubble_r + 200, Y), bubble_r, 'A-GRID')
        _text(msp, grid_labels_y[j],
              (ox + plot_w_mm + bubble_r + 200, Y), layer='A-GRID', h=250)

    # ---- building footprint (offset from plot edge by setback) ----
    # Compute building origin inside plot:
    sx = (plot_w_mm - bldg_w_mm) / 2      # centred left-right
    sy = 0.0                               # will set based on setback below
    # front setback taken from params' original setbacks already; assume bottom
    # edge of building = bottom edge of plot + front setback (already accounted
    # in bldg_l_mm). We'll use a small interior margin:
    margin = 0.0
    bx = ox + sx + margin
    by = oy + (plot_l_mm - bldg_l_mm) / 2

    # Outer rectangle (external wall outer face)
    _rect(msp, bx, by, bldg_w_mm, bldg_l_mm, 'A-WALL', lw=50)
    # Inner rectangle (external wall inner face)
    _rect(msp, bx + ext_t, by + ext_t,
          bldg_w_mm - 2 * ext_t, bldg_l_mm - 2 * ext_t, 'A-WALL', lw=50)

    # Internal usable area
    in_x = bx + ext_t
    in_y = by + ext_t
    in_w = bldg_w_mm - 2 * ext_t
    in_h = bldg_l_mm - 2 * ext_t

    # ---- room layout ----
    # Split into two strips separated by a horizontal corridor
    lower_h = (in_h - corridor) * 0.45
    upper_h = in_h - corridor - lower_h

    lower_rect = (in_x, in_y, in_x + in_w, in_y + lower_h)
    corridor_rect = (in_x, in_y + lower_h, in_x + in_w, in_y + lower_h + corridor)
    upper_rect = (in_x, in_y + lower_h + corridor, in_x + in_w, in_y + in_h)

    public = [r for r in rooms if r.get('zone', 'public') == 'public']
    private = [r for r in rooms if r.get('zone', 'public') == 'private']
    if not private:
        private = public[len(public)//2:]
        public = public[:len(public)//2]

    placements = []
    if public:
        pub_areas = [max(r.get('area_m2', 10), 4) for r in public]
        rects = _squarify_rooms(lower_rect, pub_areas)
        placements += list(zip(public, rects))
    if private:
        prv_areas = [max(r.get('area_m2', 10), 4) for r in private]
        rects = _squarify_rooms(upper_rect, prv_areas)
        placements += list(zip(private, rects))

    # ---- corridor label ----
    _text(msp, 'CORRIDOR',
          ((corridor_rect[0] + corridor_rect[2]) / 2,
           (corridor_rect[1] + corridor_rect[3]) / 2),
          layer='A-ROOM', h=200)

    # ---- per-room drawing ----
    for room, (rx0, ry0, rx1, ry1) in placements:
        # internal wall rectangle
        _rect(msp, rx0, ry0, rx1 - rx0, ry1 - ry0, 'A-WALL-INT', lw=30)

        cx = (rx0 + rx1) / 2
        cy = (ry0 + ry1) / 2
        name = room.get('name', 'ROOM')
        w = (rx1 - rx0) / 1000.0
        h = (ry1 - ry0) / 1000.0
        _text(msp, name.upper(), (cx, cy + 200), layer='A-ROOM', h=230)
        _text(msp, f"{w:.2f} x {h:.2f} m", (cx, cy - 30), layer='A-ROOM', h=190)
        _text(msp, f"A = {w*h:.2f} m2", (cx, cy - 240), layer='A-ROOM', h=190)

        # ----- door (on wall facing corridor) -----
        # corridor is in the middle → door is on wall closest to mid
        mid_y = in_y + lower_h + corridor / 2
        if ry1 < mid_y:
            # room is below corridor → door on top wall
            dx = cx
            dy = ry1
            _line(msp, (dx - door_w/2, dy), (dx + door_w/2, dy),
                  'A-WALL', color=7, lw=50)  # close wall gap with thicker line
            # draw opening (erase wall by drawing gap): we instead break the wall
            # by drawing two short segments over it — this is a simplification
            # (AutoCAD will show the door outline on top of the wall line)
            _line(msp, (dx - door_w/2, dy), (dx + door_w/2, dy),
                  'A-WALL-INT', color=0, lw=0)
            # door leaf + swing arc
            hinge = (dx - door_w/2, dy)
            _line(msp, hinge, (hinge[0], hinge[1] + door_w), 'A-DOOR', lw=18)
            _arc(msp, hinge, door_w, 0, 90, 'A-DOOR')
        else:
            # room is above corridor → door on bottom wall
            dx = cx
            dy = ry0
            _line(msp, (dx - door_w/2, dy), (dx + door_w/2, dy),
                  'A-WALL-INT', color=0, lw=0)
            hinge = (dx + door_w/2, dy)
            _line(msp, hinge, (hinge[0], hinge[1] - door_w), 'A-DOOR', lw=18)
            _arc(msp, hinge, door_w, 90, 180, 'A-DOOR')

        # ----- windows (on external wall only) -----
        if room.get('needs_window'):
            # is the room touching the top external wall?
            if abs(ry1 - (in_y + in_h)) < 50:
                wx = cx
                wy = by + bldg_l_mm - ext_t / 2
                _line(msp, (wx - win_w/2, wy - 60), (wx + win_w/2, wy - 60),
                      'A-WINDW', lw=15)
                _line(msp, (wx - win_w/2, wy),       (wx + win_w/2, wy),
                      'A-WINDW', lw=15)
                _line(msp, (wx - win_w/2, wy + 60), (wx + win_w/2, wy + 60),
                      'A-WINDW', lw=15)
            # is the room touching the bottom external wall?
            if abs(ry0 - in_y) < 50:
                wx = cx
                wy = by + ext_t / 2
                _line(msp, (wx - win_w/2, wy - 60), (wx + win_w/2, wy - 60),
                      'A-WINDW', lw=15)
                _line(msp, (wx - win_w/2, wy),       (wx + win_w/2, wy),
                      'A-WINDW', lw=15)
                _line(msp, (wx - win_w/2, wy + 60), (wx + win_w/2, wy + 60),
                      'A-WINDW', lw=15)
            # side windows
            if abs(rx0 - in_x) < 50:
                wy = cy
                wx = bx + ext_t / 2
                _line(msp, (wx - 60, wy - win_w/2), (wx - 60, wy + win_w/2),
                      'A-WINDW', lw=15)
                _line(msp, (wx,      wy - win_w/2), (wx,      wy + win_w/2),
                      'A-WINDW', lw=15)
                _line(msp, (wx + 60, wy - win_w/2), (wx + 60, wy + win_w/2),
                      'A-WINDW', lw=15)
            if abs(rx1 - (in_x + in_w)) < 50:
                wy = cy
                wx = bx + bldg_w_mm - ext_t / 2
                _line(msp, (wx - 60, wy - win_w/2), (wx - 60, wy + win_w/2),
                      'A-WINDW', lw=15)
                _line(msp, (wx,      wy - win_w/2), (wx,      wy + win_w/2),
                      'A-WINDW', lw=15)
                _line(msp, (wx + 60, wy - win_w/2), (wx + 60, wy + win_w/2),
                      'A-WINDW', lw=15)

    # ---- entrance door on front wall ----
    ex = bx + bldg_w_mm / 2
    ey = by
    _line(msp, (ex - door_w/2, ey - 60), (ex + door_w/2, ey - 60), 'A-WALL', color=7, lw=50)
    _line(msp, (ex - door_w/2, ey + 60), (ex + door_w/2, ey + 60), 'A-WALL', color=7, lw=50)
    hinge = (ex - door_w/2, ey)
    _line(msp, hinge, (hinge[0], hinge[1] + door_w), 'A-DOOR', lw=18)
    _arc(msp, hinge, door_w, 0, 90, 'A-DOOR')

    # ---- dimension lines ----
    _dim_h(msp, bx, bx + bldg_w_mm, by - 800)
    _dim_v(msp, by, by + bldg_l_mm, bx - 800)

    # individual grid spacing dims along bottom
    for i in range(len(grid_x) - 1):
        _dim_h(msp, ox + grid_x[i], ox + grid_x[i+1], by - 1800)
    for j in range(len(grid_y) - 1):
        _dim_v(msp, oy + grid_y[j], oy + grid_y[j+1], bx - 1800)

    # ---- north arrow ----
    nx = ox + plot_w_mm + bubble_r * 2 + 1500
    ny = oy + plot_l_mm - 1500
    _line(msp, (nx, ny - 600), (nx, ny + 600), 'A-ANNO', lw=30)
    _solid(msp, [(nx, ny + 800),
                 (nx - 200, ny + 400),
                 (nx + 200, ny + 400)], 'A-ANNO')
    _text(msp, 'N', (nx, ny + 1200), layer='A-ANNO', h=280)

    # ---- plan title ----
    _text(msp, 'ARCHITECTURAL FLOOR PLAN',
          (bx + bldg_w_mm / 2, oy + plot_l_mm + 2600),
          layer='A-ANNO', h=420)
    _text(msp, 'PRELIMINARY — NOT FOR CONSTRUCTION  |  ECP 203 / ECP 202',
          (bx + bldg_w_mm / 2, oy + plot_l_mm + 2100),
          layer='A-ANNO', h=220, color=1)

    return (bx, by, bldg_w_mm, bldg_l_mm)

# ======================================================================
# Structural plan
# ======================================================================
def _draw_struct_plan(msp, ox, oy, params, plot_w_mm, plot_l_mm,
                      bldg_w_mm, bldg_l_mm,
                      grid_x, grid_y, grid_labels_x, grid_labels_y,
                      col_size_edge, col_size_int,
                      beam_b, beam_d, slab_t):
    """Draw structural framing plan. Returns (column list, beam list, slab list)."""

    sx = (plot_w_mm - bldg_w_mm) / 2
    sy = (plot_l_mm - bldg_l_mm) / 2
    bx = ox + sx
    by = oy + sy

    # building outline (thin, dashed outside)
    _rect(msp, bx, by, bldg_w_mm, bldg_l_mm, 'S-SLAB', lw=13)

    # grid bubbles
    br = 350.0
    for i, gx in enumerate(grid_x):
        X = bx + gx
        _line(msp, (X, by - 300), (X, by + bldg_l_mm + 600), 'S-GRID', lw=9)
        _circle(msp, (X, by + bldg_l_mm + 900), br, 'S-GRID')
        _text(msp, grid_labels_x[i], (X, by + bldg_l_mm + 900),
              layer='S-GRID', h=250)
    for j, gy in enumerate(grid_y):
        Y = by + gy
        _line(msp, (bx - 300, Y), (bx + bldg_w_mm + 600, Y), 'S-GRID', lw=9)
        _circle(msp, (bx + bldg_w_mm + 900, Y), br, 'S-GRID')
        _text(msp, grid_labels_y[j], (bx + bldg_w_mm + 900, Y),
              layer='S-GRID', h=250)

    # ---- columns at every grid intersection ----
    columns = []
    cid = 1
    for i, gx in enumerate(grid_x):
        for j, gy in enumerate(grid_y):
            is_edge = (i == 0 or i == len(grid_x) - 1 or
                       j == 0 or j == len(grid_y) - 1)
            size = col_size_edge if is_edge else col_size_int
            X = bx + gx
            Y = by + gy
            # column outline
            _rect(msp, X - size/2, Y - size/2, size, size, 'S-COLS', lw=40)
            # 4 main bars as dots
            cover = 40.0
            br_r = size/2 - cover - 20
            for dx, dy in [(-br_r, -br_r), (br_r, -br_r),
                           (br_r, br_r), (-br_r, br_r)]:
                _circle(msp, (X + dx, Y + dy), 18, 'S-REBAR')
            # label
            lbl = f"C{cid}"
            _line(msp, (X + size/2, Y + size/2),
                  (X + size/2 + 700, Y + size/2 + 700), 'S-ANNO', lw=9)
            _text(msp, f"{lbl}\n{int(size)}x{int(size)}",
                  (X + size/2 + 800, Y + size/2 + 900),
                  layer='S-ANNO', h=180, align='left')
            columns.append({'id': lbl, 'x': X, 'y': Y,
                            'size': size, 'edge': is_edge})
            cid += 1

    # ---- beams between adjacent columns ----
    beams = []
    bid = 1
    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            X = bx + grid_x[i]
            Y = by + grid_y[j]
            if i < len(grid_x) - 1:
                X2 = bx + grid_x[i + 1]
                # horizontal beam
                _rect(msp, X, Y - beam_b/2, X2 - X, beam_b, 'S-BEAM', lw=30)
                _text(msp, f"B{bid}",
                      ((X + X2) / 2, Y + beam_b/2 + 200),
                      layer='S-BEAM', h=190)
                beams.append({'id': f"B{bid}", 'axis': 'X',
                              'span_m': (X2 - X) / 1000.0,
                              'b': beam_b, 'd': beam_d})
                bid += 1
            if j < len(grid_y) - 1:
                Y2 = by + grid_y[j + 1]
                _rect(msp, X - beam_b/2, Y, beam_b, Y2 - Y, 'S-BEAM', lw=30)
                _text(msp, f"B{bid}",
                      (X + beam_b/2 + 200, (Y + Y2) / 2),
                      layer='S-BEAM', h=190, rotation=90)
                beams.append({'id': f"B{bid}", 'axis': 'Y',
                              'span_m': (Y2 - Y) / 1000.0,
                              'b': beam_b, 'd': beam_d})
                bid += 1

    # ---- slab panels ----
    slabs = []
    sid = 1
    for i in range(len(grid_x) - 1):
        for j in range(len(grid_y) - 1):
            cx = bx + (grid_x[i] + grid_x[i + 1]) / 2
            cy = by + (grid_y[j] + grid_y[j + 1]) / 2
            lbl = f"S{sid}"
            _text(msp, lbl, (cx, cy + 200), layer='S-SLAB', h=240)
            _text(msp, f"h={int(slab_t)}", (cx, cy - 100),
                  layer='S-SLAB', h=180)
            slabs.append({'id': lbl, 'w': (grid_x[i+1]-grid_x[i])/1000.0,
                          'h': (grid_y[j+1]-grid_y[j])/1000.0,
                          't': slab_t})
            sid += 1

    # ---- dimensions ----
    _dim_h(msp, bx, bx + bldg_w_mm, by - 800)
    _dim_v(msp, by, by + bldg_l_mm, bx - 800)

    # ---- title ----
    _text(msp, 'STRUCTURAL FRAMING PLAN',
          (bx + bldg_w_mm / 2, by + bldg_l_mm + 3200),
          layer='S-ANNO', h=420)
    _text(msp, 'PRELIMINARY — ECP 203 rule-of-thumb  |  verify with engineer',
          (bx + bldg_w_mm / 2, by + bldg_l_mm + 2700),
          layer='S-ANNO', h=220, color=1)

    return columns, beams, slabs

# ======================================================================
# Foundation plan
# ======================================================================
def _draw_foundation_plan(msp, ox, oy, columns,
                          bldg_w_mm, bldg_l_mm,
                          grid_x, grid_y, grid_labels_x, grid_labels_y,
                          foot_edge, foot_int, foot_d):
    """Draw foundation plan. Returns list of footings."""
    # local translation: use min column coords as origin
    xs = [c['x'] for c in columns]
    ys = [c['y'] for c in columns]
    bx0 = min(xs)
    by0 = min(ys)

    # grid
    br = 350.0
    for i, gx in enumerate(grid_x):
        X = bx0 + (gx - grid_x[0])   # already offset? see columns' coordinates
    # The above doesn't work cleanly because gx are relative to building origin.
    # Simpler: derive grid positions from columns list.
    # We'll just use columns' absolute positions to place grid.
    uniq_x = sorted(set(round(c['x'], 1) for c in columns))
    uniq_y = sorted(set(round(c['y'], 1) for c in columns))
    for i, X in enumerate(uniq_x):
        _line(msp, (X, by0 - 500), (X, oy + bldg_l_mm + 500), 'S-GRID', lw=9)
        _circle(msp, (X, oy + bldg_l_mm + 900), br, 'S-GRID')
        _text(msp, grid_labels_x[i] if i < len(grid_labels_x) else str(i + 1),
              (X, oy + bldg_l_mm + 900), layer='S-GRID', h=250)
    for j, Y in enumerate(uniq_y):
        _line(msp, (ox - 500, Y), (bx0 + bldg_w_mm + 500, Y), 'S-GRID', lw=9)
        _circle(msp, (bx0 + bldg_w_mm + 900, Y), br, 'S-GRID')
        _text(msp, grid_labels_y[j] if j < len(grid_labels_y) else str(j + 1),
              (bx0 + bldg_w_mm + 900, Y), layer='S-GRID', h=250)

    footings = []
    fid = 1
    for c in columns:
        size = foot_edge if c['edge'] else foot_int
        X, Y = c['x'], c['y']
        # footing rectangle
        _rect(msp, X - size/2, Y - size/2, size, size, 'S-FNDN', lw=40)
        # stub column
        _rect(msp, X - c['size']/2, Y - c['size']/2,
              c['size'], c['size'], 'S-COLS', lw=40)
        # rebar dots (mesh)
        n = 5
        for ii in range(n):
            for jj in range(n):
                px = X - size/2 + (ii + 0.5) * size / n
                py = Y - size/2 + (jj + 0.5) * size / n
                _circle(msp, (px, py), 12, 'S-REBAR')
        # label
        lbl = f"F{fid}"
        _line(msp, (X + size/2, Y + size/2),
              (X + size/2 + 700, Y + size/2 + 700), 'S-ANNO', lw=9)
        _text(msp, f"{lbl}\n{int(size)}x{int(size)}x{int(foot_d)}",
              (X + size/2 + 800, Y + size/2 + 900),
              layer='S-ANNO', h=170, align='left')
        footings.append({'id': lbl, 'size': size, 'd': foot_d,
                         'x': X, 'y': Y})
        fid += 1

    # title
    _text(msp, 'FOUNDATION PLAN',
          (ox + bldg_w_mm / 2, oy + bldg_l_mm + 3200),
          layer='S-ANNO', h=420)
    _text(msp, 'PRELIMINARY — ECP 202  |  bearing capacity & settlement check required',
          (ox + bldg_w_mm / 2, oy + bldg_l_mm + 2700),
          layer='S-ANNO', h=220, color=1)

    return footings

# ======================================================================
# Section details
# ======================================================================
def _draw_column_section(msp, ox, oy, size, main_bars, main_dia,
                         stirrup_dia, stirrup_sp, cover):
    """Draw column cross-section with rebar. Size/dia in mm."""
    _text(msp, f"COLUMN SECTION  {int(size)}x{int(size)}",
          (ox + size/2, oy + size + 700), layer='S-ANNO', h=280)
    _rect(msp, ox, oy, size, size, 'S-COLS', lw=50)
    # stirrup inner rect
    s = cover
    _rect(msp, ox + s, oy + s, size - 2*s, size - 2*s, 'S-REBAR', lw=15)
    # main bars at corners
    r = main_dia / 2
    c = cover + r + 5
    for dx, dy in [(c, c), (size - c, c), (size - c, size - c), (c, size - c)]:
        _circle(msp, (ox + dx, oy + dy), r, 'S-REBAR')
    # side bars (intermediate) — for size > 300 use 2 extra on each face
    if size > 300:
        for dx in (c, size - c):
            for frac in (0.5,):
                _circle(msp, (ox + dx, oy + size * frac), r, 'S-REBAR')
        for dy in (c, size - c):
            for frac in (0.5,):
                _circle(msp, (ox + size * frac, oy + dy), r, 'S-REBAR')
    # annotations
    _text(msp, f"{main_bars}Ø{int(main_dia)} main",
          (ox + size/2, oy - 300), layer='S-ANNO', h=170)
    _text(msp, f"Ø{int(stirrup_dia)} @ {int(stirrup_sp)} c/c stirrups",
          (ox + size/2, oy - 550), layer='S-ANNO', h=170)
    _text(msp, f"cover = {int(cover)} mm  (ECP 203)",
          (ox + size/2, oy - 800), layer='S-ANNO', h=170, color=1)

def _draw_beam_section(msp, ox, oy, b, d, top_n, top_d, bot_n, bot_d,
                       stirrup_dia, stirrup_sp, cover):
    _text(msp, f"BEAM SECTION  {int(b)}x{int(d)}",
          (ox + b/2, oy + d + 700), layer='S-ANNO', h=280)
    _rect(msp, ox, oy, b, d, 'S-BEAM', lw=50)
    _rect(msp, ox + cover, oy + cover,
          b - 2*cover, d - 2*cover, 'S-REBAR', lw=15)
    r_t = top_d / 2
    r_b = bot_d / 2
    # top bars
    span = b - 2*cover
    for k in range(top_n):
        fx = (k + 1) / (top_n + 1)
        px = ox + cover + span * fx
        _circle(msp, (px, oy + d - cover - r_t - 3), r_t, 'S-REBAR')
    # bottom bars
    for k in range(bot_n):
        fx = (k + 1) / (bot_n + 1)
        px = ox + cover + span * fx
        _circle(msp, (px, oy + cover + r_b + 3), r_b, 'S-REBAR')
    _text(msp, f"TOP {top_n}Ø{int(top_d)}  |  BOT {bot_n}Ø{int(bot_d)}",
          (ox + b/2, oy - 300), layer='S-ANNO', h=170)
    _text(msp, f"Ø{int(stirrup_dia)} @ {int(stirrup_sp)} c/c",
          (ox + b/2, oy - 550), layer='S-ANNO', h=170)

def _draw_footing_section(msp, ox, oy, w, d, cover,
                          main_dia, main_sp):
    """Draw footing cross-section with bottom mesh and column stub."""
    _text(msp, f"FOOTING SECTION  {int(w)}x{int(d)}",
          (ox + w/2, oy + d + 1600), layer='S-ANNO', h=280)
    # footing rect
    _rect(msp, ox, oy, w, d, 'S-FNDN', lw=50)
    # bottom mesh (two lines)
    _line(msp, (ox + cover, oy + cover),
          (ox + w - cover, oy + cover), 'S-REBAR', lw=15)
    _line(msp, (ox + cover, oy + cover + 25),
          (ox + w - cover, oy + cover + 25), 'S-REBAR', lw=15)
    # stub column above
    stub_w = 300
    stub_h = 700
    _rect(msp, ox + w/2 - stub_w/2, oy + d, stub_w, stub_h, 'S-COLS', lw=40)
    # stub rebar
    for dx in (-1, 1):
        px = ox + w/2 + dx * (stub_w/2 - cover - 8)
        _line(msp, (px, oy + d + 50), (px, oy + d + stub_h - 20),
              'S-REBAR', lw=15)
    _text(msp, f"Ø{int(main_dia)} @ {int(main_sp)} c/c both ways",
          (ox + w/2, oy - 300), layer='S-ANNO', h=170)
    _text(msp, f"cover = {int(cover)} mm  (ECP 202)",
          (ox + w/2, oy - 550), layer='S-ANNO', h=170, color=1)

# ======================================================================
# Tables / schedules / BOQ
# ======================================================================
def _draw_table(msp, ox, oy, title, headers, rows,
                col_widths, row_h=350, hdr_h=400, layer='S-ANNO'):
    """Draw a table with title above. Returns total height."""
    ncols = len(headers)
    total_w = sum(col_widths)
    # title
    _text(msp, title, (ox + total_w/2, oy + hdr_h + 350),
          layer=layer, h=280)
    # header row
    x = ox
    y_top = oy + hdr_h + row_h * len(rows)
    # header background as border
    _rect(msp, ox, y_top, total_w, hdr_h, layer, lw=25)
    for i, hcell in enumerate(headers):
        _text(msp, hcell, (x + col_widths[i]/2, y_top + hdr_h/2),
              layer=layer, h=170)
        x += col_widths[i]
    # rows
    for j, row in enumerate(rows):
        y_row = y_top - (j + 1) * row_h
        _rect(msp, ox, y_row, total_w, row_h, layer, lw=15)
        x = ox
        for i, cell in enumerate(row):
            _text(msp, str(cell), (x + col_widths[i]/2, y_row + row_h/2),
                  layer=layer, h=150)
            x += col_widths[i]
    # vertical dividers
    x = ox
    for w in col_widths[:-1]:
        x += w
        _line(msp, (x, y_top - row_h * len(rows)), (x, y_top + hdr_h),
              layer, lw=15)
    return hdr_h + row_h * (len(rows) + 1) + 600

# ======================================================================
# Title block
# ======================================================================
def _draw_title_block(msp, ox, oy, params, info):
    W, H = 14000, 4500
    _rect(msp, ox, oy, W, H, 'G-TTLB', lw=40)
    # vertical / horizontal splits
    _line(msp, (ox, oy + H*0.55), (ox + W, oy + H*0.55), 'G-TTLB')
    _line(msp, (ox + W*0.62, oy), (ox + W*0.62, oy + H), 'G-TTLB')
    _line(msp, (ox + W*0.62, oy + H*0.55), (ox + W, oy + H*0.55), 'G-TTLB')
    # title
    _text(msp, 'SMART EGY-CIVIL AI',
          (ox + 300, oy + H*0.78), layer='G-TTLB', h=380, align='left')
    _text(msp, 'PRELIMINARY LAYOUT DRAWING — NOT FOR CONSTRUCTION',
          (ox + 300, oy + H*0.62), layer='G-TTLB', h=240, align='left', color=1)
    # project info
    info_txt = (
        f"Project: Residential Building\n"
        f"Plot: {info['plot_area']:.0f} m2    "
        f"Footprint: {info['footprint_area']:.1f} m2\n"
        f"Building: {info['building_length']:.2f} x {info['building_width']:.2f} m    "
        f"Floors: {info['num_floors']} of {info['max_floors']} max\n"
        f"Location: {info['location']}    "
        f"Street width: {info['street_width']} m"
    )
    for k, line in enumerate(info_txt.split('\n')):
        _text(msp, line, (ox + 300, oy + H*0.42 - k*350),
              layer='G-TTLB', h=180, align='left')
    # codes
    _text(msp, 'CODES:',
          (ox + W*0.64, oy + H*0.78), layer='G-TTLB', h=220, align='left')
    _text(msp, 'ECP 203 — R.C. Structures',
          (ox + W*0.64, oy + H*0.66), layer='G-TTLB', h=200, align='left')
    _text(msp, 'ECP 202 — Soil & Foundations',
          (ox + W*0.64, oy + H*0.53), layer='G-TTLB', h=200, align='left')
    # disclaimer
    _text(msp, 'ALL DIMENSIONS & REINFORCEMENT MUST BE VERIFIED BY A',
          (ox + W*0.64, oy + H*0.28), layer='G-TTLB', h=180, align='left', color=1)
    _text(msp, 'LICENSED ENGINEER BEFORE ANY CONSTRUCTION.',
          (ox + W*0.64, oy + H*0.14), layer='G-TTLB', h=180, align='left', color=1)

# ======================================================================
# MAIN GENERATOR
# ======================================================================
def build_complete_project(params):
    """Build a preliminary architectural + structural DXF.

    Returns:
        dict with keys:
          'dxf'  — bytes of the DXF file (UTF-8 encoded ASCII DXF)
          'boq'  — list of dicts with the bill of quantities
          'info' — summary dict
    """

    # ---------- 1. Plot geometry ----------
    if params.get('plot_polygon') is not None:
        coords = list(params['plot_polygon'])
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        try:
            poly = Polygon(coords)
        except Exception:
            poly = None
        if poly is None or not poly.is_valid or poly.area < 1.0:
            area = params.get('plot_area_m2', 200)
            pw = params.get('plot_width') or math.sqrt(area / 1.5)
            pl = area / pw
            poly = box(0, 0, pl, pw)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
    else:
        area = params.get('plot_area_m2', 200)
        pw = params.get('plot_width') or math.sqrt(area / 1.5)
        pl = area / pw
        poly = box(0, 0, pl, pw)

    min_x, min_y, max_x, max_y = poly.bounds
    plot_w_m = max_x - min_x
    plot_l_m = max_y - min_y
    if plot_w_m <= 0 or plot_l_m <= 0:
        plot_w_m, plot_l_m = 10.0, 20.0

    # ---------- 2. Setbacks ----------
    street_w = float(params.get('street_width_m', 10))
    if street_w >= 12:   front_sb = 3.0
    elif street_w >= 8:  front_sb = 2.5
    elif street_w >= 6:  front_sb = 2.0
    else:                front_sb = 1.5
    rear_sb = 2.0
    side_sb = 1.5

    bldg_l_m = max(plot_l_m - front_sb - rear_sb, 5.0)
    bldg_w_m = max(plot_w_m - 2 * side_sb, 5.0)

    # enforce max footprint
    footprint_ratio = 0.60
    max_fp = poly.area * footprint_ratio
    if bldg_l_m * bldg_w_m > max_fp:
        s = math.sqrt(max_fp / (bldg_l_m * bldg_w_m))
        bldg_l_m *= s
        bldg_w_m *= s

    # ---------- 3. Floors ----------
    if street_w >= 12:   max_floors = 4
    elif street_w >= 8:  max_floors = 3
    elif street_w >= 6:  max_floors = 2
    else:                max_floors = 1
    num_floors = min(int(params.get('num_floors', max_floors)), max_floors)
    if num_floors < 1:
        num_floors = 1
    floor_h = float(params.get('floor_height_m', 3.0))

    # ---------- 4. Geometry params ----------
    door_w = float(params.get('door_width_mm', 900))
    win_w  = float(params.get('window_width_mm', 1200))

    beam_b = float(params.get('beam_width_mm', 300))
    beam_d = float(params.get('beam_depth_mm', 500))
    slab_t = float(params.get('slab_thickness_mm', 150))
    col_edge = float(params.get('column_size_edge_mm', 300))
    col_int  = float(params.get('column_size_int_mm', 250))
    foot_edge_w = float(params.get('footing_edge_mm', 1200))
    foot_int_w  = float(params.get('footing_int_mm', 1000))
    foot_d      = float(params.get('footing_depth_mm', 400))
    main_dia    = float(params.get('rebar_main_diam_mm', 16))
    stir_dia    = float(params.get('rebar_stirrup_diam_mm', 8))
    stir_sp     = float(params.get('rebar_spacing_mm', 150))

    # ---------- 5. Units in mm for drawing ----------
    plot_w_mm = plot_w_m * 1000.0
    plot_l_mm = plot_l_m * 1000.0
    bldg_w_mm = bldg_w_m * 1000.0
    bldg_l_mm = bldg_l_m * 1000.0
    floor_h_mm = floor_h * 1000.0

    # ---------- 6. Rooms ----------
    layout_plan = params.get('layout_plan') or {}
    rooms = layout_plan.get('rooms', [])
    if not rooms:
        rooms = [
            {"name": "Living Room",    "zone": "public",  "area_m2": 25, "needs_window": True},
            {"name": "Kitchen",        "zone": "public",  "area_m2": 9,  "needs_window": True},
            {"name": "Master Bedroom", "zone": "private", "area_m2": 16, "needs_window": True},
            {"name": "Bedroom 2",      "zone": "private", "area_m2": 12, "needs_window": True},
            {"name": "Bathroom",       "zone": "private", "area_m2": 4,  "needs_window": False},
        ]
    num_units = int(params.get('num_units_per_floor', 1)) or 1

    # ---------- 7. Grid ----------
    n_bays_x = max(2, int(math.ceil(bldg_w_m / 5.0)))
    n_bays_y = max(2, int(math.ceil(bldg_l_m / 5.0)))
    gx = [i * bldg_w_mm / n_bays_x for i in range(n_bays_x + 1)]
    gy = [j * bldg_l_mm / n_bays_y for j in range(n_bays_y + 1)]
    grid_labels_x = [chr(ord('A') + i) for i in range(n_bays_x + 1)]
    grid_labels_y = [str(i + 1) for i in range(n_bays_y + 1)]

    # ---------- 8. DXF setup ----------
    doc = ezdxf.new(dxfversion='R2010', setup=True)
    msp = doc.modelspace()
    doc.header['$INSUNITS'] = 4  # mm

    layers = {
        'A-PLOT':      {'color': 8,  'lw': 20},
        'A-GRID':      {'color': 8,  'lw': 9},
        'A-WALL':      {'color': 7,  'lw': 50},
        'A-WALL-INT':  {'color': 8,  'lw': 30},
        'A-DOOR':      {'color': 3,  'lw': 18},
        'A-WINDW':     {'color': 5,  'lw': 15},
        'A-ROOM':      {'color': 4,  'lw': 13},
        'A-ANNO':      {'color': 2,  'lw': 15},
        'S-GRID':      {'color': 8,  'lw': 9},
        'S-COLS':      {'color': 1,  'lw': 40},
        'S-BEAM':      {'color': 3,  'lw': 30},
        'S-SLAB':      {'color': 9,  'lw': 15},
        'S-FNDN':      {'color': 6,  'lw': 40},
        'S-REBAR':     {'color': 10, 'lw': 13},
        'S-ANNO':      {'color': 4,  'lw': 15},
        'ANNO-DIMS':   {'color': 2,  'lw': 15},
        'ANNO-TEXT':   {'color': 4,  'lw': 13},
        'G-TTLB':      {'color': 7,  'lw': 40},
    }
    for name, spec in layers.items():
        create_dxf_layer(doc, name, color=spec['color'], lineweight=spec['lw'])

    # ---------- 9. ZONES ----------
    zone_gap = 8000.0

    # Zone 1: architectural plan at origin
    arch_ox, arch_oy = 0.0, 0.0

    # Zone 2: structural plan to the right
    struct_ox = plot_w_mm + zone_gap + 2000.0
    struct_oy = 0.0

    # Zone 3: foundation plan below structural
    found_ox = struct_ox
    found_oy = -(plot_l_mm + zone_gap)

    # Zone 4: schedules to the right of structural plan
    sched_ox = struct_ox + plot_w_mm + zone_gap
    sched_oy = 0.0

    # Zone 5: sections below architectural plan
    sec_ox = 0.0
    sec_oy = -(plot_l_mm + zone_gap)

    # Zone 6: BOQ below sections
    boq_ox = 0.0
    boq_oy = sec_oy - 6500.0

    # Zone 7: title block
    tb_ox = sched_ox
    tb_oy = -(plot_l_mm + zone_gap)

    # ---------- 10. DRAW ----------
    _draw_arch_plan(
        msp, arch_ox, arch_oy, params,
        plot_w_mm, plot_l_mm, bldg_w_mm, bldg_l_mm,
        rooms, door_w, win_w,
        gx, gy, grid_labels_x, grid_labels_y
    )

    columns, beams, slabs = _draw_struct_plan(
        msp, struct_ox, struct_oy, params,
        plot_w_mm, plot_l_mm, bldg_w_mm, bldg_l_mm,
        gx, gy, grid_labels_x, grid_labels_y,
        col_edge, col_int, beam_b, beam_d, slab_t
    )

    # shift columns for foundation plan
    # The struct plan placed columns at struct_ox + bx + gx etc. Because foundation
    # plan uses a different origin, we simply re-draw there with a rigid translation
    # of -struct_ox + found_ox + offset in X and Y
    dx = found_ox - struct_ox
    dy = found_oy - struct_oy
    shifted_cols = [{'id': c['id'], 'x': c['x'] + dx, 'y': c['y'] + dy,
                     'size': c['size'], 'edge': c['edge']} for c in columns]
    footings = _draw_foundation_plan(
        msp, found_ox, found_oy, shifted_cols,
        bldg_w_mm, bldg_l_mm,
        gx, gy, grid_labels_x, grid_labels_y,
        foot_edge_w, foot_int_w, foot_d
    )

    # Sections (drawn near sec_ox, sec_oy)
    s1_x = sec_ox + 500
    s1_y = sec_oy - 1500
    _draw_column_section(msp, s1_x, s1_y, col_edge, 4, main_dia,
                         stir_dia, stir_sp, 40)
    _draw_beam_section(msp, s1_x + 2500, s1_y, beam_b, beam_d,
                       2, main_dia, 3, main_dia, stir_dia,
                       stir_sp * 2, 40)
    _draw_footing_section(msp, s1_x + 5500, s1_y, foot_edge_w, foot_d,
                          50, 12, 150)

    _text(msp, 'TYPICAL SECTION DETAILS — ECP 203 / ECP 202',
          (s1_x + 3500, s1_y + 3200), layer='S-ANNO', h=380)

    # Schedules
    col_rows = []
    for c in columns:
        col_rows.append([
            c['id'],
            f"{int(c['size'])}x{int(c['size'])}",
            f"4Ø{int(main_dia)}" if c['size'] <= 250 else f"4Ø{int(main_dia+2)}",
            f"Ø{int(stir_dia)} @ {int(stir_sp)}",
            "40"
        ])
    used_h = _draw_table(
        msp, sched_ox, sched_oy,
        "COLUMN SCHEDULE (ECP 203)",
        ["ID", "Size (mm)", "Main Bars", "Stirrups (mm)", "Cover"],
        col_rows,
        [1400, 1800, 1700, 2400, 1400],
        row_h=330, hdr_h=380
    )

    beam_rows = []
    for b in beams:
        beam_rows.append([
            b['id'],
            f"{int(b['b'])}x{int(b['d'])}",
            f"2Ø{int(main_dia)}",
            f"3Ø{int(main_dia)}",
            f"Ø{int(stir_dia)} @ {int(stir_sp*2)}"
        ])
    used_h2 = _draw_table(
        msp, sched_ox, sched_oy - used_h - 500,
        "BEAM SCHEDULE (ECP 203)",
        ["ID", "Size (mm)", "Top Bars", "Bottom Bars", "Stirrups"],
        beam_rows,
        [1400, 1800, 1700, 1800, 2400],
        row_h=330, hdr_h=380
    )

    foot_rows = []
    for f in footings:
        foot_rows.append([
            f['id'],
            f"{int(f['size'])}x{int(f['size'])}",
            f"{int(f['d'])}",
            "Ø12 @ 150 both ways",
            "50"
        ])
    used_h3 = _draw_table(
        msp, sched_ox, sched_oy - used_h - used_h2 - 1000,
        "FOOTING SCHEDULE (ECP 202)",
        ["ID", "Plan (mm)", "Depth (mm)", "Bottom R/F", "Cover"],
        foot_rows,
        [1400, 2000, 1600, 3200, 1400],
        row_h=330, hdr_h=380
    )

    # ---------- 11. BOQ ----------
    n_cols = len(columns)
    slab_vol = (bldg_w_m * bldg_l_m * slab_t / 1000.0) * num_floors
    col_vol = sum((c['size'] / 1000.0) ** 2 * floor_h for c in columns) * num_floors
    beam_vol = 0.0
    for b in beams:
        beam_vol += (b['b'] / 1000.0) * (b['d'] / 1000.0) * b['span_m']
    beam_vol *= num_floors
    foot_vol = sum((f['size'] / 1000.0) ** 2 * (f['d'] / 1000.0) for f in footings)
    total_conc = slab_vol + col_vol + beam_vol + foot_vol

    # rebar (rule of thumb 100 kg/m3)
    total_rebar_kg = total_conc * 100.0

    # formwork ≈ perimeter walls + beams sides
    formwork = 2 * (bldg_w_m + bldg_l_m) * floor_h * num_floors

    # bricks: internal + external walls, 55 bricks/m2
    wall_area = (2 * (bldg_w_m + bldg_l_m) * floor_h
                 + sum(2 * ((r.get('area_m2', 10) ** 0.5) * 2) for r in rooms) * floor_h)
    wall_area *= num_floors
    brick_count = wall_area * 55.0

    flooring_area = bldg_w_m * bldg_l_m * num_floors
    paint_area = wall_area * 2

    num_doors = (len(rooms) + 1) * num_units * num_floors
    num_windows = sum(1 for r in rooms if r.get('needs_window')) * num_units * num_floors

    items = [
        ('C1', 'Concrete (structure)',  round(total_conc, 2),   'm3', BOQ_RATES.get('concrete_m3', 2500)),
        ('C2', 'Reinforcement steel',   round(total_rebar_kg, 0), 'kg', BOQ_RATES.get('rebar_ton', 45000) / 1000.0),
        ('C3', 'Formwork',              round(formwork, 2),     'm2', BOQ_RATES.get('formwork_m2', 350)),
        ('C4', 'Brick masonry',         round(brick_count, 0),  'nos', BOQ_RATES.get('bricks_nos', 3)),
        ('C5', 'Flooring (ceramic)',    round(flooring_area, 2), 'm2', BOQ_RATES.get('flooring_m2', 220)),
        ('C6', 'Internal paint',        round(paint_area, 2),   'm2', BOQ_RATES.get('paint_m2', 45)),
        ('C7', 'Doors',                 num_doors,              'nos', 3000.0),
        ('C8', 'Windows',               num_windows,            'nos', 2000.0),
    ]
    total_cost = 0.0
    boq_records = []
    for code, desc, qty, unit, rate in items:
        line = qty * rate
        total_cost += line
        boq_records.append({
            'Code': code,
            'Item': desc,
            'Quantity': qty,
            'Unit': unit,
            'Unit Rate (EGP)': round(rate, 2),
            'Total Cost (EGP)': round(line, 2),
        })

    boq_rows = [[r['Code'], r['Item'], f"{r['Quantity']:,}",
                 r['Unit'], f"{r['Unit Rate (EGP)']:,}",
                 f"{r['Total Cost (EGP)']:,}"] for r in boq_records]

    _draw_table(
        msp, boq_ox, boq_oy,
        "BILL OF QUANTITIES (PRELIMINARY)",
        ["Code", "Description", "Qty", "Unit", "Rate (EGP)", "Total (EGP)"],
        boq_rows,
        [1200, 4200, 1600, 1000, 2200, 2600],
        row_h=330, hdr_h=400
    )
    _text(msp,
          f"ESTIMATED TOTAL COST: {total_cost:,.0f} EGP",
          (boq_ox + 6400, boq_oy + 380 - 330 * (len(boq_rows) + 1) - 600),
          layer='ANNO-TEXT', h=280)

    # Title block
    info_dict = {
        'plot_area': poly.area,
        'street_width': street_w,
        'location': params.get('location', 'Egypt'),
        'max_floors': max_floors,
        'num_floors': num_floors,
        'footprint_area': round(bldg_w_m * bldg_l_m, 2),
        'building_width': round(bldg_w_m, 2),
        'building_length': round(bldg_l_m, 2),
        'front_setback': front_sb,
        'rear_setback': rear_sb,
        'side_setback': side_sb,
        'num_units': num_units,
        'num_rooms': len(rooms),
        'num_columns': n_cols,
        'grid_x_bays': n_bays_x,
        'grid_y_bays': n_bays_y,
    }
    _draw_title_block(msp, tb_ox, tb_oy, params, info_dict)

    # ---------- 12. Write DXF to bytes ----------
    buf = io.StringIO()
    doc.write(buf, fmt='asc')
    dxf_bytes = buf.getvalue().encode('utf-8')

    return {
        'dxf':  dxf_bytes,
        'boq':  boq_records,
        'info': info_dict,
    }
