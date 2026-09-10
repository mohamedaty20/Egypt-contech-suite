# services/dxf_service.py – Professional architectural + structural generator
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


# ======================================================================
# DXF reader helpers (unchanged)
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
        kws = [k for k in ['wall','column','beam','slab','footing','room','door','window','area'] if k in lc]
        layers[layer]['keywords'] = kws
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
                results.append({'layer': entity.dxf.layer, 'area_m2': round(a*scale,4),
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
    """Return the 4 corners of a wall rectangle centered on segment p1→p2."""
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2-x1, y2-y1
    L = math.hypot(dx, dy)
    if L < 1: return None
    nx, ny = -dy/L * thickness/2, dx/L * thickness/2
    return [(x1+nx, y1+ny), (x2+nx, y2+ny),
            (x2-nx, y2-ny), (x1-nx, y1-ny), (x1+nx, y1+ny)]


def draw_wall(msp, p1, p2, thickness, layer='A-WALL-EXT', color=7):
    r = _wall_rect(p1, p2, thickness)
    if r:
        msp.add_lwpolyline(r, dxfattribs={'layer': layer, 'color': color})


def draw_door(msp, cx, cy, width, wall_thickness, is_horizontal=True, flip=False):
    """Draw a door symbol: gap + leaf + swing arc."""
    # Door frame (thin lines across the wall)
    if is_horizontal:
        # wall runs horizontally — door opens upward/downward
        msp.add_line((cx - width/2, cy - wall_thickness/2),
                     (cx - width/2, cy + wall_thickness/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        msp.add_line((cx + width/2, cy - wall_thickness/2),
                     (cx + width/2, cy + wall_thickness/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        # leaf (open 90°)
        if flip:
            msp.add_line((cx - width/2, cy), (cx - width/2, cy - width),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx - width/2, cy), radius=width,
                        start_angle=270, end_angle=360,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})
        else:
            msp.add_line((cx - width/2, cy), (cx - width/2, cy + width),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx - width/2, cy), radius=width,
                        start_angle=0, end_angle=90,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})
    else:
        msp.add_line((cx - wall_thickness/2, cy - width/2),
                     (cx + wall_thickness/2, cy - width/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        msp.add_line((cx - wall_thickness/2, cy + width/2),
                     (cx + wall_thickness/2, cy + width/2),
                     dxfattribs={'layer': 'A-DOOR', 'color': 3})
        if flip:
            msp.add_line((cx, cy - width/2), (cx - width, cy - width/2),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx, cy - width/2), radius=width,
                        start_angle=180, end_angle=270,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})
        else:
            msp.add_line((cx, cy - width/2), (cx + width, cy - width/2),
                         dxfattribs={'layer': 'A-DOOR', 'color': 3})
            msp.add_arc((cx, cy - width/2), radius=width,
                        start_angle=0, end_angle=90,
                        dxfattribs={'layer': 'A-DOOR', 'color': 3})


def draw_window(msp, cx, cy, width, wall_thickness, is_horizontal=True):
    """Draw a window symbol: 3 parallel lines representing frame + glass."""
    if is_horizontal:
        for off in (-wall_thickness/2, 0, wall_thickness/2):
            msp.add_line((cx - width/2, cy + off), (cx + width/2, cy + off),
                         dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        # end caps
        msp.add_line((cx - width/2, cy - wall_thickness/2),
                     (cx - width/2, cy + wall_thickness/2),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        msp.add_line((cx + width/2, cy - wall_thickness/2),
                     (cx + width/2, cy + wall_thickness/2),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        # center mullion
        msp.add_line((cx, cy - wall_thickness/2), (cx, cy + wall_thickness/2),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})
    else:
        for off in (-wall_thickness/2, 0, wall_thickness/2):
            msp.add_line((cx + off, cy - width/2), (cx + off, cy + width/2),
                         dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        msp.add_line((cx - wall_thickness/2, cy - width/2),
                     (cx + wall_thickness/2, cy - width/2),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        msp.add_line((cx - wall_thickness/2, cy + width/2),
                     (cx + wall_thickness/2, cy + width/2),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})
        msp.add_line((cx - wall_thickness/2, cy), (cx + wall_thickness/2, cy),
                     dxfattribs={'layer': 'A-WINDOW', 'color': 5})


def place_furniture(msp, x, y, w, h, room_type):
    """Simple furniture symbol based on room type."""
    c = 'A-FURN'
    col = 6
    if room_type in ('bedroom_master', 'bedroom'):
        # bed
        bw, bh = w*0.55, h*0.65
        bx, by = x + (w-bw)/2, y + (h-bh)/2
        msp.add_lwpolyline([(bx, by), (bx+bw, by), (bx+bw, by+bh), (bx, by+bh)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
        # pillow
        msp.add_lwpolyline([(bx+50, by+bh-250), (bx+bw-50, by+bh-250),
                            (bx+bw-50, by+bh-50), (bx+50, by+bh-50)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
    elif room_type == 'living':
        # sofa + coffee table
        msp.add_lwpolyline([(x+200, y+200), (x+200+w*0.35, y+200),
                            (x+200+w*0.35, y+200+h*0.3), (x+200, y+200+h*0.3)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
        msp.add_lwpolyline([(x+w*0.55, y+h*0.35), (x+w*0.85, y+h*0.35),
                            (x+w*0.85, y+h*0.65), (x+w*0.55, y+h*0.65)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
    elif room_type == 'kitchen':
        # counter along wall
        msp.add_lwpolyline([(x+100, y+100), (x+w-100, y+100),
                            (x+w-100, y+700), (x+100, y+700)],
                           dxfattribs={'layer': c, 'color': col}, close=True)
    elif room_type == 'bathroom':
        # WC + basin
        msp.add_circle((x+400, y+400), radius=200, dxfattribs={'layer': c, 'color': col})
        msp.add_lwpolyline([(x+w-600, y+100), (x+w-100, y+100),
                            (x+w-100, y+500), (x+w-600, y+500)],
                           dxfattribs={'layer': c, 'color': col}, close=True)


# ======================================================================
# Layout engine – rectangle partition with proper walls and openings
# ======================================================================
def _partition_rect(rect, rooms):
    """Partition a rectangle into rooms. Returns list of (room, sub_rect)."""
    if not rooms:
        return []
    if len(rooms) == 1:
        return [(rooms[0], rect)]

    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    total = sum(r.get('area_m2', 10) for r in rooms) or 1
    frac = rooms[0].get('area_m2', 10) / total
    frac = max(0.20, min(0.80, frac))

    if w >= h:
        cut = x0 + int(w * frac)
        return [(rooms[0], (x0, y0, cut, y1))] + _partition_rect((cut, y0, x1, y1), rooms[1:])
    else:
        cut = y0 + int(h * frac)
        return [(rooms[0], (x0, y0, x1, cut))] + _partition_rect((x0, cut, x1, y1), rooms[1:])


# ======================================================================
# MAIN GENERATOR
# ======================================================================
def build_complete_project(params):
    # Seed for reproducible-but-varied output
    seed = hash((params.get('plot_area_m2', 200),
                 params.get('street_width_m', 10),
                 params.get('num_floors', 2),
                 params.get('num_bedrooms', 3)))
    random.seed(seed)

    # ----- 1. Plot polygon -----
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

    # ----- 2. Egyptian Law 119/2008 — Setbacks & coverage -----
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
        scale = math.sqrt(max_footprint / (avail_w * avail_l))
        avail_w *= scale
        avail_l *= scale

    building_width = max(6.0, avail_w)
    building_length = max(6.0, avail_l)

    num_floors = min(params.get('num_floors', max_floors), max_floors)
    floor_h_m = params.get('floor_height_m', 3.0)

    # ----- 3. Get room program from AI plan (or fallback) -----
    layout_plan = params.get('layout_plan') or {}
    ai_rooms = layout_plan.get('rooms', [])
    num_bedrooms = params.get('num_bedrooms', 3)
    num_bathrooms = params.get('num_bathrooms', 2)

    if not ai_rooms:
        # Fallback program – varied by inputs
        ai_rooms = [
            {"name": "Living Room", "type": "living", "area_m2": 26, "priority": 1, "zone": "public", "needs_window": True},
            {"name": "Kitchen", "type": "kitchen", "area_m2": 10, "priority": 2, "zone": "public", "needs_window": True},
        ]
        for i in range(num_bedrooms):
            name = "Master Bedroom" if i == 0 else f"Bedroom {i+1}"
            rtype = "bedroom_master" if i == 0 else "bedroom"
            area = 16 if i == 0 else 12
            ai_rooms.append({"name": name, "type": rtype, "area_m2": area,
                             "priority": 3+i, "zone": "private", "needs_window": True})
        for i in range(num_bathrooms):
            ai_rooms.append({"name": "Bathroom" if i == 0 else f"Bathroom {i+1}",
                             "type": "bathroom", "area_m2": 4.5,
                             "priority": 10+i, "zone": "private", "needs_window": False})

    # ----- 4. Convert to mm -----
    L = building_length * 1000
    W = building_width * 1000
    x0, y0 = 0.0, 0.0
    x1, y1 = L, W

    # ----- 5. DXF setup -----
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    doc.header['$INSUNITS'] = 4

    layers_def = {
        'A-WALL-EXT':   {'color': 7,  'lineweight': 50},
        'A-WALL-INT':   {'color': 8,  'lineweight': 25},
        'A-DOOR':       {'color': 3,  'lineweight': 18},
        'A-WINDOW':     {'color': 5,  'lineweight': 13},
        'A-FURN':       {'color': 6,  'lineweight': 9},
        'A-ROOM-TEXT':  {'color': 4,  'lineweight': 13},
        'A-CORE':       {'color': 4,  'lineweight': 30},
        'S-COLUMN':     {'color': 1,  'lineweight': 50},
        'S-BEAM':       {'color': 5,  'lineweight': 25},
        'S-FOOTING':    {'color': 9,  'lineweight': 40},
        'S-SLAB':       {'color': 9,  'lineweight': 18},
        'ANNO-DIM':     {'color': 2,  'lineweight': 13},
        'ANNO-TEXT':    {'color': 7,  'lineweight': 13},
        'ANNO-TITLE':   {'color': 7,  'lineweight': 25},
    }
    for n, p in layers_def.items():
        create_dxf_layer(doc, n, p['color'], lineweight=p['lineweight'])

    # ----- 6. Partition building into zones -----
    # Corridor across the middle
    corridor_h = 1300
    mid_y = (y0 + y1) / 2
    corridor_top = mid_y + corridor_h / 2
    corridor_bot = mid_y - corridor_h / 2

    # Public zone: front (lower half in DXF coordinates) — near street
    public_zone = (x0, y0, x1, corridor_bot)
    private_zone = (x0, corridor_top, x1, y1)

    public_rooms = sorted([r for r in ai_rooms if r.get('zone') == 'public'],
                          key=lambda r: r.get('priority', 99))
    private_rooms = sorted([r for r in ai_rooms if r.get('zone') == 'private'],
                           key=lambda r: r.get('priority', 99))

    # Randomize partition direction slightly to vary layouts
    if random.random() > 0.5:
        private_rooms = list(reversed(private_rooms))

    public_placements = _partition_rect(public_zone, public_rooms) if public_rooms else []
    private_placements = _partition_rect(private_zone, private_rooms) if private_rooms else []
    placements = public_placements + private_placements

    # ----- 7. Draw OUTER WALLS with thickness -----
    wall_ext_t = 250
    msp.add_lwpolyline(
        _wall_rect((x0, y0), (x0 + L, y0), wall_ext_t),   # bottom
        dxfattribs={'layer': 'A-WALL-EXT', 'color': 7})
    msp.add_lwpolyline(
        _wall_rect((x0 + L, y0), (x0 + L, y0 + W), wall_ext_t),
        dxfattribs={'layer': 'A-WALL-EXT', 'color': 7})
    msp.add_lwpolyline(
        _wall_rect((x0 + L, y0 + W), (x0, y0 + W), wall_ext_t),
        dxfattribs={'layer': 'A-WALL-EXT', 'color': 7})
    msp.add_lwpolyline(
        _wall_rect((x0, y0 + W), (x0, y0), wall_ext_t),
        dxfattribs={'layer': 'A-WALL-EXT', 'color': 7})

    # ----- 8. Draw INTERNAL PARTITION WALLS -----
    int_t = 150
    wall_segments = []   # store for door/window placement

    for room, (rx0, ry0, rx1, ry1) in placements:
        # Only draw the walls that border another room or corridor — not the outer walls
        # Left wall
        if rx0 > x0 + wall_ext_t:
            wall_segments.append(((rx0, ry0), (rx0, ry1), int_t, 'V'))
        # Right wall
        if rx1 < x1 - wall_ext_t:
            wall_segments.append(((rx1, ry0), (rx1, ry1), int_t, 'V'))
        # Bottom wall
        if ry0 > y0 + wall_ext_t:
            wall_segments.append(((rx0, ry0), (rx1, ry0), int_t, 'H'))
        # Top wall
        if ry1 < y1 - wall_ext_t:
            wall_segments.append(((rx0, ry1), (rx1, ry1), int_t, 'H'))

    for p1, p2, t, orient in wall_segments:
        r = _wall_rect(p1, p2, t)
        if r:
            msp.add_lwpolyline(r, dxfattribs={'layer': 'A-WALL-INT', 'color': 8})

    # ----- 9. Doors on every room -----
    door_w = 900
    for room, (rx0, ry0, rx1, ry1) in placements:
        # Put door on the side that faces the corridor
        room_cy = (ry0 + ry1) / 2
        if room_cy < mid_y:
            # public room → door on top wall (facing corridor)
            door_cx = (rx0 + rx1) / 2
            door_cy = ry1
            draw_door(msp, door_cx, door_cy, door_w, int_t, is_horizontal=True, flip=False)
        else:
            # private room → door on bottom wall (facing corridor)
            door_cx = (rx0 + rx1) / 2
            door_cy = ry0
            draw_door(msp, door_cx, door_cy, door_w, int_t, is_horizontal=True, flip=True)

    # ----- 10. Windows on exterior walls of public rooms -----
    win_w = 1400
    for room, (rx0, ry0, rx1, ry1) in placements:
        if not room.get('needs_window'):
            continue
        # Bottom wall of building
        if abs(ry0 - y0) < wall_ext_t:
            draw_window(msp, (rx0+rx1)/2, y0, win_w, wall_ext_t, is_horizontal=True)
        # Top wall
        if abs(ry1 - y1) < wall_ext_t:
            draw_window(msp, (rx0+rx1)/2, y1, win_w, wall_ext_t, is_horizontal=True)

    # Windows on side walls for rooms touching side
    for room, (rx0, ry0, rx1, ry1) in placements:
        if not room.get('needs_window'):
            continue
        if abs(rx0 - x0) < wall_ext_t:
            draw_window(msp, x0, (ry0+ry1)/2, win_w, wall_ext_t, is_horizontal=False)
        if abs(rx1 - x1) < wall_ext_t:
            draw_window(msp, x1, (ry0+ry1)/2, win_w, wall_ext_t, is_horizontal=False)

    # ----- 11. Furniture + labels -----
    for room, (rx0, ry0, rx1, ry1) in placements:
        w = rx1 - rx0
        h = ry1 - ry0
        area_m2 = (w * h) / 1_000_000
        place_furniture(msp, rx0, ry0, w, h, room.get('type', 'bedroom'))
        # Label
        cx_lbl = (rx0 + rx1) / 2
        cy_lbl = (ry0 + ry1) / 2
        t = msp.add_text(
            f"{room['name']}\n{area_m2:.1f} m²",
            dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 180, 'color': 4}
        )
        t.set_placement((cx_lbl, cy_lbl), align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 12. Core (staircase) -----
    core_w, core_d = 2400, 3600
    core_x = x1 - wall_ext_t - core_w - 400
    core_y = mid_y - core_d / 2
    msp.add_lwpolyline(
        [(core_x, core_y), (core_x+core_w, core_y),
         (core_x+core_w, core_y+core_d), (core_x, core_y+core_d)],
        dxfattribs={'layer': 'A-CORE', 'color': 4}, close=True
    )
    # Stair treads
    for i in range(1, 12):
        ty = core_y + (core_d / 12) * i
        msp.add_line((core_x, ty), (core_x + core_w, ty),
                     dxfattribs={'layer': 'A-CORE', 'color': 4})
    t = msp.add_text("STAIR", dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 180, 'color': 4})
    t.set_placement((core_x + core_w/2, core_y - 300),
                    align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 13. Structural grid (columns at max 5 m, ECP 203) -----
    span_max = 5000
    nx = max(2, math.ceil(L / span_max) + 1)
    ny = max(2, math.ceil(W / span_max) + 1)
    xs = np.linspace(x0, x0 + L, nx)
    ys = np.linspace(y0, y0 + W, ny)

    col_size = 300 if num_floors <= 2 else 350
    for cx in xs:
        for cy in ys:
            msp.add_lwpolyline(
                [(cx-col_size/2, cy-col_size/2), (cx+col_size/2, cy-col_size/2),
                 (cx+col_size/2, cy+col_size/2), (cx-col_size/2, cy+col_size/2)],
                dxfattribs={'layer': 'S-COLUMN', 'color': 1}, close=True
            )

    # Beams (dashed lines between columns)
    for i, cx in enumerate(xs):
        for cy in ys:
            if i < len(xs)-1:
                msp.add_line((cx, cy), (xs[i+1], cy),
                             dxfattribs={'layer': 'S-BEAM', 'color': 5, 'linetype': 'DASHED'})
    for i, cy in enumerate(ys):
        for cx in xs:
            if i < len(ys)-1:
                msp.add_line((cx, cy), (cx, ys[i+1]),
                             dxfattribs={'layer': 'S-BEAM', 'color': 5, 'linetype': 'DASHED'})

    # Footings (larger squares below columns)
    foot = 1200 if num_floors <= 2 else 1500
    for cx in xs:
        for cy in ys:
            msp.add_lwpolyline(
                [(cx-foot/2, cy-foot/2), (cx+foot/2, cy-foot/2),
                 (cx+foot/2, cy+foot/2), (cx-foot/2, cy+foot/2)],
                dxfattribs={'layer': 'S-FOOTING', 'color': 9}, close=True
            )

    # ----- 14. Dimension chains -----
    off = 700
    # Bottom chain — overall length
    msp.add_line((x0, y0-off), (x1, y0-off),
                 dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    # Tick marks
    for px in [x0, x1]:
        msp.add_line((px, y0-off-100), (px, y0-off+100),
                     dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    t = msp.add_text(f"{L/1000:.2f} m", dxfattribs={'layer': 'ANNO-TEXT', 'height': 200, 'color': 2})
    t.set_placement(((x0+x1)/2, y0-off-350), align=TextEntityAlignment.MIDDLE_CENTER)

    # Left chain — overall width
    msp.add_line((x0-off, y0), (x0-off, y1),
                 dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    for py in [y0, y1]:
        msp.add_line((x0-off-100, py), (x0-off+100, py),
                     dxfattribs={'layer': 'ANNO-DIM', 'color': 2})
    t = msp.add_text(f"{W/1000:.2f} m", dxfattribs={'layer': 'ANNO-TEXT', 'height': 200, 'color': 2})
    t.set_placement((x0-off-700, (y0+y1)/2), align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 15. Title block -----
    tb_w, tb_h = 7000, 4000
    tb_x = x1 - tb_w - 200
    tb_y = y0 - tb_h - 3000
    msp.add_lwpolyline(
        [(tb_x, tb_y), (tb_x+tb_w, tb_y), (tb_x+tb_w, tb_y+tb_h), (tb_x, tb_y+tb_h)],
        dxfattribs={'layer': 'ANNO-TITLE', 'color': 7}, close=True
    )
    # Title text
    title_lines = [
        (f"PROJECT: {params.get('project_name', 'Residential Building')[:40]}", tb_y+tb_h-350),
        (f"ENGINEER: {params.get('engineer', 'Eng. Mohamed')[:40]}", tb_y+tb_h-700),
        (f"DATE: {params.get('date', '')}", tb_y+tb_h-1050),
        (f"PLOT: {plot_area:.1f} m²   FLOORS: {num_floors}   COVERAGE: {coverage*100:.0f}%", tb_y+tb_h-1400),
        ("DRAWING: GROUND FLOOR PLAN + STRUCTURE", tb_y+tb_h-1750),
        (f"SCALE: 1:100 (mm)   UNITS: mm", tb_y+tb_h-2100),
        ("EGYPTIAN CODE: LAW 119/2008 · ECP 203", tb_y+tb_h-2450),
    ]
    for txt, yy in title_lines:
        t = msp.add_text(txt, dxfattribs={'layer': 'ANNO-TEXT', 'height': 180, 'color': 7})
        t.set_placement((tb_x + 300, yy), align=TextEntityAlignment.LEFT)

    # North arrow (top-right corner of plot)
    na_x = x1 + 500
    na_y = y1 + 500
    msp.add_line((na_x, na_y), (na_x, na_y+1200),
                 dxfattribs={'layer': 'ANNO-TITLE', 'color': 7})
    msp.add_lwpolyline([(na_x, na_y+1200), (na_x-200, na_y+800), (na_x+200, na_y+800)],
                       dxfattribs={'layer': 'ANNO-TITLE', 'color': 7}, close=True)
    t = msp.add_text("N", dxfattribs={'layer': 'ANNO-TEXT', 'height': 250, 'color': 7})
    t.set_placement((na_x, na_y+1600), align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 16. BOQ (Egyptian code based) -----
    num_cols = nx * ny
    slab_t = 140
    beam_b, beam_d = 250, 600

    slab_vol = (L/1000) * (W/1000) * (slab_t/1000) * num_floors
    col_vol = num_cols * (col_size/1000)**2 * floor_h_m * num_floors
    total_beam_length = (nx-1)*ny*L/(nx-1) + (ny-1)*nx*W/(ny-1)
    beam_vol = total_beam_length/1000 * (beam_b/1000) * (beam_d/1000) * num_floors
    foot_vol = num_cols * (foot/1000)**2 * 0.5
    total_concrete = slab_vol + col_vol + beam_vol + foot_vol

    # Rebar per ECP 203 (approx 110 kg/m³ for skeleton)
    rebar_ton = total_concrete * 0.110

    formwork = (total_beam_length/1000) * (beam_d/1000 + 2*beam_b/1000) * num_floors + col_vol * 8

    # Walls
    total_wall_length = 0
    for room, (rx0, ry0, rx1, ry1) in placements:
        total_wall_length += 2*((rx1-rx0) + (ry1-ry0)) / 1000
    wall_area = total_wall_length * floor_h_m * num_floors
    brick_count = wall_area * 50

    flooring_area = (L/1000)*(W/1000)*num_floors
    paint_area = wall_area * 2

    boq_items = [
        {'Item': 'Concrete (m³)', 'Quantity': round(total_concrete, 2), 'Unit': 'm³'},
        {'Item': 'Rebar (ton)',    'Quantity': round(rebar_ton, 2), 'Unit': 'ton'},
        {'Item': 'Formwork (m²)', 'Quantity': round(formwork, 2), 'Unit': 'm²'},
        {'Item': 'Bricks (nos)',   'Quantity': round(brick_count), 'Unit': 'nos'},
        {'Item': 'Flooring (m²)', 'Quantity': round(flooring_area, 2), 'Unit': 'm²'},
        {'Item': 'Paint (m²)',    'Quantity': round(paint_area, 2), 'Unit': 'm²'},
        {'Item': 'Windows (nos)', 'Quantity': sum(1 for r in ai_rooms if r.get('needs_window')) * num_floors, 'Unit': 'nos'},
        {'Item': 'Doors (nos)',    'Quantity': len(ai_rooms) * num_floors, 'Unit': 'nos'},
    ]
    rate_map = {'concrete': 2500.0, 'rebar': 15000.0, 'formwork': 300.0,
                'bricks': 2.5, 'flooring': 150.0, 'paint': 30.0,
                'windows': 2000.0, 'doors': 3000.0}
    for item in boq_items:
        key = item['Item'].split('(')[0].strip().lower()
        rate = rate_map.get(key, 0)
        item['Unit Rate (EGP)'] = rate
        item['Total Cost (EGP)'] = round(item['Quantity'] * rate, 2)

    boq_df = pd.DataFrame(boq_items)

    # ----- 17. Info -----
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
        'num_columns': num_cols,
        'coverage_ratio': f"{coverage*100:.0f}%",
    }

    # ----- 18. Write DXF -----
    dxf_buf = io.StringIO()
    doc.write(dxf_buf, fmt='asc')
    dxf_bytes = dxf_buf.getvalue().encode('utf-8')

    return {'dxf': dxf_bytes, 'boq': boq_df.to_dict('records'), 'info': layout_info}
