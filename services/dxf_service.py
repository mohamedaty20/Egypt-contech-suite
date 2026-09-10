# services/dxf_service.py – Complete fixed version
import io
import math
import numpy as np
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, box
from config import BOQ_RATES

# ----------------------------------------------------------------------
# Valid DXF lineweights
# ----------------------------------------------------------------------
_VALID_LINEWEIGHTS = [0, 5, 9, 13, 15, 18, 20, 25, 30, 35, 40, 50, 53, 60,
                      70, 80, 90, 100, 106, 120, 140, 158, 200, 211]

def _snap_lineweight(lw):
    try:
        lw = int(lw)
    except Exception:
        return 25
    if lw in _VALID_LINEWEIGHTS:
        return lw
    return min(_VALID_LINEWEIGHTS, key=lambda v: abs(v - lw))


# ----------------------------------------------------------------------
# Layer detection / area extraction
# ----------------------------------------------------------------------
def detect_dxf_layers(doc):
    layers = {}
    try:
        for entity in doc.modelspace():
            layer = entity.dxf.layer
            if layer not in layers:
                layers[layer] = {'count': 0, 'types': set(), 'keywords': []}
            layers[layer]['count'] += 1
            layers[layer]['types'].add(entity.dxftype())
    except Exception as e:
        print(f"Error detecting layers: {e}")
        return {}
    for layer in layers:
        lc = layer.lower()
        kws = []
        for kw in ['wall', 'column', 'beam', 'slab', 'footing', 'room', 'door', 'window', 'area']:
            if kw in lc:
                kws.append(kw)
        layers[layer]['keywords'] = kws
    return layers


def extract_areas_from_dxf(doc, unit='mm', workflow='architectural'):
    if unit == 'mm':
        area_scale = 1e-6
    elif unit == 'cm':
        area_scale = 1e-4
    else:
        area_scale = 1.0

    results = []
    msp = doc.modelspace()
    for entity in msp:
        if entity.dxftype() in ('LWPOLYLINE', 'POLYLINE'):
            if entity.closed:
                try:
                    if entity.dxftype() == 'LWPOLYLINE':
                        points = [(p.x, p.y) for p in entity.get_points()]
                    else:
                        points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
                    area = 0.0
                    for i in range(len(points)):
                        x1, y1 = points[i]
                        x2, y2 = points[(i + 1) % len(points)]
                        area += x1 * y2 - x2 * y1
                    area = abs(area) / 2.0
                    label = ""
                    cx = sum(p[0] for p in points) / len(points)
                    cy = sum(p[1] for p in points) / len(points)
                    for txt in msp.query('TEXT MTEXT'):
                        pos = txt.dxf.insert
                        if abs(pos.x - cx) < 10 and abs(pos.y - cy) < 10:
                            label = txt.dxf.text
                            break
                    results.append({
                        'layer': entity.dxf.layer,
                        'area_m2': round(area * area_scale, 4),
                        'label': label.strip() if label else '',
                        'vertices': len(points)
                    })
                except Exception as e:
                    print(f"Skipping entity: {e}")
                    continue
    return results


def create_dxf_layer(doc, name, color, lineweight=25, linetype='CONTINUOUS'):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=linetype,
                       lineweight=_snap_lineweight(lineweight))


# ----------------------------------------------------------------------
# Slice-and-dice room placement
# ----------------------------------------------------------------------
def slice_and_dice(rect, rooms):
    x0, y0, x1, y1 = rect
    if len(rooms) == 0:
        return []
    if len(rooms) == 1:
        return [(rooms[0], rect)]

    w = x1 - x0
    h = y1 - y0
    total_target = sum(r.get('area_m2', 10) for r in rooms) or 1
    first = rooms[0]
    fraction = first.get('area_m2', 10) / total_target

    if w >= h:
        cut = x0 + int(w * fraction)
        first_rect = (x0, y0, cut, y1)
        rest_rect = (cut, y0, x1, y1)
    else:
        cut = y0 + int(h * fraction)
        first_rect = (x0, y0, x1, cut)
        rest_rect = (x0, cut, x1, y1)

    return [(first, first_rect)] + slice_and_dice(rest_rect, rooms[1:])


# ----------------------------------------------------------------------
# MAIN GENERATOR
# ----------------------------------------------------------------------
def build_complete_project(params):
    # ----- 1. Plot polygon -----
    if params.get('plot_polygon') is not None:
        coords = params['plot_polygon']
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
            if not poly.is_valid:
                poly = poly.convex_hull
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
        plot_poly = poly
    else:
        area = params['plot_area_m2']
        pw = params.get('plot_width') or math.sqrt(area / 1.5)
        pl = params.get('plot_length') or (area / pw)
        plot_poly = box(0, 0, pl, pw)

    # ----- 2. Setbacks -----
    street_width = params.get('street_width_m', 10)
    if street_width >= 12:   front_setback = 3.0
    elif street_width >= 8:  front_setback = 2.5
    elif street_width >= 6:  front_setback = 2.0
    else:                    front_setback = 1.5
    rear_setback = 2.0
    side_setback = 1.5

    min_x, min_y, max_x, max_y = plot_poly.bounds
    building_length = max_x - min_x - front_setback - rear_setback
    building_width = max_y - min_y - 2 * side_setback
    if building_length <= 0 or building_width <= 0:
        footprint_ratio = 0.6
        max_footprint = plot_poly.area * footprint_ratio
        b_w = 10.0
        b_l = max_footprint / b_w
        building_length, building_width = b_l, b_w

    building_length = max(building_length, 5.0)
    building_width = max(building_width, 5.0)

    # ----- 3. Floors -----
    if street_width >= 12:   max_floors = 4
    elif street_width >= 8:  max_floors = 3
    elif street_width >= 6:  max_floors = 2
    else:                    max_floors = 1
    num_floors = min(params.get('num_floors', max_floors), max_floors)
    floor_height_m = params.get('floor_height_m', 3.0)

    # ----- 4. Architectural params -----
    door_w = params.get('door_width_mm', 900)
    door_h = params.get('door_height_mm', 2100)
    win_w = params.get('window_width_mm', 1200)
    win_h = params.get('window_height_mm', 1200)
    num_units = params.get('num_units_per_floor', 1)

    # ----- 5. Structural params -----
    col_sp_x = params.get('column_spacing_x', 5.0) * 1000
    col_sp_y = params.get('column_spacing_y', 5.0) * 1000
    beam_w = params.get('beam_width_mm', 300)
    beam_d = params.get('beam_depth_mm', 600)
    slab_t = params.get('slab_thickness_mm', 150)
    footing_w = params.get('footing_width_mm', 1000)
    footing_d = params.get('footing_depth_mm', 400)
    footing_l = params.get('footing_length_mm', 1000)
    rebar_main_d = params.get('rebar_main_diam_mm', 16)
    rebar_stirrup_d = params.get('rebar_stirrup_diam_mm', 10)
    rebar_sp = params.get('rebar_spacing_mm', 200)

    L_mm = building_length * 1000
    W_mm = building_width * 1000
    x0, y0 = 0.0, 0.0

    # ----- 6. DXF setup -----
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    doc.header['$INSUNITS'] = 4

    layers_def = {
        'A-WALL':       {'color': 7,  'lineweight': 50},
        'A-WALL-INT':   {'color': 8,  'lineweight': 30},
        'A-DOOR':       {'color': 3,  'lineweight': 20},
        'A-WINDOW':     {'color': 5,  'lineweight': 15},
        'A-ROOM-TEXT':  {'color': 4,  'lineweight': 13},
        'A-CORE':       {'color': 4,  'lineweight': 25},
        'S-COLUMN':     {'color': 6,  'lineweight': 40},
        'S-BEAM':       {'color': 1,  'lineweight': 30},
        'S-SLAB':       {'color': 9,  'lineweight': 20},
        'S-FOOTING':    {'color': 9,  'lineweight': 40},
        'S-REBAR':      {'color': 10, 'lineweight': 13},
        'ANNO-DIMS':    {'color': 2,  'lineweight': 15},
        'ANNO-TEXT':    {'color': 4,  'lineweight': 13},
    }
    for name, props in layers_def.items():
        create_dxf_layer(doc, name, props['color'], lineweight=props['lineweight'])

    # ----- 7. Outer walls -----
    msp.add_lwpolyline(
        [(x0, y0), (x0 + L_mm, y0), (x0 + L_mm, y0 + W_mm), (x0, y0 + W_mm), (x0, y0)],
        dxfattribs={'layer': 'A-WALL', 'color': 7, 'lineweight': 50}
    )

    # ----- 8. Core -----
    core_w, core_d = 2500, 3500
    core_x = (x0 + L_mm / 2) - core_w / 2
    core_y = (y0 + W_mm / 2) - core_d / 2

    msp.add_lwpolyline(
        [(core_x, core_y),
         (core_x + core_w, core_y),
         (core_x + core_w, core_y + core_d),
         (core_x, core_y + core_d),
         (core_x, core_y)],
        dxfattribs={'layer': 'A-CORE', 'color': 4, 'lineweight': 25}
    )
    t = msp.add_text("STAIR / CORE",
                     dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 150, 'color': 4})
    t.set_placement((core_x + core_w / 2, core_y + core_d / 2),
                    align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 9. AI layout plan -----
    layout_plan = params.get('layout_plan') or {}
    ai_rooms = layout_plan.get('rooms', [])
    if not ai_rooms:
        ai_rooms = [
            {"name": "Living Room", "type": "living", "area_m2": 25, "priority": 1, "zone": "public", "needs_window": True},
            {"name": "Kitchen", "type": "kitchen", "area_m2": 9, "priority": 2, "zone": "public", "needs_window": True},
            {"name": "Master Bedroom", "type": "bedroom_master", "area_m2": 16, "priority": 3, "zone": "private", "needs_window": True},
            {"name": "Bedroom 2", "type": "bedroom", "area_m2": 12, "priority": 4, "zone": "private", "needs_window": True},
            {"name": "Bathroom", "type": "bathroom", "area_m2": 4, "priority": 5, "zone": "private", "needs_window": False},
        ]

    public_rooms = sorted([r for r in ai_rooms if r.get('zone') == 'public'], key=lambda r: r.get('priority', 99))
    private_rooms = sorted([r for r in ai_rooms if r.get('zone') == 'private'], key=lambda r: r.get('priority', 99))

    corridor_h = 1200
    mid_y = y0 + W_mm / 2
    lower_rect = (x0 + 100, y0 + 100, x0 + L_mm - 100, mid_y - corridor_h / 2)
    upper_rect = (x0 + 100, mid_y + corridor_h / 2, x0 + L_mm - 100, y0 + W_mm - 100)

    placements = []
    if public_rooms:
        placements += slice_and_dice(lower_rect, public_rooms)
    if private_rooms:
        placements += slice_and_dice(upper_rect, private_rooms)

    for room, (rx0, ry0, rx1, ry1) in placements:
        msp.add_lwpolyline(
            [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1), (rx0, ry0)],
            dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30}
        )
        t = msp.add_text(room['name'],
                         dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 150, 'color': 4})
        t.set_placement(((rx0 + rx1) / 2, (ry0 + ry1) / 2),
                        align=TextEntityAlignment.MIDDLE_CENTER)

        door_cx = (rx0 + rx1) / 2
        door_cy = ry0 if ry0 > mid_y else ry1
        msp.add_lwpolyline(
            [(door_cx - 450, door_cy), (door_cx + 450, door_cy),
             (door_cx + 450, door_cy + 900), (door_cx - 450, door_cy + 900),
             (door_cx - 450, door_cy)],
            dxfattribs={'layer': 'A-DOOR', 'color': 3, 'lineweight': 20}
        )

        if room.get('needs_window'):
            win_cx = (rx0 + rx1) / 2
            if ry1 >= y0 + W_mm - 200:
                win_cy = ry1
                msp.add_lwpolyline(
                    [(win_cx - 600, win_cy), (win_cx + 600, win_cy),
                     (win_cx + 600, win_cy + 150), (win_cx - 600, win_cy + 150),
                     (win_cx - 600, win_cy)],
                    dxfattribs={'layer': 'A-WINDOW', 'color': 5, 'lineweight': 15}
                )
            elif ry0 <= y0 + 200:
                win_cy = ry0
                msp.add_lwpolyline(
                    [(win_cx - 600, win_cy), (win_cx + 600, win_cy),
                     (win_cx + 600, win_cy - 150), (win_cx - 600, win_cy - 150),
                     (win_cx - 600, win_cy)],
                    dxfattribs={'layer': 'A-WINDOW', 'color': 5, 'lineweight': 15}
                )

    # ----- 10. Structural grid -----
    span_x = min(5000, max(3000, L_mm / max(1, math.ceil(L_mm / 5000))))
    span_y = min(5000, max(3000, W_mm / max(1, math.ceil(W_mm / 5000))))

    cols_x = np.arange(x0, x0 + L_mm + 1, span_x)
    cols_y = np.arange(y0, y0 + W_mm + 1, span_y)

    for cx in cols_x:
        for cy in cols_y:
            msp.add_circle((cx, cy), radius=150,
                           dxfattribs={'layer': 'S-COLUMN', 'color': 6, 'lineweight': 40})

    for cx in cols_x:
        for cy in cols_y:
            if cx < x0 + L_mm:
                msp.add_line((cx, cy), (cx + span_x, cy),
                             dxfattribs={'layer': 'S-BEAM', 'color': 1, 'lineweight': 30})
            if cy < y0 + W_mm:
                msp.add_line((cx, cy), (cx, cy + span_y),
                             dxfattribs={'layer': 'S-BEAM', 'color': 1, 'lineweight': 30})

    msp.add_lwpolyline(
        [(x0, y0), (x0 + L_mm, y0), (x0 + L_mm, y0 + W_mm), (x0, y0 + W_mm), (x0, y0)],
        dxfattribs={'layer': 'S-SLAB', 'color': 9, 'lineweight': 20}
    )

    for cx in cols_x:
        for cy in cols_y:
            msp.add_lwpolyline(
                [(cx - footing_l / 2, cy - footing_w / 2),
                 (cx + footing_l / 2, cy - footing_w / 2),
                 (cx + footing_l / 2, cy + footing_w / 2),
                 (cx - footing_l / 2, cy + footing_w / 2),
                 (cx - footing_l / 2, cy - footing_w / 2)],
                dxfattribs={'layer': 'S-FOOTING', 'color': 9, 'lineweight': 40}
            )

    for cx in cols_x:
        for cy in cols_y:
            msp.add_circle((cx, cy), radius=rebar_main_d / 2,
                           dxfattribs={'layer': 'S-REBAR', 'color': 10, 'lineweight': 13})

    # ----- 11. Dimensions -----
    msp.add_line((x0, y0 - 400), (x0 + L_mm, y0 - 400),
                 dxfattribs={'layer': 'ANNO-DIMS', 'color': 2, 'lineweight': 15})
    t = msp.add_text(f"L = {L_mm/1000:.2f} m",
                     dxfattribs={'layer': 'ANNO-TEXT', 'height': 200, 'color': 2})
    t.set_placement((x0 + L_mm / 2, y0 - 600), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_line((x0 - 400, y0), (x0 - 400, y0 + W_mm),
                 dxfattribs={'layer': 'ANNO-DIMS', 'color': 2, 'lineweight': 15})
    t = msp.add_text(f"W = {W_mm/1000:.2f} m",
                     dxfattribs={'layer': 'ANNO-TEXT', 'height': 200, 'color': 2})
    t.set_placement((x0 - 600, y0 + W_mm / 2), align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 12. BOQ -----
    num_columns = len(cols_x) * len(cols_y)

    slab_vol = (L_mm / 1000) * (W_mm / 1000) * (slab_t / 1000) * num_floors
    col_vol = num_columns * (0.3 * 0.3) * floor_height_m * num_floors
    beam_vol = ((len(cols_x) - 1) * len(cols_y) * (beam_w / 1000) * (beam_d / 1000) * (span_x / 1000)
                + (len(cols_y) - 1) * len(cols_x) * (beam_w / 1000) * (beam_d / 1000) * (span_y / 1000)) * num_floors
    footing_vol = num_columns * (footing_l / 1000) * (footing_w / 1000) * (footing_d / 1000)
    total_concrete = slab_vol + col_vol + beam_vol + footing_vol

    main_len = floor_height_m * 4 * num_columns * num_floors
    main_weight = main_len * (math.pi * (rebar_main_d / 1000) ** 2 / 4 * 7850) / 1000
    stirrup_perim = 2 * (0.3 + 0.3) * 1000
    num_stirrups = (floor_height_m / (rebar_sp / 1000) + 1) * num_columns * num_floors
    stirrup_len = stirrup_perim * num_stirrups / 1000
    stirrup_weight = stirrup_len * (math.pi * (rebar_stirrup_d / 1000) ** 2 / 4 * 7850) / 1000
    total_rebar = main_weight + stirrup_weight

    formwork = 2 * (L_mm / 1000 + W_mm / 1000) * floor_height_m * num_floors
    total_floor_area = (L_mm / 1000) * (W_mm / 1000) * num_floors

    wall_area = 0
    for _r, (rx0, ry0, rx1, ry1) in placements:
        wall_area += 2 * ((rx1 - rx0) + (ry1 - ry0)) / 1000 * floor_height_m
    wall_area *= num_floors
    brick_count = wall_area * 50
    flooring_area = total_floor_area
    paint_area = wall_area * 2

    num_windows = sum(1 for r in ai_rooms if r.get('needs_window')) * num_units * num_floors
    num_doors = len(ai_rooms) * num_units * num_floors

    boq_items = [
        {'Item': 'Concrete (m³)', 'Quantity': round(total_concrete, 2), 'Unit': 'm³'},
        {'Item': 'Rebar (ton)',    'Quantity': round(total_rebar / 1000, 2), 'Unit': 'ton'},
        {'Item': 'Formwork (m²)', 'Quantity': round(formwork, 2), 'Unit': 'm²'},
        {'Item': 'Bricks (nos)',   'Quantity': round(brick_count), 'Unit': 'nos'},
        {'Item': 'Flooring (m²)', 'Quantity': round(flooring_area, 2), 'Unit': 'm²'},
        {'Item': 'Paint (m²)',    'Quantity': round(paint_area, 2), 'Unit': 'm²'},
        {'Item': 'Windows (nos)', 'Quantity': num_windows, 'Unit': 'nos'},
        {'Item': 'Doors (nos)',    'Quantity': num_doors, 'Unit': 'nos'},
    ]
    rate_map = {
        'concrete': BOQ_RATES['concrete_m3'],
        'rebar':    BOQ_RATES['rebar_ton'],
        'formwork': BOQ_RATES['formwork_m2'],
        'bricks':   BOQ_RATES['bricks_nos'],
        'flooring': BOQ_RATES['flooring_m2'],
        'paint':    BOQ_RATES['paint_m2'],
        'windows':  2000.0,
        'doors':    3000.0,
    }
    for item in boq_items:
        key = item['Item'].split('(')[0].strip().lower()
        rate = rate_map.get(key, 0)
        item['Unit Rate (EGP)'] = rate
        item['Total Cost (EGP)'] = round(item['Quantity'] * rate, 2)

    boq_df = pd.DataFrame(boq_items)

    layout_info = {
        'plot_area': plot_poly.area,
        'street_width': street_width,
        'location': params.get('location', ''),
        'max_floors': max_floors,
        'num_floors': num_floors,
        'footprint_area': round((L_mm / 1000) * (W_mm / 1000), 2),
        'building_width': round(W_mm / 1000, 2),
        'building_length': round(L_mm / 1000, 2),
        'front_setback': front_setback,
        'rear_setback': rear_setback,
        'side_setback': side_setback,
        'num_units': num_units,
        'num_rooms': len(placements),
        'num_columns': num_columns,
    }

    dxf_buffer = io.BytesIO()
    doc.write(dxf_buffer)
    return {
        'dxf': dxf_buffer.getvalue(),
        'boq': boq_df.to_dict('records'),
        'info': layout_info
    }
