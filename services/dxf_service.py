import io
import numpy as np
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon
from config import BOQ_RATES

# ----- DXF layer detection -----
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
        keywords = []
        if 'wall' in lc: keywords.append('wall')
        if 'column' in lc: keywords.append('column')
        if 'beam' in lc: keywords.append('beam')
        if 'slab' in lc: keywords.append('slab')
        if 'footing' in lc or 'foundation' in lc: keywords.append('foundation')
        if 'room' in lc: keywords.append('room')
        if 'door' in lc: keywords.append('door')
        if 'window' in lc: keywords.append('window')
        if 'area' in lc: keywords.append('area')
        layers[layer]['keywords'] = keywords
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
                        x2, y2 = points[(i+1) % len(points)]
                        area += x1*y2 - x2*y1
                    area = abs(area) / 2.0
                    label = ""
                    centroid_x = sum(p[0] for p in points) / len(points)
                    centroid_y = sum(p[1] for p in points) / len(points)
                    for txt in msp.query('TEXT MTEXT'):
                        if txt.dxftype() == 'TEXT':
                            pos = txt.dxf.insert
                        else:
                            pos = txt.dxf.insert
                        if abs(pos.x - centroid_x) < 10 and abs(pos.y - centroid_y) < 10:
                            label = txt.dxf.text
                            break
                    raw_area = area * area_scale
                    results.append({
                        'layer': entity.dxf.layer,
                        'area_m2': round(raw_area, 4),
                        'label': label.strip() if label else '',
                        'vertices': len(points)
                    })
                except Exception as e:
                    print(f"Skipping entity: {e}")
                    continue
    return results

# ----- Enhanced AutoCAD layout generator -----
def create_dxf_layer(doc, name, color, lineweight=25, linetype='CONTINUOUS'):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=linetype, lineweight=lineweight)

def build_complete_project(params):
    """
    Enhanced layout generator using shapely, Egyptian setbacks, and production DXF.
    params: dict with keys:
        - plot_polygon: list of (x,y) coordinates or None (if None, rectangle derived from plot_area_m2)
        - plot_area_m2: float
        - street_width_m: float
        - location: str
        - num_floors: int
        - floor_height_m: float
        - wall_thickness_mm: int
        - door_width_mm, door_height_mm, window_width_mm, window_height_mm
        - column_spacing_x, column_spacing_y (m)
        - beam_width_mm, beam_depth_mm, slab_thickness_mm
        - footing_width_mm, footing_depth_mm, footing_length_mm
        - rebar_main_diam_mm, rebar_stirrup_diam_mm, rebar_spacing_mm
        - project_name, engineer, date
    Returns: dict with keys 'dxf' (bytes), 'boq' (list of dict), 'info' (dict)
    """
    # 1. Determine polygon
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
        plot_area = poly.area
    else:
        area = params['plot_area_m2']
        aspect = 1.5
        width = np.sqrt(area / aspect)
        length = area / width
        poly = Polygon([(0,0), (length,0), (length,width), (0,width)])
        plot_area = area

    # 2. Setbacks based on Egyptian practice (simplified)
    street_width = params['street_width_m']
    if street_width >= 12:
        front_setback = 3.0
    elif street_width >= 8:
        front_setback = 2.5
    elif street_width >= 6:
        front_setback = 2.0
    else:
        front_setback = 1.5
    rear_setback = 2.0
    side_setback = 1.5

    min_x, min_y, max_x, max_y = poly.bounds
    building_length = max_x - min_x - front_setback - rear_setback
    building_width = max_y - min_y - 2 * side_setback
    if building_length <= 0 or building_width <= 0:
        footprint_ratio = 0.6
        max_footprint = plot_area * footprint_ratio
        b_w = 10.0
        b_l = max_footprint / b_w
        building_length = b_l
        building_width = b_w

    if building_width < 5: building_width = 5
    if building_length < 5: building_length = 5

    # 3. Max floors based on street width (Egyptian rule)
    if street_width >= 12:
        max_floors = 4
    elif street_width >= 8:
        max_floors = 3
    elif street_width >= 6:
        max_floors = 2
    else:
        max_floors = 1
    num_floors = min(params.get('num_floors', max_floors), max_floors)

    # 4. Extract other params
    wall_thick = params['wall_thickness_mm']
    door_w = params['door_width_mm']
    door_h = params['door_height_mm']
    win_w = params['window_width_mm']
    win_h = params['window_height_mm']
    col_sp_x = params['column_spacing_x'] * 1000
    col_sp_y = params['column_spacing_y'] * 1000
    beam_w = params['beam_width_mm']
    beam_d = params['beam_depth_mm']
    slab_t = params['slab_thickness_mm']
    footing_w = params['footing_width_mm']
    footing_d = params['footing_depth_mm']
    footing_l = params['footing_length_mm']
    rebar_main_d = params['rebar_main_diam_mm']
    rebar_stirrup_d = params['rebar_stirrup_diam_mm']
    rebar_sp = params['rebar_spacing_mm']
    floor_h = params['floor_height_m'] * 1000

    # 5. Create DXF document with professional layers
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    doc.header['$INSUNITS'] = 4

    layers_def = {
        'A-WALL': {'color': 7, 'lineweight': 40},
        'A-WALL-INT': {'color': 8, 'lineweight': 30},
        'A-DOOR': {'color': 3, 'lineweight': 20},
        'A-WINDOW': {'color': 5, 'lineweight': 15},
        'A-ROOM': {'color': 4, 'lineweight': 10},
        'S-COLUMN': {'color': 6, 'lineweight': 30},
        'S-BEAM': {'color': 7, 'lineweight': 50},
        'S-SLAB': {'color': 8, 'lineweight': 20},
        'S-FOOTING': {'color': 9, 'lineweight': 40},
        'S-REBAR': {'color': 10, 'lineweight': 15},   # fixed: was 10, now 15
        'ANNO-DIMS': {'color': 2, 'lineweight': 15},
        'ANNO-TEXT': {'color': 4, 'lineweight': 10},
        'ANNO-HATCH': {'color': 1, 'lineweight': 5},
    }
    for name, props in layers_def.items():
        create_dxf_layer(doc, name, props['color'], lineweight=props['lineweight'])

    L = building_length * 1000
    W = building_width * 1000
    x0 = 0.0
    y0 = 0.0

    # ========== ARCHITECTURAL LAYOUT ==========
    outer_pts = [(x0, y0), (x0+L, y0), (x0+L, y0+W), (x0, y0+W), (x0, y0)]
    msp.add_lwpolyline(outer_pts, dxfattribs={'layer': 'A-WALL', 'color': 7, 'lineweight': 40})

    h_corridor1 = y0 + W * 0.4
    msp.add_line((x0, h_corridor1), (x0+L, h_corridor1), dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30})
    h_corridor2 = y0 + W * 0.7
    msp.add_line((x0, h_corridor2), (x0+L, h_corridor2), dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30})
    v1 = x0 + L * 0.3
    msp.add_line((v1, y0), (v1, y0+W), dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30})
    v2 = x0 + L * 0.6
    msp.add_line((v2, y0), (v2, y0+W), dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30})

    door_positions = [
        (x0+0.1*L, y0+0.1*W),
        (x0+0.4*L, y0+0.5*W),
        (x0+0.7*L, y0+0.2*W),
        (x0+0.9*L, y0+0.8*W)
    ]
    for (dx, dy) in door_positions:
        msp.add_lwpolyline(
            [(dx, dy), (dx+door_w, dy), (dx+door_w, dy+door_h), (dx, dy+door_h), (dx, dy)],
            dxfattribs={'layer': 'A-DOOR', 'color': 3, 'lineweight': 20}
        )
        msp.add_arc((dx, dy), radius=door_w, start_angle=0, end_angle=90,
                    dxfattribs={'layer': 'A-DOOR', 'color': 3})

    win_positions = [
        (x0+0.1*L, y0+0.9*W),
        (x0+0.5*L, y0+0.9*W),
        (x0+0.9*L, y0+0.1*W),
        (x0+0.9*L, y0+0.4*W)
    ]
    for (wx, wy) in win_positions:
        msp.add_lwpolyline(
            [(wx, wy), (wx+win_w, wy), (wx+win_w, wy+win_h), (wx, wy+win_h), (wx, wy)],
            dxfattribs={'layer': 'A-WINDOW', 'color': 5, 'lineweight': 15}
        )

    room_texts = [
        ("Living Room", x0+0.15*L, y0+0.2*W),
        ("Kitchen", x0+0.45*L, y0+0.5*W),
        ("Bedroom 1", x0+0.75*L, y0+0.15*W),
        ("Bedroom 2", x0+0.75*L, y0+0.6*W),
        ("Bathroom", x0+0.15*L, y0+0.7*W)
    ]
    for (txt, tx, ty) in room_texts:
        msp.add_text(
            txt,
            dxfattribs={'layer': 'ANNO-TEXT', 'height': 80, 'color': 4}
        ).set_pos((tx, ty), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_line((x0, y0-100), (x0+L, y0-100), dxfattribs={'layer': 'ANNO-DIMS', 'color': 2})
    msp.add_text(
        f"L = {L/1000:.2f} m",
        dxfattribs={'layer': 'ANNO-TEXT', 'height': 50, 'color': 2}
    ).set_pos((x0+L/2, y0-150), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_line((x0-100, y0), (x0-100, y0+W), dxfattribs={'layer': 'ANNO-DIMS', 'color': 2})
    msp.add_text(
        f"W = {W/1000:.2f} m",
        dxfattribs={'layer': 'ANNO-TEXT', 'height': 50, 'color': 2}
    ).set_pos((x0-150, y0+W/2), align=TextEntityAlignment.MIDDLE_CENTER)

    # ========== STRUCTURAL LAYER ==========
    cols_x = np.arange(x0, x0+L+col_sp_x, col_sp_x)
    cols_y = np.arange(y0, y0+W+col_sp_y, col_sp_y)
    for cx in cols_x:
        for cy in cols_y:
            msp.add_circle(
                (cx, cy), radius=100,
                dxfattribs={'layer': 'S-COLUMN', 'color': 6, 'lineweight': 30}
            )
            msp.add_text(
                f"C{int((cx-x0)/1000)+1}{int((cy-y0)/1000)+1}",
                dxfattribs={'layer': 'ANNO-TEXT', 'height': 40, 'color': 6}
            ).set_pos((cx+120, cy), align=TextEntityAlignment.MIDDLE_LEFT)

    for cx in cols_x:
        for cy in cols_y:
            if cx < x0+L - col_sp_x:
                msp.add_line((cx, cy), (cx+col_sp_x, cy),
                             dxfattribs={'layer': 'S-BEAM', 'color': 7, 'lineweight': 50})
            if cy < y0+W - col_sp_y:
                msp.add_line((cx, cy), (cx, cy+col_sp_y),
                             dxfattribs={'layer': 'S-BEAM', 'color': 7, 'lineweight': 50})

    slab_pts = [(x0, y0), (x0+L, y0), (x0+L, y0+W), (x0, y0+W)]
    msp.add_lwpolyline(slab_pts, dxfattribs={'layer': 'S-SLAB', 'color': 8, 'lineweight': 20})

    for cx in cols_x:
        for cy in cols_y:
            msp.add_lwpolyline(
                [(cx-footing_l/2, cy-footing_w/2),
                 (cx+footing_l/2, cy-footing_w/2),
                 (cx+footing_l/2, cy+footing_w/2),
                 (cx-footing_l/2, cy+footing_w/2),
                 (cx-footing_l/2, cy-footing_w/2)],
                dxfattribs={'layer': 'S-FOOTING', 'color': 9, 'lineweight': 40}
            )

    for cx in cols_x:
        for cy in cols_y:
            msp.add_circle((cx, cy), radius=rebar_main_d/2,
                           dxfattribs={'layer': 'S-REBAR', 'color': 10, 'lineweight': 15})
            stirrup_offset = 20
            msp.add_lwpolyline(
                [(cx-100-stirrup_offset, cy-100-stirrup_offset),
                 (cx+100+stirrup_offset, cy-100-stirrup_offset),
                 (cx+100+stirrup_offset, cy+100+stirrup_offset),
                 (cx-100-stirrup_offset, cy+100+stirrup_offset),
                 (cx-100-stirrup_offset, cy-100-stirrup_offset)],
                dxfattribs={'layer': 'S-REBAR', 'color': 10, 'lineweight': 15}
            )

    # ========== BOQ CALCULATIONS ==========
    num_columns = len(cols_x) * len(cols_y)
    slab_vol = (L/1000) * (W/1000) * (slab_t/1000) * num_floors
    col_vol = num_columns * (0.2*0.2) * (floor_h/1000) * num_floors
    beam_vol = ( (len(cols_x)-1) * len(cols_y) * (beam_w/1000) * (beam_d/1000) * (col_sp_x/1000)
                + (len(cols_y)-1) * len(cols_x) * (beam_w/1000) * (beam_d/1000) * (col_sp_y/1000) ) * num_floors
    footing_vol = num_columns * (footing_l/1000) * (footing_w/1000) * (footing_d/1000)
    total_concrete = slab_vol + col_vol + beam_vol + footing_vol

    main_len = (floor_h/1000) * 4 * num_columns * num_floors
    main_weight = main_len * (np.pi*(rebar_main_d/1000)**2/4 * 7850) / 1000
    stirrup_perim = 2*(0.2+0.2)*1000
    num_stirrups = (floor_h/1000 / (rebar_sp/1000) + 1) * num_columns * num_floors
    stirrup_len = stirrup_perim * num_stirrups / 1000
    stirrup_weight = stirrup_len * (np.pi*(rebar_stirrup_d/1000)**2/4 * 7850) / 1000
    total_rebar = main_weight + stirrup_weight

    formwork = 2*(L/1000 + W/1000) * (floor_h/1000) * num_floors
    internal_wall_length = (L/1000) * 2 + (W/1000) * 2
    wall_area = internal_wall_length * (floor_h/1000) * num_floors
    brick_count = wall_area * 50
    flooring_area = (L/1000) * (W/1000) * num_floors
    paint_area = wall_area * 2

    boq_items = [
        {'Item': 'Concrete (m³)', 'Quantity': total_concrete, 'Unit': 'm³'},
        {'Item': 'Rebar (ton)', 'Quantity': total_rebar/1000, 'Unit': 'ton'},
        {'Item': 'Formwork (m²)', 'Quantity': formwork, 'Unit': 'm²'},
        {'Item': 'Bricks (nos)', 'Quantity': brick_count, 'Unit': 'nos'},
        {'Item': 'Flooring (m²)', 'Quantity': flooring_area, 'Unit': 'm²'},
        {'Item': 'Paint (m²)', 'Quantity': paint_area, 'Unit': 'm²'},
    ]
    for item in boq_items:
        key = item['Item'].split('(')[0].strip().lower().replace(' ', '_')
        rate = BOQ_RATES.get(key, 0)
        item['Unit Rate (EGP)'] = rate
        item['Total Cost (EGP)'] = round(item['Quantity'] * rate, 2)

    boq_df = pd.DataFrame(boq_items)

    layout_info = {
        'plot_area': plot_area,
        'street_width': street_width,
        'location': params.get('location', ''),
        'max_floors': max_floors,
        'num_floors': num_floors,
        'footprint_area': round((L/1000)*(W/1000), 2),
        'building_width': round(W/1000, 2),
        'building_length': round(L/1000, 2),
        'front_setback': front_setback,
        'rear_setback': rear_setback,
        'side_setback': side_setback,
    }

    dxf_buffer = io.BytesIO()
    doc.write(dxf_buffer)
    dxf_bytes = dxf_buffer.getvalue()

    return {
        'dxf': dxf_bytes,
        'boq': boq_df.to_dict('records'),
        'info': layout_info
    }
