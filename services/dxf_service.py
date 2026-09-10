# services/dxf_service.py – Complete upgraded version

import io
import math
import numpy as np
import pandas as pd
import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from config import BOQ_RATES

# ----------------------------------------------------------------------
# Helper functions (keep these)
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

def create_dxf_layer(doc, name, color, lineweight=25, linetype='CONTINUOUS'):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=linetype, lineweight=lineweight)

# ----------------------------------------------------------------------
# Main generator – completely rewritten for professional layouts
# ----------------------------------------------------------------------

def build_complete_project(params):
    """
    Enhanced professional layout generator.
    Produces a complete architectural + structural DXF with detailed BOQ.
    """
    # ----- 1. Determine building polygon (with repair) -----
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
        # Create a rectangle from area and aspect ratio
        area = params['plot_area_m2']
        aspect = 1.5
        width = np.sqrt(area / aspect)
        length = area / width
        plot_poly = box(0, 0, length, width)

    # ----- 2. Setbacks (Egyptian practice) -----
    street_width = params.get('street_width_m', 10)
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

    # Shrink polygon by setbacks (using buffer with negative distance)
    # For simplicity, we use the bounding box offset method (more predictable)
    min_x, min_y, max_x, max_y = plot_poly.bounds
    building_length = max_x - min_x - front_setback - rear_setback
    building_width = max_y - min_y - 2 * side_setback
    if building_length <= 0 or building_width <= 0:
        # fallback to percentage of plot area
        footprint_ratio = 0.6
        max_footprint = plot_poly.area * footprint_ratio
        b_w = 10.0
        b_l = max_footprint / b_w
        building_length = b_l
        building_width = b_w

    # Ensure minimum dimensions
    building_length = max(building_length, 5.0)
    building_width = max(building_width, 5.0)

    # ----- 3. Max floors based on street width -----
    if street_width >= 12:
        max_floors = 4
    elif street_width >= 8:
        max_floors = 3
    elif street_width >= 6:
        max_floors = 2
    else:
        max_floors = 1
    num_floors = min(params.get('num_floors', max_floors), max_floors)
    floor_height_m = params.get('floor_height_m', 3.0)

    # ----- 4. Extract architectural parameters -----
    wall_thick = params.get('wall_thickness_mm', 200)
    door_w = params.get('door_width_mm', 900)
    door_h = params.get('door_height_mm', 2100)
    win_w = params.get('window_width_mm', 1200)
    win_h = params.get('window_height_mm', 1200)

    # ----- 5. Number of units per floor (new) -----
    num_units = params.get('num_units_per_floor', 2)  # default 2

    # ----- 6. Structural parameters -----
    col_sp_x = params.get('column_spacing_x', 4.0) * 1000   # in mm
    col_sp_y = params.get('column_spacing_y', 4.0) * 1000
    beam_w = params.get('beam_width_mm', 300)
    beam_d = params.get('beam_depth_mm', 500)
    slab_t = params.get('slab_thickness_mm', 150)
    footing_w = params.get('footing_width_mm', 800)
    footing_d = params.get('footing_depth_mm', 400)
    footing_l = params.get('footing_length_mm', 800)
    rebar_main_d = params.get('rebar_main_diam_mm', 16)
    rebar_stirrup_d = params.get('rebar_stirrup_diam_mm', 10)
    rebar_sp = params.get('rebar_spacing_mm', 200)

    # Convert to mm
    L_mm = building_length * 1000
    W_mm = building_width * 1000
    # We'll place the building with its bottom-left at (0,0) for simplicity
    x0 = 0.0
    y0 = 0.0

    # ----- 7. Create DXF document with professional layers -----
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    doc.header['$INSUNITS'] = 4   # millimeters

    # Layer definitions (AIA standard)
    layers_def = {
        'A-WALL':          {'color': 7, 'lineweight': 40},
        'A-WALL-INT':      {'color': 8, 'lineweight': 30},
        'A-DOOR':          {'color': 3, 'lineweight': 20},
        'A-WINDOW':        {'color': 5, 'lineweight': 15},
        'A-ROOM-TEXT':     {'color': 4, 'lineweight': 10},
        'A-FURN':          {'color': 6, 'lineweight': 10},
        'S-COLUMN':        {'color': 6, 'lineweight': 30},
        'S-BEAM':          {'color': 7, 'lineweight': 50},
        'S-SLAB':          {'color': 8, 'lineweight': 20},
        'S-FOOTING':       {'color': 9, 'lineweight': 40},
        'S-REBAR':         {'color': 10, 'lineweight': 15},
        'ANNO-DIMS':       {'color': 2, 'lineweight': 15},
        'ANNO-TEXT':       {'color': 4, 'lineweight': 10},
        'ANNO-HATCH':      {'color': 1, 'lineweight': 5},
        'A-CORE':          {'color': 4, 'lineweight': 20},   # core walls
    }
    for name, props in layers_def.items():
        create_dxf_layer(doc, name, props['color'], lineweight=props['lineweight'])

    # ----- 8. LAYOUT GENERATION -----

    # 8a. Divide building into units and core
    # We'll create a simple corridor in the middle and units on each side.
    # For more units, we split the length.
    if num_units == 1:
        unit_areas = [(0, L_mm, 0, W_mm)]   # single unit
    else:
        # Create a central core (stairs/elevator) at the middle, then split remaining space
        core_width = 2500   # mm
        core_depth = 2000   # mm
        # Place core at the center
        core_x = (L_mm - core_width) / 2 + x0
        core_y = (W_mm - core_depth) / 2 + y0
        # Draw core rectangle (will be filled later)
        msp.add_lwpolyline([
            (core_x, core_y),
            (core_x + core_width, core_y),
            (core_x + core_width, core_y + core_depth),
            (core_x, core_y + core_depth),
            (core_x, core_y)
        ], dxfattribs={'layer': 'A-CORE', 'color': 4, 'lineweight': 20})
        msp.add_text("CORE", dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 200, 'color': 4}
                     ).set_pos((core_x + core_width/2, core_y + core_depth/2),
                               align=TextEntityAlignment.MIDDLE_CENTER)
        # Divide the remaining space into units (left and right of core)
        # We'll create two units: left and right, each with rooms
        # Left unit: from x0 to core_x, full height
        left_width = core_x - x0
        right_width = (x0 + L_mm) - (core_x + core_width)
        # For simplicity, we assign both sides to units
        unit_areas = [
            (x0, core_x, y0, y0 + W_mm),          # left unit
            (core_x + core_width, x0 + L_mm, y0, y0 + W_mm)  # right unit
        ]
        # If more than 2 units, we could split vertically as well, but we'll keep it simple.

    # 8b. For each unit, generate rooms
    # We'll create a simple room layout: living, kitchen, bathroom, bedrooms
    # This is a basic example – you can expand it.
    room_defs = {
        'Living Room': {'min_area': 15, 'prefer_width': 1.0, 'prefer_depth': 1.0},
        'Kitchen':     {'min_area': 8,  'prefer_width': 0.8, 'prefer_depth': 0.8},
        'Bathroom':    {'min_area': 4,  'prefer_width': 0.6, 'prefer_depth': 0.6},
        'Bedroom 1':   {'min_area': 12, 'prefer_width': 0.9, 'prefer_depth': 0.9},
        'Bedroom 2':   {'min_area': 10, 'prefer_width': 0.8, 'prefer_depth': 0.8},
    }

    # Function to generate rooms within a given rectangle (x1,x2,y1,y2)
    def generate_rooms(x1, x2, y1, y2, unit_label):
        # Compute available area
        width = x2 - x1
        depth = y2 - y1
        total_area = width * depth / (1000*1000)  # in m²
        # We'll roughly allocate areas
        # For simplicity, we divide the rectangle into a grid
        # We'll create a corridor along the long side
        if width > depth:
            # horizontal corridor along the side
            corridor_width = 1000  # mm
            # We'll split the depth into rooms
            num_rooms = len(room_defs)
            # We'll stack rooms along the depth
            room_depth = (depth - corridor_width) / num_rooms
            # Create rooms
            room_rects = []
            for i, (name, props) in enumerate(room_defs.items()):
                y_start = y1 + i * room_depth
                y_end = y_start + room_depth
                # Leave a gap for internal walls
                x_start = x1 + 100
                x_end = x2 - 100
                # Add room rectangle
                msp.add_lwpolyline([
                    (x_start, y_start),
                    (x_end, y_start),
                    (x_end, y_end),
                    (x_start, y_end),
                    (x_start, y_start)
                ], dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30})
                # Room label
                msp.add_text(f"{name}\nUnit {unit_label}",
                             dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 150, 'color': 4}
                             ).set_pos(( (x_start+x_end)/2, (y_start+y_end)/2 ),
                                       align=TextEntityAlignment.MIDDLE_CENTER)
                # Place doors (simple: at the middle of the wall facing the corridor)
                door_x = (x_start + x_end) / 2
                door_y = y_start  # bottom wall (towards corridor)
                msp.add_lwpolyline([
                    (door_x - door_w/2, door_y),
                    (door_x + door_w/2, door_y),
                    (door_x + door_w/2, door_y + door_h),
                    (door_x - door_w/2, door_y + door_h),
                    (door_x - door_w/2, door_y)
                ], dxfattribs={'layer': 'A-DOOR', 'color': 3, 'lineweight': 20})
                msp.add_arc((door_x - door_w/2, door_y), radius=door_w,
                            start_angle=0, end_angle=90,
                            dxfattribs={'layer': 'A-DOOR', 'color': 3})
                # Place windows (on outer walls if possible)
                # For simplicity, put one window on the outer wall
                win_x = (x_start + x_end) / 2
                if i % 2 == 0:
                    # top wall
                    win_y = y_end
                else:
                    # bottom wall
                    win_y = y_start
                msp.add_lwpolyline([
                    (win_x - win_w/2, win_y),
                    (win_x + win_w/2, win_y),
                    (win_x + win_w/2, win_y + win_h),
                    (win_x - win_w/2, win_y + win_h),
                    (win_x - win_w/2, win_y)
                ], dxfattribs={'layer': 'A-WINDOW', 'color': 5, 'lineweight': 15})
                room_rects.append((x_start, x_end, y_start, y_end))
        else:
            # vertical corridor
            corridor_width = 1000
            room_width = (width - corridor_width) / num_rooms
            for i, (name, props) in enumerate(room_defs.items()):
                x_start = x1 + i * room_width
                x_end = x_start + room_width
                y_start = y1 + 100
                y_end = y2 - 100
                msp.add_lwpolyline([
                    (x_start, y_start),
                    (x_end, y_start),
                    (x_end, y_end),
                    (x_start, y_end),
                    (x_start, y_start)
                ], dxfattribs={'layer': 'A-WALL-INT', 'color': 8, 'lineweight': 30})
                msp.add_text(f"{name}\nUnit {unit_label}",
                             dxfattribs={'layer': 'A-ROOM-TEXT', 'height': 150, 'color': 4}
                             ).set_pos(( (x_start+x_end)/2, (y_start+y_end)/2 ),
                                       align=TextEntityAlignment.MIDDLE_CENTER)
                # Door
                door_x = x_start
                door_y = (y_start + y_end) / 2
                msp.add_lwpolyline([
                    (door_x, door_y - door_w/2),
                    (door_x, door_y + door_w/2),
                    (door_x + door_h, door_y + door_w/2),
                    (door_x + door_h, door_y - door_w/2),
                    (door_x, door_y - door_w/2)
                ], dxfattribs={'layer': 'A-DOOR', 'color': 3, 'lineweight': 20})
                msp.add_arc((door_x, door_y - door_w/2), radius=door_w,
                            start_angle=0, end_angle=90,
                            dxfattribs={'layer': 'A-DOOR', 'color': 3})
                # Window
                win_x = x_end if i%2==0 else x_start
                win_y = (y_start + y_end) / 2
                msp.add_lwpolyline([
                    (win_x, win_y - win_w/2),
                    (win_x, win_y + win_w/2),
                    (win_x + win_h, win_y + win_w/2),
                    (win_x + win_h, win_y - win_w/2),
                    (win_x, win_y - win_w/2)
                ], dxfattribs={'layer': 'A-WINDOW', 'color': 5, 'lineweight': 15})

    # Generate rooms for each unit
    for idx, (ux1, ux2, uy1, uy2) in enumerate(unit_areas):
        generate_rooms(ux1, ux2, uy1, uy2, idx+1)

    # ----- 9. STRUCTURAL GRID -----
    # Place columns at intersections of grid lines
    cols_x = np.arange(x0, x0 + L_mm + col_sp_x, col_sp_x)
    cols_y = np.arange(y0, y0 + W_mm + col_sp_y, col_sp_y)
    for cx in cols_x:
        for cy in cols_y:
            # Skip if column is inside core (optional)
            msp.add_circle(
                (cx, cy), radius=100,
                dxfattribs={'layer': 'S-COLUMN', 'color': 6, 'lineweight': 30}
            )
            msp.add_text(
                f"C{int((cx-x0)/1000)+1}{int((cy-y0)/1000)+1}",
                dxfattribs={'layer': 'ANNO-TEXT', 'height': 40, 'color': 6}
            ).set_pos((cx+120, cy), align=TextEntityAlignment.MIDDLE_LEFT)

    # Beams (between columns)
    for cx in cols_x:
        for cy in cols_y:
            if cx < x0 + L_mm - col_sp_x:
                msp.add_line((cx, cy), (cx+col_sp_x, cy),
                             dxfattribs={'layer': 'S-BEAM', 'color': 7, 'lineweight': 50})
            if cy < y0 + W_mm - col_sp_y:
                msp.add_line((cx, cy), (cx, cy+col_sp_y),
                             dxfattribs={'layer': 'S-BEAM', 'color': 7, 'lineweight': 50})

    # Slab boundary
    slab_pts = [(x0, y0), (x0+L_mm, y0), (x0+L_mm, y0+W_mm), (x0, y0+W_mm)]
    msp.add_lwpolyline(slab_pts, dxfattribs={'layer': 'S-SLAB', 'color': 8, 'lineweight': 20})

    # Footings under columns
    for cx in cols_x:
        for cy in cols_y:
            msp.add_lwpolyline([
                (cx-footing_l/2, cy-footing_w/2),
                (cx+footing_l/2, cy-footing_w/2),
                (cx+footing_l/2, cy+footing_w/2),
                (cx-footing_l/2, cy+footing_w/2),
                (cx-footing_l/2, cy-footing_w/2)
            ], dxfattribs={'layer': 'S-FOOTING', 'color': 9, 'lineweight': 40})

    # Rebar (main bars and stirrups) – simplified
    for cx in cols_x:
        for cy in cols_y:
            msp.add_circle((cx, cy), radius=rebar_main_d/2,
                           dxfattribs={'layer': 'S-REBAR', 'color': 10, 'lineweight': 15})
            stirrup_offset = 20
            msp.add_lwpolyline([
                (cx-100-stirrup_offset, cy-100-stirrup_offset),
                (cx+100+stirrup_offset, cy-100-stirrup_offset),
                (cx+100+stirrup_offset, cy+100+stirrup_offset),
                (cx-100-stirrup_offset, cy+100+stirrup_offset),
                (cx-100-stirrup_offset, cy-100-stirrup_offset)
            ], dxfattribs={'layer': 'S-REBAR', 'color': 10, 'lineweight': 15})

    # ----- 10. DIMENSION LINES -----
    msp.add_line((x0, y0-100), (x0+L_mm, y0-100), dxfattribs={'layer': 'ANNO-DIMS', 'color': 2})
    msp.add_text(
        f"L = {L_mm/1000:.2f} m",
        dxfattribs={'layer': 'ANNO-TEXT', 'height': 50, 'color': 2}
    ).set_pos((x0+L_mm/2, y0-150), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_line((x0-100, y0), (x0-100, y0+W_mm), dxfattribs={'layer': 'ANNO-DIMS', 'color': 2})
    msp.add_text(
        f"W = {W_mm/1000:.2f} m",
        dxfattribs={'layer': 'ANNO-TEXT', 'height': 50, 'color': 2}
    ).set_pos((x0-150, y0+W_mm/2), align=TextEntityAlignment.MIDDLE_CENTER)

    # ----- 11. BOQ CALCULATIONS (enhanced) -----
    num_columns = len(cols_x) * len(cols_y)

    # Concrete volumes
    slab_vol = (L_mm/1000) * (W_mm/1000) * (slab_t/1000) * num_floors
    col_vol = num_columns * (0.2*0.2) * (floor_height_m) * num_floors  # assume 200x200 columns
    beam_vol = ( (len(cols_x)-1) * len(cols_y) * (beam_w/1000) * (beam_d/1000) * (col_sp_x/1000)
                + (len(cols_y)-1) * len(cols_x) * (beam_w/1000) * (beam_d/1000) * (col_sp_y/1000) ) * num_floors
    footing_vol = num_columns * (footing_l/1000) * (footing_w/1000) * (footing_d/1000)
    total_concrete = slab_vol + col_vol + beam_vol + footing_vol

    # Rebar weight
    main_len = (floor_height_m) * 4 * num_columns * num_floors
    main_weight = main_len * (np.pi*(rebar_main_d/1000)**2/4 * 7850) / 1000  # kg
    stirrup_perim = 2*(0.2+0.2)*1000  # mm
    num_stirrups = (floor_height_m / (rebar_sp/1000) + 1) * num_columns * num_floors
    stirrup_len = stirrup_perim * num_stirrups / 1000  # m
    stirrup_weight = stirrup_len * (np.pi*(rebar_stirrup_d/1000)**2/4 * 7850) / 1000
    total_rebar = main_weight + stirrup_weight  # kg

    # Formwork
    formwork = 2*(L_mm/1000 + W_mm/1000) * floor_height_m * num_floors

    # Bricks (for walls) – we approximate total wall length from internal walls
    # We'll calculate based on the number of rooms and corridors
    # For simplicity, we use a rough factor: wall area = 0.6 * floor area per floor
    total_floor_area = (L_mm/1000) * (W_mm/1000) * num_floors
    wall_area = total_floor_area * 0.6  # rough estimate
    brick_count = wall_area * 50  # 50 bricks per m²

    # Flooring
    flooring_area = total_floor_area

    # Paint (walls)
    paint_area = wall_area * 2  # both sides

    # Windows and doors count (from the layout)
    # We'll count the number of windows and doors we added
    # We can compute from the room generation loops – but for simplicity, we estimate:
    num_windows = 2 * num_units * len(room_defs)  # 2 windows per room average
    num_doors = num_windows  # similar

    # Build BOQ DataFrame
    boq_items = [
        {'Item': 'Concrete (m³)', 'Quantity': total_concrete, 'Unit': 'm³'},
        {'Item': 'Rebar (ton)', 'Quantity': total_rebar/1000, 'Unit': 'ton'},
        {'Item': 'Formwork (m²)', 'Quantity': formwork, 'Unit': 'm²'},
        {'Item': 'Bricks (nos)', 'Quantity': brick_count, 'Unit': 'nos'},
        {'Item': 'Flooring (m²)', 'Quantity': flooring_area, 'Unit': 'm²'},
        {'Item': 'Paint (m²)', 'Quantity': paint_area, 'Unit': 'm²'},
        {'Item': 'Windows (nos)', 'Quantity': num_windows, 'Unit': 'nos'},
        {'Item': 'Doors (nos)', 'Quantity': num_doors, 'Unit': 'nos'},
    ]
    for item in boq_items:
        key = item['Item'].split('(')[0].strip().lower().replace(' ', '_')
        rate = BOQ_RATES.get(key, 0)
        item['Unit Rate (EGP)'] = rate
        item['Total Cost (EGP)'] = round(item['Quantity'] * rate, 2)

    boq_df = pd.DataFrame(boq_items)

    layout_info = {
        'plot_area': plot_poly.area,
        'street_width': street_width,
        'location': params.get('location', ''),
        'max_floors': max_floors,
        'num_floors': num_floors,
        'footprint_area': round((L_mm/1000)*(W_mm/1000), 2),
        'building_width': round(W_mm/1000, 2),
        'building_length': round(L_mm/1000, 2),
        'front_setback': front_setback,
        'rear_setback': rear_setback,
        'side_setback': side_setback,
        'num_units': num_units,
    }

    # Write DXF to bytes
    dxf_buffer = io.BytesIO()
    doc.write(dxf_buffer)
    dxf_bytes = dxf_buffer.getvalue()

    return {
        'dxf': dxf_bytes,
        'boq': boq_df.to_dict('records'),
        'info': layout_info
    }
