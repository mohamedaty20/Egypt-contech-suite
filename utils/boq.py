import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from config import UNIT_RATES, MASS_SCHEMAS, ARCH_SCHEMAS

def normalize_keys(obj, aliases):
    new_obj = {}
    for k, v in obj.items():
        if k in aliases:
            new_obj[aliases[k]] = v
        else:
            new_obj[k] = v
    return new_obj

def compute_architectural_quantities(element_type, data, user_params):
    schema_info = ARCH_SCHEMAS.get(element_type)
    if not schema_info:
        return [], 0, []

    required = schema_info['required']
    results = []
    total_quantity = 0
    missing_fields = []

    for req in required:
        if req not in data or data[req] is None:
            missing_fields.append(req)

    if missing_fields:
        return results, total_quantity, [{'label': 'General', 'idx': 0, 'missing': missing_fields}]

    if element_type in ['flooring', 'wall_finishing', 'ceilings']:
        qty = schema_info['formula'](data) if callable(schema_info['formula']) else 0
        total_quantity += qty
        results.append({
            'label': element_type.capitalize(),
            'quantity': qty,
            'unit': 'm²'
        })
    elif element_type == 'doors_windows':
        door_count, window_count = schema_info['formula'](data)
        if door_count:
            results.append({
                'label': 'Doors',
                'quantity': door_count,
                'unit': 'nos'
            })
            total_quantity += door_count
        if window_count:
            results.append({
                'label': 'Windows',
                'quantity': window_count,
                'unit': 'nos'
            })
            total_quantity += window_count

    return results, total_quantity, []

def generate_arch_boq_table(results, element_type, wastage):
    rows = []
    for r in results:
        rows.append({
            'Item': f"{element_type.capitalize()} - {r['label']}",
            'Count': 1,
            'Unit': r['unit'],
            'Quantity (net)': round(r['quantity'], 2),
            'Wastage %': wastage,
            'Quantity (with waste)': round(r['quantity'] * (1 + wastage/100), 2),
            'Unit Rate (EGP)': round(UNIT_RATES.get(r['label'], 0), 2),
            'Total Cost (EGP)': round(r['quantity'] * (1 + wastage/100) * UNIT_RATES.get(r['label'], 0), 2)
        })
    if rows:
        total_row = {
            'Item': 'TOTAL',
            'Count': '',
            'Unit': '',
            'Quantity (net)': round(sum(r['Quantity (net)'] for r in rows), 2),
            'Wastage %': '',
            'Quantity (with waste)': round(sum(r['Quantity (with waste)'] for r in rows), 2),
            'Unit Rate (EGP)': '',
            'Total Cost (EGP)': round(sum(r['Total Cost (EGP)'] for r in rows), 2)
        }
        rows.append(total_row)
    return pd.DataFrame(rows)

def compute_mass_from_ai_data(element_type, data, user_params):
    schema_info = MASS_SCHEMAS.get(element_type)
    required = schema_info['required']
    results = []
    total_concrete = 0

    for group in data:
        all_present = True
        for req in required:
            if req not in group or group[req] is None:
                all_present = False
                break
        if not all_present:
            continue

        if element_type == 'columns':
            if group.get('height_mm') is None and user_params.get('use_floor_height', False):
                group['height_mm'] = user_params.get('floor_height_mm', 3000)
            if group.get('height_mm') is None:
                continue
            vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (group['height_mm']/1000) * group.get('count', 1)
        elif element_type == 'beams':
            vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (group['length_mm']/1000) * group.get('count', 1)
        elif element_type == 'slabs':
            vol = group['area_m2'] * (group['thickness_mm']/1000)
        elif element_type == 'footings':
            vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (group['length_mm']/1000) * group.get('count', 1)
        elif element_type == 'walls':
            vol = group['length_m'] * group['height_m'] * (group['thickness_mm']/1000) * group.get('count', 1)
        else:
            vol = 0

        total_concrete += vol
        results.append({
            'label': group.get('label', 'Unknown'),
            'count': group.get('count', 1),
            'concrete_m3': vol,
            'rebar_ton': 0
        })

    total_concrete = round(total_concrete, 2)
    return results, total_concrete, 0

def generate_boq_table(results, branch, element_type, wastage, mode):
    rows = []
    if mode == 'mass':
        for r in results:
            if 'concrete_m3' in r:
                rows.append({
                    'Item': f"{element_type.capitalize()} - {r.get('label', '')}",
                    'Count': r.get('count', 1),
                    'Unit': 'm³',
                    'Quantity (net)': round(r['concrete_m3'], 2),
                    'Wastage %': wastage,
                    'Quantity (with waste)': round(r['concrete_m3'] * (1 + wastage/100), 2),
                    'Unit Rate (EGP)': round(UNIT_RATES.get('Concrete (C30/37)', 2500), 2),
                    'Total Cost (EGP)': round(r['concrete_m3'] * (1 + wastage/100) * UNIT_RATES.get('Concrete (C30/37)', 2500), 2)
                })
    else:  # rebar
        for r in results:
            if 'rebar_ton' in r:
                rows.append({
                    'Item': f"{element_type.capitalize()} - {r.get('label', '')} - Rebar",
                    'Count': r.get('count', 1),
                    'Unit': 'ton',
                    'Quantity (net)': round(r['rebar_ton'], 2),
                    'Wastage %': wastage,
                    'Quantity (with waste)': round(r['rebar_ton'] * (1 + wastage/100), 2),
                    'Unit Rate (EGP)': round(UNIT_RATES.get('Rebar (Grade 400)', 15000), 2),
                    'Total Cost (EGP)': round(r['rebar_ton'] * (1 + wastage/100) * UNIT_RATES.get('Rebar (Grade 400)', 15000), 2)
                })
            if 'concrete_m3' in r:
                rows.append({
                    'Item': f"{element_type.capitalize()} - {r.get('label', '')} - Concrete",
                    'Count': r.get('count', 1),
                    'Unit': 'm³',
                    'Quantity (net)': round(r['concrete_m3'], 2),
                    'Wastage %': wastage,
                    'Quantity (with waste)': round(r['concrete_m3'] * (1 + wastage/100), 2),
                    'Unit Rate (EGP)': round(UNIT_RATES.get('Concrete (C30/37)', 2500), 2),
                    'Total Cost (EGP)': round(r['concrete_m3'] * (1 + wastage/100) * UNIT_RATES.get('Concrete (C30/37)', 2500), 2)
                })
    if rows:
        total_row = {
            'Item': 'TOTAL',
            'Count': '',
            'Unit': '',
            'Quantity (net)': round(sum(r['Quantity (net)'] for r in rows), 2),
            'Wastage %': '',
            'Quantity (with waste)': round(sum(r['Quantity (with waste)'] for r in rows), 2),
            'Unit Rate (EGP)': '',
            'Total Cost (EGP)': round(sum(r['Total Cost (EGP)'] for r in rows), 2)
        }
        rows.append(total_row)
    return pd.DataFrame(rows)

def compute_rebar_quantities(element_type, data, user_params):
    results = []
    total_concrete = 0
    total_rebar = 0
    for group in data:
        required = ['label', 'count', 'width_mm', 'depth_mm', 'height_mm', 'rebar']
        all_present = True
        for req in required:
            if req not in group or group[req] is None:
                all_present = False
                break
        if not all_present:
            continue
        height = group.get('height_mm') or user_params.get('floor_height_mm', 3000)
        vol = (group['width_mm']/1000) * (group['depth_mm']/1000) * (height/1000) * group.get('count', 1)
        total_concrete += vol
        rebar = group.get('rebar', {})
        main_d = rebar.get('main_diameter_mm', 0)
        stirrup_d = rebar.get('stirrup_diameter_mm', 0)
        spacing = rebar.get('spacing_mm', 200)
        count = group.get('count', 1)
        height_m = height / 1000
        main_length = height_m * 4 * count
        perimeter = 2 * ((group['width_mm'] + group['depth_mm']) / 1000)
        num_stirrups = (height_m / (spacing/1000)) + 1
        stirrup_length = perimeter * num_stirrups * count
        main_weight = main_length * ( (3.1416 * (main_d/1000)**2 / 4) * 7850 )
        stirrup_weight = stirrup_length * ( (3.1416 * (stirrup_d/1000)**2 / 4) * 7850 )
        total_rebar += (main_weight + stirrup_weight)
        results.append({
            'label': group.get('label', 'Unknown'),
            'count': count,
            'concrete_m3': vol,
            'rebar_ton': (main_weight + stirrup_weight) / 1000
        })
    total_concrete = round(total_concrete, 2)
    total_rebar = round(total_rebar / 1000, 2)
    return results, total_concrete, total_rebar

def generate_charts(df, element_type):
    df_no_total = df[df['Item'] != 'TOTAL'].copy()
    if df_no_total.empty:
        return None, None

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=df_no_total['Item'],
        y=df_no_total['Quantity (net)'],
        name='Concrete Volume (m³)',
        marker_color='#FF8C00',
        text=df_no_total['Quantity (net)'],
        textposition='auto',
    ))
    fig_bar.update_layout(
        title=f'{element_type.capitalize()} - Concrete Volume per Group',
        template='plotly_dark',
        paper_bgcolor='#0d1a35',
        plot_bgcolor='#0d1a35',
        font=dict(color='white'),
        margin=dict(t=40, b=20, l=40, r=20),
        height=400,
        xaxis_tickangle=-45,
    )

    fig_pie = go.Figure(data=[go.Pie(
        labels=df_no_total['Item'],
        values=df_no_total['Total Cost (EGP)'],
        hole=0.4,
        marker=dict(colors=px.colors.sequential.Oranges_r),
        textinfo='label+percent',
        textposition='auto',
    )])
    fig_pie.update_layout(
        title=f'{element_type.capitalize()} - Cost Distribution',
        template='plotly_dark',
        paper_bgcolor='#0d1a35',
        plot_bgcolor='#0d1a35',
        font=dict(color='white'),
        margin=dict(t=40, b=20, l=40, r=20),
        height=400,
    )
    return fig_bar, fig_pie
