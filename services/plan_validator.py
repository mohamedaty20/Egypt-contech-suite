"""
services/plan_validator.py

Spatial and circulation validator for AI-generated floor plans.
Returns a list of violation dicts.
"""

CORRIDOR_EDGE_TOL = 350


def _rect_inside(inner, outer, margin=0):
    return (inner[0] >= outer[0] - margin
            and inner[1] >= outer[1] - margin
            and inner[2] <= outer[2] + margin
            and inner[3] <= outer[3] + margin)


def _rects_overlap(a, b, margin=0):
    return not (a[2] + margin < b[0] or b[2] + margin < a[0]
                or a[3] + margin < b[1] or b[3] + margin < a[1])


def _overlap_area(a, b):
    if not _rects_overlap(a, b):
        return 0.0
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def _shared_edge(room, target, tol=200, min_shared_len=400):
    rx0, ry0, rx1, ry1 = room
    tx0, ty0, tx1, ty1 = target
    edges = []
    x_overlap = min(rx1, tx1) - max(rx0, tx0)
    y_overlap = min(ry1, ty1) - max(ry0, ty0)
    if x_overlap >= min_shared_len:
        if abs(ry0 - ty1) <= tol:
            edges.append('S')
        if abs(ry1 - ty0) <= tol:
            edges.append('N')
    if y_overlap >= min_shared_len:
        if abs(rx0 - tx1) <= tol:
            edges.append('W')
        if abs(rx1 - tx0) <= tol:
            edges.append('E')
    return edges


def _room_rect(r):
    return (float(r['x']), float(r['y']),
            float(r['x']) + float(r['w']),
            float(r['y']) + float(r['h']))


def _is_bathroom(room):
    rt = (room.get('type') or '').lower()
    rn = (room.get('name') or '').lower()
    return (rt == 'bathroom'
            or 'bath' in rn or 'toilet' in rn or 'wc' in rn)


def _is_kitchen(room):
    rt = (room.get('type') or '').lower()
    rn = (room.get('name') or '').lower()
    return rt == 'kitchen' or 'kitchen' in rn


def validate_plan(layout, plot_data=None):
    violations = []

    if not layout:
        return [{'type': 'no_layout', 'severity': 'error',
                 'message': 'Layout is None or empty', 'room': None}]

    building = layout.get('building')
    corridor = layout.get('corridor')
    rooms = layout.get('rooms', [])

    if not building:
        return [{'type': 'no_building', 'severity': 'error',
                 'message': 'Layout has no building envelope', 'room': None}]

    b_rect = (float(building['x']), float(building['y']),
              float(building['x']) + float(building['w']),
              float(building['y']) + float(building['h']))

    if not corridor:
        violations.append({
            'type': 'no_corridor', 'severity': 'error',
            'message': 'Layout has no corridor rectangle', 'room': None,
        })
        c_rect = None
    else:
        c_rect = (float(corridor['x']), float(corridor['y']),
                  float(corridor['x']) + float(corridor['w']),
                  float(corridor['y']) + float(corridor['h']))
        if corridor['w'] < 1000 or corridor['h'] < 1000:
            violations.append({
                'type': 'small_corridor', 'severity': 'error',
                'message': (f"Corridor too narrow: "
                            f"{corridor['w']} x {corridor['h']} mm"),
                'room': None,
            })

    for r in rooms:
        r_rect = _room_rect(r)
        if not _rect_inside(r_rect, b_rect, margin=100):
            violations.append({
                'type': 'out_of_bounds', 'severity': 'error',
                'message': (f"Room '{r.get('name')}' is outside building bounds"),
                'room': r.get('name'),
            })

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = _room_rect(rooms[i])
            b = _room_rect(rooms[j])
            ov = _overlap_area(a, b)
            if ov > 200_000:
                violations.append({
                    'type': 'room_overlap', 'severity': 'error',
                    'message': (f"{rooms[i].get('name')} overlaps "
                                f"{rooms[j].get('name')} by {ov / 1e6:.2f} m2"),
                    'room': rooms[i].get('name'),
                })

    if c_rect:
        for r in rooms:
            r_rect = _room_rect(r)
            ov = _overlap_area(r_rect, c_rect)
            if ov > 100_000:
                violations.append({
                    'type': 'room_overlaps_corridor', 'severity': 'error',
                    'message': (f"Room '{r.get('name')}' overlaps the corridor"),
                    'room': r.get('name'),
                })

    if c_rect:
        for r in rooms:
            r_rect = _room_rect(r)
            shared = _shared_edge(r_rect, c_rect, tol=CORRIDOR_EDGE_TOL)
            if not shared:
                violations.append({
                    'type': 'no_corridor_access', 'severity': 'error',
                    'message': (f"Room '{r.get('name')}' does not touch the corridor"),
                    'room': r.get('name'),
                })

    if c_rect:
        for r in rooms:
            if not _is_bathroom(r):
                continue
            r_rect = _room_rect(r)
            if _shared_edge(r_rect, c_rect, tol=CORRIDOR_EDGE_TOL):
                continue
            neighbours = []
            for other in rooms:
                if other is r:
                    continue
                if _shared_edge(r_rect, _room_rect(other), tol=200):
                    neighbours.append(other)
            if neighbours and all(_is_kitchen(o) for o in neighbours):
                violations.append({
                    'type': 'bathroom_through_kitchen', 'severity': 'error',
                    'message': (f"Bathroom '{r.get('name')}' can only be reached "
                                f"through the kitchen"),
                    'room': r.get('name'),
                })

    def _touches_boundary(rect, tol=350):
        return (abs(rect[0] - b_rect[0]) <= tol
                or abs(rect[2] - b_rect[2]) <= tol
                or abs(rect[1] - b_rect[1]) <= tol
                or abs(rect[3] - b_rect[3]) <= tol)

    entry_found = False
    if c_rect and _touches_boundary(c_rect):
        entry_found = True
    if not entry_found:
        for r in rooms:
            if _touches_boundary(_room_rect(r)):
                entry_found = True
                break
    if not entry_found:
        violations.append({
            'type': 'no_entry', 'severity': 'error',
            'message': 'No room or corridor touches the building boundary',
            'room': None,
        })

    return violations
