"""
services/plan_validator.py

Spatial and circulation validator for AI-generated floor plans.
Does NOT draw anything, does NOT modify the plan.
Returns a list of violation dicts.

Each violation: {
    'type': str,       # short machine key
    'severity': str,   # 'error' | 'warning'
    'message': str,    # human-readable
    'room': str | None # optional room name
}
"""


def _rect_inside(inner, outer, margin=0):
    """(x0,y0,x1,y1) fully inside outer with `margin` allowance."""
    return (inner[0] >= outer[0] - margin
            and inner[1] >= outer[1] - margin
            and inner[2] <= outer[2] + margin
            and inner[3] <= outer[3] + margin)


def _rects_overlap(a, b, margin=0):
    """True if rectangles a and b overlap by more than 0 area."""
    return not (a[2] + margin < b[0] or b[2] + margin < a[0]
                or a[3] + margin < b[1] or b[3] + margin < a[1])


def _overlap_area(a, b):
    if not _rects_overlap(a, b):
        return 0.0
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, dx) * max(0.0, dy)


def _shared_edge(room, target, tol=200, min_shared_len=400):
    """
    Return a list of direction strings where `room` shares a wall with `target`.
    Directions are the room's own wall: 'N','S','E','W'.

    Rule:
      - A shared HORIZONTAL edge requires the x-ranges to overlap.
      - A shared VERTICAL   edge requires the y-ranges to overlap.
      - The coordinate lines (ry0/ry1 vs ty0/ty1 or rx0/rx1 vs tx0/tx1)
        must be within `tol` mm of each other.
      - The overlap along the shared line must be at least `min_shared_len` mm.
    """
    rx0, ry0, rx1, ry1 = room
    tx0, ty0, tx1, ty1 = target
    edges = []

    # How much the x-ranges overlap (perpendicular to a horizontal shared edge)
    x_overlap = min(rx1, tx1) - max(rx0, tx0)
    # How much the y-ranges overlap (perpendicular to a vertical shared edge)
    y_overlap = min(ry1, ty1) - max(ry0, ty0)

    # Horizontal edges (room above/below target)
    if x_overlap >= min_shared_len:
        if abs(ry0 - ty1) <= tol:       # room's S wall meets target's N wall
            edges.append('S')
        if abs(ry1 - ty0) <= tol:       # room's N wall meets target's S wall
            edges.append('N')

    # Vertical edges (room beside target)
    if y_overlap >= min_shared_len:
        if abs(rx0 - tx1) <= tol:       # room's W wall meets target's E wall
            edges.append('W')
        if abs(rx1 - tx0) <= tol:       # room's E wall meets target's W wall
            edges.append('E')

    return edges


def _room_rect(r):
    return (float(r['x']), float(r['y']),
            float(r['x']) + float(r['w']),
            float(r['y']) + float(r['h']))


def validate_plan(layout, plot_data=None):
    """
    Run all checks. Return list of violations (empty = clean).
    """
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

    # --- 1. Corridor presence ---
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
                            f"{corridor['w']} x {corridor['h']} mm "
                            f"(min 1000 mm each side)"),
                'room': None,
            })

    # --- 2. Rooms inside building ---
    for r in rooms:
        r_rect = _room_rect(r)
        if not _rect_inside(r_rect, b_rect, margin=100):
            violations.append({
                'type': 'out_of_bounds', 'severity': 'error',
                'message': (f"Room '{r.get('name')}' "
                            f"({r_rect[0]:.0f},{r_rect[1]:.0f})-"
                            f"({r_rect[2]:.0f},{r_rect[3]:.0f}) is outside "
                            f"building bounds "
                            f"({b_rect[0]:.0f},{b_rect[1]:.0f})-"
                            f"({b_rect[2]:.0f},{b_rect[3]:.0f})"),
                'room': r.get('name'),
            })

    # --- 3. Room-room overlaps ---
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a = _room_rect(rooms[i])
            b = _room_rect(rooms[j])
            ov = _overlap_area(a, b)
            if ov > 200_000:  # 0.2 m2
                violations.append({
                    'type': 'room_overlap', 'severity': 'error',
                    'message': (f"{rooms[i].get('name')} overlaps "
                                f"{rooms[j].get('name')} by "
                                f"{ov / 1e6:.2f} m2"),
                    'room': rooms[i].get('name'),
                })

    # --- 4. Room-corridor overlaps ---
    if c_rect:
        for r in rooms:
            r_rect = _room_rect(r)
            ov = _overlap_area(r_rect, c_rect)
            if ov > 100_000:
                violations.append({
                    'type': 'room_overlaps_corridor', 'severity': 'error',
                    'message': (f"Room '{r.get('name')}' overlaps the "
                                f"corridor by {ov / 1e6:.2f} m2"),
                    'room': r.get('name'),
                })

    # --- 5. Every room must touch the corridor ---
    if c_rect:
        for r in rooms:
            r_rect = _room_rect(r)
            shared = _shared_edge(r_rect, c_rect, tol=200)
            if not shared:
                violations.append({
                    'type': 'no_corridor_access', 'severity': 'error',
                    'message': (f"Room '{r.get('name')}' does not touch "
                                f"the corridor — no way in without passing "
                                f"through another room"),
                    'room': r.get('name'),
                })
            else:
                # Check that door_wall (if present) matches the corridor-facing edge
                dw = r.get('door_wall')
                if dw and dw not in shared:
                    violations.append({
                        'type': 'door_wall_mismatch', 'severity': 'warning',
                        'message': (f"Room '{r.get('name')}' has "
                                    f"door_wall='{dw}' but its corridor-facing "
                                    f"edge(s) are {shared}"),
                        'room': r.get('name'),
                    })

    # --- 6. Entry from outside ---
    # At least one room or the corridor must touch the building boundary.
    def _touches_boundary(rect, tol=200):
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
            'message': 'No room or corridor touches the building boundary — '
                       'there is no way in from outside',
            'room': None,
        })

    return violations
