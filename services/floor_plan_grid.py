"""
services/floor_plan_grid.py

Pre-computes a valid floor plan structure: building envelope, corridor,
and a set of non-overlapping cells arranged below and above the corridor.
The AI then assigns rooms to cells. Geometry is owned by Python.
"""

MIN_CELL_W_MM = 1400


def build_grid(plot_data):
    """
    Envelope = OUTER face of exterior walls.
    Cells live inside the INNER face so rooms don't overlap the walls.
    """
    pw_mm = int((plot_data.get('plot_width') or 12) * 1000)
    pl_mm = int((plot_data.get('plot_length') or 16) * 1000)

    sw = plot_data.get('street_width_m', 10)
    if sw >= 12:   front, rear, side = 3000, 2000, 1500
    elif sw >= 8:  front, rear, side = 2500, 1800, 1500
    else:          front, rear, side = 2000, 1800, 1200

    wall_mm = int(plot_data.get('wall_thickness_mm', 250))

    bx = side
    by = front
    bw = max(pw_mm - 2 * side, 8000)
    bh = max(pl_mm - front - rear, 9000)

    ix0 = bx + wall_mm
    iy0 = by + wall_mm
    ix1 = bx + bw - wall_mm
    iy1 = by + bh - wall_mm
    iw = ix1 - ix0
    ih = iy1 - iy0

    corridor_h = 1300
    mid_y = iy0 + ih / 2
    corridor_y = int(mid_y - corridor_h / 2)

    corridor_x = int(ix0)
    corridor_w = int(max(0, iw))
    if corridor_y < iy0:
        corridor_y = int(iy0)
    if corridor_y + corridor_h > iy1:
        corridor_y = int(iy1 - corridor_h)
    if corridor_h > ih:
        corridor_h = int(ih)

    corridor = {'x': corridor_x, 'y': corridor_y,
                'w': corridor_w, 'h': corridor_h}

    lower_y0 = iy0
    lower_y1 = corridor_y
    upper_y0 = corridor_y + corridor_h
    upper_y1 = iy1

    def _pick_n(target_n, avail_w):
        if avail_w <= 0:
            return 1
        n = max(1, target_n)
        while n > 1 and (avail_w / n) < MIN_CELL_W_MM:
            n -= 1
        return n

    def split_row(y0, y1, target_n, x0, w, prefix, side):
        n = _pick_n(target_n, w)
        cells = []
        base_w = w // n
        remainder = w - base_w * n
        x_cursor = x0
        door_wall = 'top' if side == 'lower' else 'bottom'
        for i in range(n):
            cw = base_w + (1 if i < remainder else 0)
            cells.append({
                'id':     f'{prefix}{i+1}',
                'x':      int(x_cursor),
                'y':      int(y0),
                'w':      int(cw),
                'h':      int(y1 - y0),
                'side':   side,
                'faces_corridor': True,
                'door_wall':      door_wall,
            })
            x_cursor += cw
        return cells

    lower = split_row(lower_y0, lower_y1, 3, ix0, iw, 'L', 'lower')
    upper = split_row(upper_y0, upper_y1, 4, ix0, iw, 'U', 'upper')

    if len(upper) >= 3:
        stair_idx = len(upper) // 2
        upper[stair_idx]['is_stair'] = True
        upper[stair_idx]['faces_corridor'] = False
        upper[stair_idx]['door_wall'] = None

    return {
        'building':   {'x': bx, 'y': by, 'w': bw, 'h': bh},
        'corridor':   corridor,
        'entry_wall': 'S',
        'cells':      lower + upper,
    }
