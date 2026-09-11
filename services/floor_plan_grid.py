"""
services/floor_plan_grid.py

Pre-computes a valid floor plan structure: building envelope, corridor,
and a set of non-overlapping cells arranged below and above the corridor.
The AI then assigns rooms to cells. Geometry is owned by Python.
"""


def build_grid(plot_data):
    """
    Returns:
      {
        'building':    {'x','y','w','h'},
        'corridor':    {'x','y','w','h'},
        'entry_wall':  'S' | 'N' | 'E' | 'W',
        'cells':       [ {id, x, y, w, h, side}, ... ],
      }
    'side' is 'lower' (below corridor) or 'upper' (above corridor).
    """
    pw_mm = int((plot_data.get('plot_width') or 12) * 1000)
    pl_mm = int((plot_data.get('plot_length') or 16) * 1000)

    sw = plot_data.get('street_width_m', 10)
    if sw >= 12:   front, rear, side = 3000, 2000, 1500
    elif sw >= 8:  front, rear, side = 2500, 1800, 1500
    else:          front, rear, side = 2000, 1800, 1200

    bx = side
    by = front
    bw = pw_mm - 2 * side
    bh = pl_mm - front - rear

    # Make sure we have enough room for a reasonable building
    bw = max(bw, 8000)
    bh = max(bh, 9000)

    corridor_h = 1300
    mid_y = by + bh / 2
    corridor_y = int(mid_y - corridor_h / 2)
    corridor = {'x': bx, 'y': corridor_y, 'w': bw, 'h': corridor_h}

    lower_y0 = by
    lower_y1 = corridor_y
    upper_y0 = corridor_y + corridor_h
    upper_y1 = by + bh

    def split_row(y0, y1, n, x0, w, prefix, side):
        cells = []
        cell_w = w / n
        for i in range(n):
            cells.append({
                'id':     f'{prefix}{i+1}',
                'x':      int(x0 + i * cell_w),
                'y':      int(y0),
                'w':      int(cell_w),
                'h':      int(y1 - y0),
                'side':   side,
            })
        return cells

    # Lower zone: 3 cells side by side
    lower = split_row(lower_y0, lower_y1, 3, bx, bw, 'L', 'lower')
    # Upper zone: 4 cells side by side
    upper = split_row(upper_y0, upper_y1, 4, bx, bw, 'U', 'upper')

    # Reserve the middle upper cell for the staircase — it will not
    # receive a room assignment. The DXF layer draws the stair steps
    # inside it, and no door is drawn on that cell's corridor wall.
    if len(upper) >= 3:
        upper[len(upper) // 2]['is_stair'] = True

    return {
        'building':   {'x': bx, 'y': by, 'w': bw, 'h': bh},
        'corridor':   corridor,
        'entry_wall': 'S',
        'cells':      lower + upper,
    }
