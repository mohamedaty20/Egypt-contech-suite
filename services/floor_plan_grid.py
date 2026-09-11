"""
services/floor_plan_grid.py

Pre-computes a valid floor plan structure with SEEDED VARIATION.
"""
import random

MIN_CELL_W_MM = 1400


def build_grid(plot_data):
    seed = int(plot_data.get('variation_seed') or 0)
    rng = random.Random(seed)

    pw_mm = int((plot_data.get('plot_width') or 12) * 1000)
    pl_mm = int((plot_data.get('plot_length') or 16) * 1000)
    nb  = int(plot_data.get('num_bedrooms', 3) or 3)
    nba = int(plot_data.get('num_bathrooms', 2) or 2)

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

    # --- VARIATION 1: corridor height ---
    corridor_h = rng.choice([1200, 1300, 1400, 1500])

    # --- VARIATION 2: corridor vertical bias ---
    mid_bias = rng.choice([0.0, 0.0, 0.0, -0.03, 0.03])
    mid_y = iy0 + ih * (0.5 + mid_bias)
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

    # --- VARIATION 3: cell counts ---
    min_cells = nb + nba + 2
    lower_n = rng.choice([2, 3, 3, 4])
    upper_n = rng.choice([3, 4, 4, 5])
    guard = 0
    while lower_n + upper_n < min_cells and guard < 20:
        if rng.random() > 0.5:
            lower_n += 1
        else:
            upper_n += 1
        guard += 1

    # --- VARIATION 4: mirror left-right ---
    mirror = rng.random() > 0.5

    def _build_row(y0, y1, n, side_tag):
        n = _pick_n(n, iw)
        base_w = iw // n
        remainder = iw - base_w * n
        widths = [base_w + (1 if i < remainder else 0) for i in range(n)]
        if mirror:
            widths = list(reversed(widths))
        door_wall = 'top' if side_tag == 'lower' else 'bottom'
        x_cursor = ix0
        cells = []
        for i in range(n):
            cw = widths[i]
            cells.append({
                'id':     f"{'L' if side_tag == 'lower' else 'U'}{i+1}",
                'x':      int(x_cursor),
                'y':      int(y0),
                'w':      int(cw),
                'h':      int(y1 - y0),
                'side':   side_tag,
                'faces_corridor': True,
                'door_wall':      door_wall,
            })
            x_cursor += cw
        return cells

    lower = _build_row(lower_y0, lower_y1, lower_n, 'lower')
    upper = _build_row(upper_y0, upper_y1, upper_n, 'upper')

    # --- VARIATION 5: stair position ---
    if len(upper) >= 3:
        mid_i = len(upper) // 2
        stair_idx = rng.choice([max(1, mid_i - 1), mid_i,
                                min(len(upper) - 1, mid_i + 1)])
        upper[stair_idx]['is_stair'] = True
        upper[stair_idx]['faces_corridor'] = False
        upper[stair_idx]['door_wall'] = None

    return {
        'building':   {'x': bx, 'y': by, 'w': bw, 'h': bh},
        'corridor':   corridor,
        'entry_wall': 'S',
        'cells':      lower + upper,
    }
