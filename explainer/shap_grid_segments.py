import numpy as np

def build_grid_segments(h: int, w: int, cell: int = 16) -> np.ndarray:
    """
    Returns an (h, w) int32 array labeling a fixed grid of square cells.
    Cells are ~cell x cell (last row/col may be smaller if h/w % cell != 0).
    Labels are 0..M-1 in row-major order.
    """
    rows = (h + cell - 1) // cell
    cols = (w + cell - 1) // cell
    seg = np.zeros((h, w), dtype=np.int32)
    lab = 0
    for r in range(rows):
        y0 = r * cell
        y1 = min((r + 1) * cell, h)
        for c in range(cols):
            x0 = c * cell
            x1 = min((c + 1) * cell, w)
            seg[y0:y1, x0:x1] = lab
            lab += 1
    return seg
