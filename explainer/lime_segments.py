import numpy as np
from skimage.segmentation import slic

def slic_segments(
    img_rgb_uint8: np.ndarray,
    n_segments: int = 80,           # lower for bigger superpixels (e.g 60–90 at 351x351)
    compactness: float = 18.0,      # higher gives more regular (less texture-driven) shapes
    sigma: float = 1.5,             # pre-smoothing helps merge noisy texture
    start_label: int = 0,
    enforce_connectivity: bool = True,
) -> np.ndarray:
    """
    Returns labels: (H, W) int32.
    """
    seg = slic(
        img_rgb_uint8,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=int(start_label),
        enforce_connectivity=bool(enforce_connectivity),
    )
    return seg

def build_grid_segments(H: int, W: int, cell: int = 16):
    """
    Regular grid superpixels. Returns (H, W) int32 labels.
    Each cell is ~cell x cell at the given HxW.
    """
    import numpy as np
    h = int(cell)
    w = int(cell)
    yy = (np.arange(H) // h).astype(np.int32)
    xx = (np.arange(W) // w).astype(np.int32)
    grid = yy[:, None] * ((W + w - 1) // w) + xx[None, :]
    return grid.astype(np.int32)