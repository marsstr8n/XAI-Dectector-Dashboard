# Step 4

import numpy as np
from typing import Optional, Tuple
from .face_cropper_io import Box

def crop_to_box(image_rgb: np.ndarray, box: Box) -> np.ndarray:
    """Return the face crop (RGB)."""
    return image_rgb[box.y1:box.y2, box.x1:box.x2, :]

def ensure_min_size(crop: np.ndarray, min_side: int = 128) -> np.ndarray:
    """
    If the crop is too small, upscale (nearest/bilinear) to at least min_side on the shorter edge.
    """
    h, w = crop.shape[:2]
    if min(h, w) >= min_side:
        return crop

    scale = max(min_side / h, min_side / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # PIL resize is simple and decent quality
    from PIL import Image
    return np.array(Image.fromarray(crop).resize((new_w, new_h), resample=Image.BILINEAR))
