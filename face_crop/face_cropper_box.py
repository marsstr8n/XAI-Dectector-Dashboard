# step 3

from typing import List, Literal, Optional, Tuple
import numpy as np
from .face_cropper_io import Box

def choose_face(
    boxes: List[Box],
    strategy: Literal["largest", "highest_score"] = "largest"
) -> Optional[Box]:
    """
    Choose one face when multiple are present.
    - 'largest': pick the box with greatest area (good default)
    - 'highest_score': pick by detector confidence
    """
    if not boxes:
        return None
    if strategy == "highest_score":
        return max(boxes, key=lambda b: b.score)
    # default: largest area
    return max(boxes, key=lambda b: b.width() * b.height())

def expand_box(
    box: Box,
    image_shape: Tuple[int, int, int],
    margin: float = 0.2
) -> Box:
    """
    Expand a box by a relative margin (e.g., 0.2 = expand 20% each side)
    and clip to image boundaries. Gives more context for GradCAM and LIME
    """
    h, w = image_shape[:2]
    bw, bh = box.width(), box.height()

    # grow equally on all sides
    dx = int(bw * margin)
    dy = int(bh * margin)

    x1 = max(0, box.x1 - dx)
    y1 = max(0, box.y1 - dy)
    x2 = min(w, box.x2 + dx)
    y2 = min(h, box.y2 + dy)

    # Ensure valid box after clipping
    if x2 <= x1: x2 = min(w, x1 + 1)
    if y2 <= y1: y2 = min(h, y1 + 1)

    return Box(x1, y1, x2, y2, score=box.score)
