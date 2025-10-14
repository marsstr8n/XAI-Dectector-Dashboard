# Step 5

from typing import Optional, Literal
import os
import numpy as np

from .face_cropper_io import load_image, save_image, Box
from .face_cropper_detect import detect_faces_mediapipe
from .face_cropper_box import choose_face, expand_box
from .face_cropper_crop import crop_to_box, ensure_min_size

def crop_face_from_image(
    input_path: str,
    output_path: str,
    *,
    min_score: float = 0.5,
    strategy: Literal["largest", "highest_score"] = "largest",
    margin: float = 0.2,
    min_side: int = 160,
    model_selection: int = 1
) -> Optional[str]:
    """
    Detects faces in `input_path`, picks one (by `strategy`), expands by `margin`,
    crops, ensures minimal size, and writes to `output_path`.
    Returns output_path if successful, else None (no face found).
    """
    img = load_image(input_path)
    boxes = detect_faces_mediapipe(img, min_score=min_score, model_selection=model_selection)
    if not boxes:
        return None

    face = choose_face(boxes, strategy=strategy)
    face = expand_box(face, img.shape, margin=margin)
    crop = crop_to_box(img, face)
    crop = ensure_min_size(crop, min_side=min_side)

    save_image(crop, output_path)
    return output_path
