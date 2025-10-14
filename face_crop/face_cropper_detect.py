# step 2

import mediapipe as mp
import numpy as np
from typing import List
from .face_cropper_io import Box  

_mp_fd = mp.solutions.face_detection

def detect_faces_mediapipe(
    image_rgb: np.ndarray,
    min_score: float = 0.5,
    model_selection: int = 1
) -> List[Box]:
    """
    Detect faces using MediaPipe FaceDetection.
    Returns a list of Box(x1,y1,x2,y2,score) in pixel coordinates.
    - model_selection=0: short-range (fast, faces within ~2m)
    - model_selection=1: full-range (works better for varied sizes)
    """
    h, w = image_rgb.shape[:2]
    results_boxes: List[Box] = []

    with _mp_fd.FaceDetection(model_selection=model_selection, min_detection_confidence=min_score) as fd:
        results = fd.process(image_rgb)

    if not results.detections:
        return results_boxes

    for det in results.detections:
        score = det.score[0] if det.score else 1.0
        # MediaPipe gives normalised bounding boxes (relative to W,H)
        rel = det.location_data.relative_bounding_box
        x1 = max(0, int(rel.xmin * w))
        y1 = max(0, int(rel.ymin * h))
        x2 = min(w, int((rel.xmin + rel.width) * w))
        y2 = min(h, int((rel.ymin + rel.height) * h))

        # Defensive check in case something is degenerate
        if x2 > x1 and y2 > y1:
            results_boxes.append(Box(x1, y1, x2, y2, score))

    return results_boxes
