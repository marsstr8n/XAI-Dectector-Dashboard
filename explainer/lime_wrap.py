import numpy as np
from typing import Callable, Optional

class LimePredictor:
    """
    Wrap predict_proba(images_rgb_01: list[np.ndarray]) -> (N,C)
    so LIME can call classifier_fn(batch_uint8: (N,H,W,3) uint8) -> (N,C).
    """
    def __init__(self, predict_proba_fn: Callable[[list[np.ndarray]], np.ndarray]):
        self.predict_proba_fn = predict_proba_fn

    def __call__(self, batch_uint8: np.ndarray) -> np.ndarray:
        # LIME sends float or uint8; normalise to [0,1]
        if batch_uint8.dtype != np.uint8:
            batch_uint8 = np.clip(batch_uint8, 0, 255).astype(np.uint8)
        imgs_01 = [(x.astype(np.float32) / 255.0) for x in batch_uint8]
        probs = self.predict_proba_fn(imgs_01)   # (N,C)
        return probs
