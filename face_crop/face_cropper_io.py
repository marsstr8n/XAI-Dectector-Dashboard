# step 1

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import os

@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float = 1.0  # detector confidence (if available)

    def width(self) -> int:
        return self.x2 - self.x1

    def height(self) -> int:
        return self.y2 - self.y1

def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk and return an RGB numpy array (H, W, 3), uint8.
    """
    img = Image.open(path).convert("RGB")
    return np.array(img)

def save_image(arr: np.ndarray, out_path: str) -> None:
    """
    Save an RGB numpy array to disk (formats inferred by extension: .png/.jpg).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(arr).save(out_path)
