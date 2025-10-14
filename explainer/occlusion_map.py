from __future__ import annotations
import numpy as np, cv2, torch
import torch.nn.functional as F
from typing import Optional, Literal
from .config import IMG_SIZE
from .tensor_io import to_tensor_rgb01

def _overlay_redblue(img_rgb_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Pure red/blue overlay, heat in [-1,1]. Red=↑target (supports), Blue=↓target (opposes)."""
    base = img_rgb_u8.astype(np.float32) / 255.0
    h = np.clip(heat, -1.0, 1.0).astype(np.float32)
    mag = np.abs(h)
    red  = np.array([1.0, 0.20, 0.20], dtype=np.float32)
    blue = np.array([0.20, 0.45, 1.0], dtype=np.float32)
    color = np.zeros((*h.shape, 3), dtype=np.float32)
    color[h >= 0] = red
    color[h <  0] = blue
    out = np.clip(base * (1 - alpha * mag[..., None]) + color * (alpha * mag[..., None]), 0, 1)
    return (out * 255).astype(np.uint8)

def occlusion_explain(
    model: torch.nn.Module,
    device: str,
    img_rgb_uint8: np.ndarray,
    *,
    target_label: int = 1,                     # 0=real, 1=fake
    patch: int = 32,                           # occluder size @ model input resolution
    stride: int = 16,                          # slide step
    baseline: Literal["gray","blur","mean"] = "gray",
    aggregate: Literal["logit","prob"] = "logit",
    normalize: Literal["robust","maxabs"] = "robust",
    robust_pct: float = 99.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      heat_up  : (H0,W0) float32 in [-1,1]  (positive = region supports target)
      overlay  : (H0,W0,3) uint8 red/blue overlay
    """
    model.eval().to(device)

    # Resize to model size
    H0, W0 = img_rgb_uint8.shape[:2]
    Hm = Wm = IMG_SIZE
    imr = cv2.resize(img_rgb_uint8, (Wm, Hm), interpolation=cv2.INTER_AREA)

    # Baseline content
    if baseline == "gray":
        base_val = int(np.round(np.mean(imr)))
        base_img = np.full_like(imr, base_val, dtype=np.uint8)
    elif baseline == "blur":
        base_img = cv2.GaussianBlur(imr, (21, 21), 8)
    else:  # mean per-channel
        m = imr.reshape(-1, 3).mean(axis=0, keepdims=True)
        base_img = np.tile(m.astype(np.uint8), (Hm * Wm, 1)).reshape(Hm, Wm, 3)

    # Reference score on original
    with torch.no_grad():
        x0_t = to_tensor_rgb01(imr, device)
        logits0 = model(x0_t)
        ref = (F.softmax(logits0, dim=1)[0, target_label].item()
               if aggregate == "prob" else logits0[0, target_label].item())

    # Scan
    heat = np.zeros((Hm, Wm), dtype=np.float32)
    counts = np.zeros((Hm, Wm), dtype=np.float32)

    for y in range(0, Hm, stride):
        for x in range(0, Wm, stride):
            y0, y1 = y, min(y + patch, Hm)
            x0, x1 = x, min(x + patch, Wm)


            occl = imr.copy()
            occl[y0:y1, x0:x1] = base_img[y0:y1, x0:x1]

            with torch.no_grad():
                t = to_tensor_rgb01(occl, device)
                logits = model(t)
                val = (F.softmax(logits, dim=1)[0, target_label].item()
                       if aggregate == "prob" else logits[0, target_label].item())

            delta = ref - val   # drop when occluded (positive = region supports target)
            heat[y0:y1, x0:x1] += delta
            counts[y0:y1, x0:x1] += 1.0

    heat /= np.maximum(counts, 1e-6)

    # Normalize to [-1,1]
    if normalize == "robust":
        lo = np.percentile(heat, 100.0 - robust_pct)
        hi = np.percentile(heat, robust_pct)
        if hi > lo:
            h = (np.clip(heat, lo, hi) - lo) / (hi - lo + 1e-12) * 2.0 - 1.0
        else:
            h = heat / (np.max(np.abs(heat)) + 1e-12)
    else:
        h = heat / (np.max(np.abs(heat)) + 1e-12)

    heat_up = cv2.resize(h, (W0, H0), interpolation=cv2.INTER_CUBIC)
    overlay = _overlay_redblue(img_rgb_uint8, heat_up, alpha=0.55)
    return heat_up, overlay
