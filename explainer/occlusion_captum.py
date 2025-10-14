from __future__ import annotations
import numpy as np, cv2, torch
import torch.nn.functional as F
from typing import Literal, Optional, Tuple
from captum.attr import Occlusion

from .config import IMG_SIZE
from .tensor_io import to_tensor_rgb01

def _overlay_redblue(img_rgb_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Pure red/blue overlay, heat in [-1,1].
    Red = supports target (increase target score), Blue = opposes (decrease target score).
    """
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

def occlusion_explain_captum(
    model: torch.nn.Module,
    device: str,
    img_rgb_uint8: np.ndarray,
    *,
    target_label: int = 1,                         # 0=real, 1=fake
    patch: int = 32,                               # occluder size @ model input
    stride: int = 16,                              # slide step
    baseline: Literal["gray","blur","mean"] = "gray",
    normalize: Literal["robust","maxabs"] = "robust",
    robust_pct: float = 99.0,
    perturbations_per_eval: int = 16,              # Captum batching
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      heat_up  : (H0,W0) float32 in [-1,1]
      overlay  : (H0,W0,3) uint8 red/blue overlay
    Notes:
      - Uses Captum.Occlusion with channel-wise sliding window & stride.
      - We sum attributions over channels to get a scalar per pixel.
    """
    model.eval().to(device)

    # 1) Resize to model input size and make input tensor
    H0, W0 = img_rgb_uint8.shape[:2]
    Hm = Wm = IMG_SIZE
    imr = cv2.resize(img_rgb_uint8, (Wm, Hm), interpolation=cv2.INTER_AREA)
    x = to_tensor_rgb01(imr, device)             # (1,3,Hm,Wm), float32 0..1 (your pipeline)
    x.requires_grad_(False)

    # 2) Build baseline tensor matching input
    if baseline == "gray":
        val = float(np.mean(imr)) / 255.0
        base = torch.full_like(x, val)
    elif baseline == "blur":
        bl = cv2.GaussianBlur(imr, (21, 21), 8).astype(np.float32) / 255.0
        base = torch.from_numpy(bl.transpose(2, 0, 1)).unsqueeze(0).to(device)
    else:  # "mean" per-channel
        mean_c = (imr.astype(np.float32)/255.0).reshape(-1,3).mean(axis=0)
        base = torch.from_numpy(mean_c[None, :, None, None]).expand_as(x).contiguous().to(device)

    # 3) Captum occlusion
    occ = Occlusion(model)
    sliding_window_shapes = (x.shape[1], patch, patch)   # (channels, h, w)
    strides               = (x.shape[1], stride, stride)

    attrs = occ.attribute(
        inputs=x,
        strides=strides,
        sliding_window_shapes=sliding_window_shapes,
        baselines=base,
        target=target_label,                 # class index to explain
        perturbations_per_eval=perturbations_per_eval,
    )  # -> (1,3,Hm,Wm)

    # 4) Collapse channels to scalar per-pixel attribution
    a = attrs[0].detach().cpu().numpy()      # (3,Hm,Wm)
    heat = a.sum(axis=0)                     # (Hm,Wm), signed

    # 5) Normalize to [-1,1]
    if normalize == "robust":
        lo = np.percentile(heat, 100.0 - robust_pct)
        hi = np.percentile(heat, robust_pct)
        if hi > lo:
            h = (np.clip(heat, lo, hi) - lo) / (hi - lo + 1e-12) * 2.0 - 1.0
        else:
            h = heat / (np.max(np.abs(heat)) + 1e-12)
    else:
        h = heat / (np.max(np.abs(heat)) + 1e-12)

    # 6) Upsample to original size and color it
    heat_up = cv2.resize(h, (W0, H0), interpolation=cv2.INTER_CUBIC)
    overlay = _overlay_redblue(img_rgb_uint8, heat_up, alpha=0.55)
    return heat_up, overlay
