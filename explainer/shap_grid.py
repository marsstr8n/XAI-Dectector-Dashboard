from __future__ import annotations
import numpy as np, cv2, torch
import torch.nn.functional as F
from typing import Optional, Literal
import shap

from .config import IMG_SIZE
from .tensor_io import to_tensor_rgb01
from .shap_grid_segments import build_grid_segments

# replace _overlay_bwr with a red/blue-only overlay
def _overlay_redblue(
    img_rgb_u8: np.ndarray,
    heat: np.ndarray,            # [-1, 1]
    alpha_scale: float = 0.55,   # overall opacity multiplier
) -> np.ndarray:
    """
    Map heat in [-1,1] to red (pos) and blue (neg) only.
    Opacity = |heat| * alpha_scale; 0 is transparent.
    """
    base = img_rgb_u8.astype(np.float32) / 255.0
    h = np.clip(heat, -1.0, 1.0).astype(np.float32)
    mag = np.abs(h)

    # colors (RGB in [0,1])
    red  = np.array([1.0, 0.20, 0.20], dtype=np.float32)  # tweak if you like
    blue = np.array([0.20, 0.45, 1.0], dtype=np.float32)

    # build color image per pixel
    color = np.zeros((*h.shape, 3), dtype=np.float32)
    pos = (h > 0)
    neg = ~pos
    color[pos] = red
    color[neg] = blue

    alpha = (mag * alpha_scale)[..., None]               # (H,W,1)
    out = np.clip(base * (1 - alpha) + color * alpha, 0, 1)
    return (out * 255).astype(np.uint8)

def shap_grid_explain(
    model: torch.nn.Module,
    device: str,
    img_rgb_uint8: np.ndarray,
    *,
    target_label: int = 1,                  # 0=real, 1=fake
    cell: int = 16,                         # grid cell size on model input (dot size)
    background: Literal["blur", "gray"] = "blur",
    output: Literal["logit", "prob"] = "logit",
    max_evals: int = 3000,                  # total SHAP samples budget
    batch_chunk: int = 32,                  # compose/predict this many images at a time
    heat_blur_sigma: float = 0.0,           # 0 for crisp; >0 softens
    robust_pct: float = 99.0,               # robust scaling pct for [-1,1]
    # “dots-only” rendering (optional)
    dots_only: bool = False,
    tau: float = 0.25,                      # keep |heat| >= tau when dots_only=True
    dots_on_white: bool = True,             # else overlay on original image
):
    """
    Returns:
      heat_up:  (H0,W0) float32 in [-1,1]
      overlay:  (H0,W0,3) uint8  (dots-on-white or colored overlay)
    """
    model.eval().to(device)

    # Resize to model input size
    H0, W0 = img_rgb_uint8.shape[:2]
    Hm = Wm = IMG_SIZE
    img_model = cv2.resize(img_rgb_uint8, (Wm, Hm), interpolation=cv2.INTER_AREA)

    # Grid segments (no SHAP API tricks; we’ll do KernelExplainer over segments)
    segments = build_grid_segments(Hm, Wm, cell=cell)
    seg_ids = np.unique(segments)
    M = int(seg_ids.size)

    # Build a single background image
    if background == "blur":
        bkg = cv2.GaussianBlur(img_model, (17, 17), 8)
    else:
        gray = int(np.round(np.mean(img_model)))
        bkg = np.full_like(img_model, gray, dtype=np.uint8)

    # Compose images from binary segment masks
    def compose_from_mask(Z_bin: np.ndarray) -> np.ndarray:
        """Z_bin: (N, M) -> (N, Hm, Wm, 3) uint8"""
        N = Z_bin.shape[0]
        out = np.empty((N, Hm, Wm, 3), dtype=np.uint8)
        for i in range(N):
            keep = Z_bin[i] > 0.5
            img_i = bkg.copy()
            for idx, sid in enumerate(seg_ids):
                if keep[idx]:
                    img_i[segments == sid] = img_model[segments == sid]
            out[i] = img_i
        return out

    # Model scoring function over Z (superpixel space)
    def f_superpixels(Z_bin: np.ndarray, chunk: int = batch_chunk) -> np.ndarray:
        outs = []
        N = Z_bin.shape[0]
        for i in range(0, N, chunk):
            Zi = Z_bin[i:i + chunk]
            batch_hw3 = compose_from_mask(Zi)
            ts = torch.cat([to_tensor_rgb01(x, device) for x in batch_hw3], dim=0)
            with torch.no_grad():
                logits = model(ts)
                if output == "prob":
                    vals = F.softmax(logits, dim=1)[:, target_label].detach().cpu().numpy()
                else:
                    vals = logits[:, target_label].detach().cpu().numpy()
            outs.append(vals[:, None])   # (chunk,1)
        return np.vstack(outs)            # (N,1)

    # KernelExplainer over superpixels: background = all OFF, explain all ON
    z0 = np.zeros((1, M), dtype=float)  # background point
    z1 = np.ones((1, M), dtype=float)   # point to explain
    ke = shap.KernelExplainer(f_superpixels, z0)

    # Cap nsamples so runtime stays sane with many cells
    ns = int(min(max_evals, 2 * M + 2048))
    shap_vals = ke.shap_values(z1, nsamples=ns)

    # Robustly extract a 1-D vector s of length M
    v = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 2:
        s = v[0] if v.shape[0] == 1 else v[:, 0]
    elif v.ndim == 1:
        s = v
    else:
        s = v.reshape(-1)
    if s.shape[0] != M:
        s = np.resize(s, M).astype(np.float32)

    # Paint SHAP per-cell back to image grid
    sv_map = np.zeros((Hm, Wm), dtype=np.float32)
    for idx, sid in enumerate(seg_ids):
        sv_map[segments == sid] = s[idx]

    # Normalize to [-1, 1] robustly
    mx = float(np.max(np.abs(sv_map))) + 1e-12
    heat = sv_map / mx
    lo = np.percentile(heat, 100.0 - robust_pct)
    hi = np.percentile(heat, robust_pct)
    if hi > lo + 1e-12:
        heat = (np.clip(heat, lo, hi) - lo) / (hi - lo) * 2.0 - 1.0

    if heat_blur_sigma > 0:
        k = int(max(3, round(heat_blur_sigma * 4)) // 2 * 2 + 1)
        heat = cv2.GaussianBlur(heat, (k, k), heat_blur_sigma)

    # Back to original size
    heat_up = cv2.resize(heat, (W0, H0), interpolation=cv2.INTER_NEAREST)

    # Build “dots” overlay
    if dots_only:
        # only keep strong |heat| >= tau
        h = np.clip(heat_up, -1, 1).astype(np.float32)
        keep = (np.abs(h) >= float(tau))
        # colorize like above
        red  = np.array([1.0, 0.20, 0.20], dtype=np.float32)
        blue = np.array([0.20, 0.45, 1.0], dtype=np.float32)
        color = np.zeros((H0, W0, 3), dtype=np.float32)
        color[h > 0] = red
        color[h <= 0] = blue

        # opacity from magnitude; optionally scale it a bit higher for dots
        alpha = (np.abs(h) * 0.85)[..., None]  # 0..0.85
        alpha *= keep[..., None].astype(np.float32)

        if dots_on_white:
            base = np.ones_like(color, dtype=np.float32)
        else:
            base = img_rgb_uint8.astype(np.float32) / 255.0

        out = np.clip(base * (1 - alpha) + color * alpha, 0, 1)
        overlay = (out * 255).astype(np.uint8)
    else:
        overlay = _overlay_redblue(img_rgb_uint8, heat_up, alpha_scale=0.55)


    return heat_up, overlay
