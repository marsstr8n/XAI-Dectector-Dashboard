from __future__ import annotations
import numpy as np, cv2, torch
import torch.nn.functional as F
from typing import Literal, Tuple

from SALib.sample import saltelli
from SALib.analyze import sobol

from .config import IMG_SIZE
from .tensor_io import to_tensor_rgb01
from .lime_segments import build_grid_segments  # grid superpixels

def _overlay_redblue(img_rgb_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Pure red/blue overlay for a signed heatmap in [-1,1].
    Red = supports the target class; Blue = opposes.
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

def sobol_grid_explain(
    model: torch.nn.Module,
    device: str,
    img_rgb_uint8: np.ndarray,
    *,
    target_label: int = 1, # 0=real, 1=fake
    cell: int = 28, # grid cell size @ model input (bigger => fewer cells)
    N: int = 32, # Saltelli base samples (runtime knob)
    calc_second_order: bool = False, # usually keep False for speed
    baseline: Literal["gray","blur","mean"] = "gray",
    aggregate: Literal["logit","prob"] = "logit",
    normalize: Literal["robust","maxabs"] = "robust",
    robust_pct: float = 97.0,
    batch: int = 32,                          # model batch for evaluation
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
      heat_up  : (H0,W0) float32 in [-1,1]  (signed ST; red supports target, blue opposes)
      overlay  : (H0,W0,3) uint8 red/blue overlay
      stats    : dict with 'S1', 'ST', 'corr' (per grid cell)

    """
    model.eval().to(device)

    # --- 1) Resize to model size ---
    H0, W0 = img_rgb_uint8.shape[:2]
    Hm = Wm = IMG_SIZE
    imr = cv2.resize(img_rgb_uint8, (Wm, Hm), interpolation=cv2.INTER_AREA)

    # --- 2) Baseline image (uint8) ---
    if baseline == "gray":
        val = int(np.round(np.mean(imr)))
        base_img = np.full_like(imr, val, dtype=np.uint8)
    elif baseline == "blur":
        base_img = cv2.GaussianBlur(imr, (21, 21), 8)
    else:  # mean per-channel
        m = imr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
        base_img = np.tile(m[None, None, :], (Hm, Wm, 1))

    # --- 3) Grid segments & index remap (for vectorized composition) ---
    segments = build_grid_segments(Hm, Wm, cell=int(cell)).astype(np.int32)  # (Hm,Wm) labels
    seg_ids = np.unique(segments)
    D = int(seg_ids.size)

    # Map labels to 0..D-1 quickly via LUT
    max_label = int(seg_ids.max())
    lut = np.full(max_label + 1, -1, dtype=np.int32)
    lut[seg_ids] = np.arange(D, dtype=np.int32)
    seg_idx = lut[segments]  # (Hm,Wm) int32 in [0..D-1]

    # --- 4) Problem spec & guard against sample explosion ---
    problem = {
        "num_vars": D,
        "names": [f"s{i}" for i in range(D)],
        "bounds": [(0.0, 1.0)] * D,
    }

    # Saltelli total Ns ≈ N * (D + 2)  (when second_order=False)
    NS_MAX = 6000  # keep within a few minutes on CPU; adjust to your box
    Ns_est = int(N * (D + 2)) if not calc_second_order else int(N * (2 * D + 2))
    if Ns_est > NS_MAX:
        N_old = N
        denom = (D + 2) if not calc_second_order else (2 * D + 2)
        N = max(8, NS_MAX // max(denom, 1))
        print(f"[Sobol] D={D}, clamping N: {N_old} -> {N} (Ns≈{N * denom}) to limit runtime.")

    # --- 5) Saltelli samples ---
    Z = saltelli.sample(problem, N, calc_second_order=calc_second_order)  # (Ns, D)
    Ns = Z.shape[0]
    print(f"[Sobol] evaluating Ns={Ns} samples (batch={batch}, D={D})...")

    # --- Vectorized composer: out = (1-α)*baseline + α*original ---
    base_f = base_img.astype(np.float32)
    imr_f  = imr.astype(np.float32)

    def compose_from_alpha(alpha_row: np.ndarray) -> np.ndarray:
        # α per-segment -> per-pixel via seg_idx (vectorized)
        alpha_map = alpha_row.astype(np.float32)[seg_idx]           # (Hm,Wm)
        out = ( (1.0 - alpha_map)[..., None] * base_f +
                  alpha_map[..., None]       * imr_f )
        return np.clip(out, 0, 255).astype(np.uint8)

    # --- 6) Evaluate model in batches ---
    Ys = []
    i = 0
    while i < Ns:
        j = min(i + batch, Ns)
        imgs = [compose_from_alpha(Z[k]) for k in range(i, j)]
        with torch.no_grad():
            ts = torch.cat([to_tensor_rgb01(im, device) for im in imgs], dim=0)  # (B,3,Hm,Wm)
            logits = model(ts)
            if aggregate == "prob":
                vals = F.softmax(logits, dim=1)[:, target_label].detach().cpu().numpy()
            else:
                vals = logits[:, target_label].detach().cpu().numpy()
        Ys.append(vals)
        i = j
        # light heartbeat
        if j % (batch * 8) == 0 or j == Ns:
            print(f"[Sobol] {j}/{Ns} evaluated")

    Y = np.concatenate(Ys, axis=0)  # (Ns,)

    # --- 7) Sobol analysis (S1, ST) ---
    res = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)
    S1 = np.asarray(res["S1"], dtype=np.float32)   # (D,)
    ST = np.asarray(res["ST"], dtype=np.float32)   # (D,)

    # --- 8) Direction via correlation sign between α_d and Y ---
    corr = np.zeros(D, dtype=np.float32)
    for d in range(D):
        xd = Z[:, d]
        c = np.corrcoef(xd, Y)[0, 1]
        corr[d] = c if np.isfinite(c) else 0.0

    signed_ST = ST * np.sign(corr).astype(np.float32)  # (D,)

    # --- 9) Paint to image (vectorized) ---
    # Instead of looping cells, just index by seg_idx
    heat_m = signed_ST[seg_idx].astype(np.float32)     # (Hm,Wm)

    # --- 10) Normalise to [-1,1] & upsample ---
    if normalize == "robust":
        lo = np.percentile(heat_m, 100.0 - robust_pct)
        hi = np.percentile(heat_m, robust_pct)
        if hi > lo:
            h = (np.clip(heat_m, lo, hi) - lo) / (hi - lo + 1e-12) * 2.0 - 1.0
        else:
            h = heat_m / (np.max(np.abs(heat_m)) + 1e-12)
    else:
        h = heat_m / (np.max(np.abs(heat_m)) + 1e-12)

    heat_up = cv2.resize(h, (W0, H0), interpolation=cv2.INTER_CUBIC)
    overlay = _overlay_redblue(img_rgb_uint8, heat_up, alpha=0.55)

    stats = {"S1": S1, "ST": ST, "corr": corr}
    return heat_up, overlay, stats
