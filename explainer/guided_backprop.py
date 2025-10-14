import numpy as np, torch, inspect
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
from .tensor_io import to_tensor_rgb01

def _normalize_grad_rgb(g: np.ndarray) -> np.ndarray:
    """
    Use abs, per-channel percentile stretching (1..99), then scale to uint8.
    """
    g = np.abs(g)  # magnitude
    for c in range(3):
        ch = g[..., c]
        lo, hi = np.percentile(ch, 1), np.percentile(ch, 99)
        if hi > lo:
            ch = np.clip((ch - lo) / (hi - lo), 0, 1)
        else:
            ch = np.clip(ch, 0, 1)
        g[..., c] = ch
    return (g * 255).astype(np.uint8)

def _deprocess_like_repo(g: np.ndarray, clip_percent=1.0) -> np.ndarray:
    """
    Center to zero-mean, scale by std, then shift to 0.5 mid-gray and clip to 0..1.
    Finally map to uint8. This mimics Grad-CAM repo visuals.
    """
    g = g.astype(np.float32)
    g -= g.mean()
    g_std = g.std() + 1e-8
    g /= g_std
    g = g * 0.1 + 0.5  # contrast + mid-gray
    # optional: light clipping to remove tiny extremes
    lo, hi = np.percentile(g, clip_percent), np.percentile(g, 100 - clip_percent)
    g = np.clip((g - lo) / (hi - lo + 1e-8), 0, 1)
    return (g * 255).astype(np.uint8)

def _make_gb_model(model, device: str):
    sig = inspect.signature(GuidedBackpropReLUModel)
    if "device" in sig.parameters:
        return GuidedBackpropReLUModel(model=model, device=device)
    if "use_cuda" in sig.parameters:
        return GuidedBackpropReLUModel(model=model, use_cuda=device.startswith("cuda"))
    return GuidedBackpropReLUModel(model=model)

def guided_backprop(model, device: str, img_rgb_uint8: np.ndarray, target_category="auto"):
    model.eval().to(device)
    gb_model = _make_gb_model(model, device)
    t = to_tensor_rgb01(img_rgb_uint8, device)

    with torch.no_grad():
        logits = model(t)
    cat_idx = int(target_category) if isinstance(target_category, int) else int(torch.argmax(logits, 1).item())

    gb = gb_model(t, target_category=cat_idx)  # HxWx3 float
    return _deprocess_like_repo(gb)

    # return _normalize_grad_rgb(gb)

