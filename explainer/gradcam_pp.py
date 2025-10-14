from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .targets import get_last_conv_layer
from .tensor_io import to_tensor_rgb01
import numpy as np, cv2, torch
from typing import Optional, Union

def _contrast(cam: np.ndarray, gamma: float = 1.0, eps: float = 1e-6):
    cam = np.clip(cam, 0.0, 1.0)
    if gamma != 1.0:
        cam = np.power(cam + eps, gamma)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
    return cam

def _mask_low(cam: np.ndarray, keep_percent: float = 30.0):
    if keep_percent <= 0:
        return cam
    thr = np.percentile(cam, 100.0 - keep_percent)
    cam = cam.copy()
    cam[cam < thr] = 0.0
    cam /= (cam.max() + 1e-6)
    return cam

def gradcam_pp(
    model,
    device: str,
    img_rgb_uint8: np.ndarray,
    *,
    target_category: Optional[Union[int, str]] = "auto",
    eigen_smooth: bool = True,
    aug_smooth: bool = True,
    gamma: float = 1.25,
    keep_percent: float = 30.0,
):
    """
    Last-conv Grad-CAM++ with smoothing + contrast + top-x% masking.
    Returns (grayscale_cam[H,W], overlay_rgb_uint8[H,W,3]).
    """
    model.eval().to(device)

    # 1) prep input
    t = to_tensor_rgb01(img_rgb_uint8, device)

    # 2) pick class
    with torch.no_grad():
        logits = model(t)
    cat_idx = int(target_category) if isinstance(target_category, int) else int(torch.argmax(logits, 1).item())
    targets = [ClassifierOutputTarget(cat_idx)]

    # 3) last conv layer
    target_layer = get_last_conv_layer(model)

    # 4) CAM (new API: no use_cuda kwarg)
    with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
        grayscale = cam(
            input_tensor=t,
            targets=targets,
            eigen_smooth=eigen_smooth,
            aug_smooth=aug_smooth
        )[0]  # [Hc,Wc] in [0,1]

    # 5) resize to original + enhance
    H, W = img_rgb_uint8.shape[:2]
    cam_res = cv2.resize(grayscale, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_res = _contrast(cam_res, gamma=gamma)
    cam_res = _mask_low(cam_res, keep_percent=keep_percent)

    # 6) overlay (50/50)
    base01 = img_rgb_uint8.astype(np.float32) / 255.0
    heat_u8 = (cam_res * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET).astype(np.float32) / 255.0
    overlay = np.clip(base01 * 0.5 + color * 0.5, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return cam_res.astype(np.float32), overlay
