import numpy as np, cv2
from typing import Optional, Union
from .gradcam_pp import gradcam_pp
from .guided_backprop import guided_backprop  # returns mid-gray GB 

def guided_gradcam(
    model,
    device: str,
    img_rgb_uint8: np.ndarray,
    *,
    target_category: Optional[Union[int, str]] = "auto",
    mask_percent: float = 30.0,    # keep top-x% of CAM
    alpha: float = 1.2,            # CAM gate strength (1.0–2.0)
    background: str = "gray",      # 'gray' | 'black' | 'gb'
):
    """
    Guided Grad-CAM with controllable background.
    - background='gray'   
    - background='black'  
    - background='gb'     
    """
    # 1) Grad-CAM++ heatmap at original size, with enhancement
    cam_gray, _ = gradcam_pp(
        model, device, img_rgb_uint8,
        target_category=target_category,
        eigen_smooth=True, aug_smooth=True,
        gamma=1.25, keep_percent=mask_percent
    )  # (H0,W0) float in [0,1]

    # 2) Guided Backprop (uint8, mid-gray deprocess): (H1,W1,3)
    gb = guided_backprop(model, device, img_rgb_uint8, target_category=target_category).astype(np.float32)
    H1, W1 = gb.shape[:2]
    gb01 = gb / 255.0

    # 3) Align CAM to GB size and build a soft gate
    cam = cv2.resize(cam_gray, (W1, H1)).astype(np.float32)  # (H1,W1)
    cam = np.clip(cam, 0.0, 1.0) ** float(alpha)
    cam = cam[..., None]  # (H1,W1,1)

    # 4) Choose the background
    if background == "gray":
        base = np.full_like(gb01, 0.5, dtype=np.float32)        # mid-gray
    elif background == "gb":
        base = gb01.copy()                                      # keep GB everywhere
    else:  # "black"
        base = np.zeros_like(gb01, dtype=np.float32)

    # 5) Blend between background and masked GB
    eps = 1e-6
    masked = gb01 * cam
    out = np.clip(base * (1.0 - cam + eps) + masked, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)
