import numpy as np
import torch
from .config import IMG_SIZE, MEAN, STD

def to_tensor_rgb01(img_rgb_uint8: np.ndarray, device: str) -> torch.Tensor:
    """
    img_rgb_uint8: HxWx3 uint8 in [0,255]
    returns: 1x3xH'xW' float32 tensor normalized with (x-mean)/std on device
    """
    im = img_rgb_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(im.transpose(2, 0, 1)).unsqueeze(0)  # 1x3xHxW
    t = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                                        mode="bilinear", align_corners=False)
    t = t.to(device)
    t = (t - MEAN.to(device)) / STD.to(device)
    return t
