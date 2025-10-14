import numpy as np
from typing import Optional, Union, Callable
from lime import lime_image
from skimage.segmentation import mark_boundaries
from .lime_segments import slic_segments

def lime_explain_contours(
    img_rgb_uint8: np.ndarray,
    classifier_fn: Callable[[np.ndarray], np.ndarray],
    *,
    target_label: Optional[Union[int, str]] = "auto",
    num_samples: int = 2000,
    n_segments: int = 80,
    compactness: float = 18.0,
    sigma: float = 1.5,
    top_k_features: int = 2, # initial K
    positive_only: bool = True,
    min_total_area_ratio: float = 0.12,  # grow until union area >= 12% (set 0 to disable)
    max_k: int = 12, # don’t grow beyond this
    random_state: int = 42,
    contour_color=(1, 1, 0), # yellow
    thickness_mode: str = "thick", # 'thick' | 'inner' | 'outer'
):
    """
    Returns:
      mask_union_bool : (H, W) union of selected superpixels
      overlay         : (H, W, 3) original image with a single yellow outline
    """
    explainer = lime_image.LimeImageExplainer(verbose=False, random_state=random_state)

    def segmentation_fn(x):
        x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8) if x.dtype != np.uint8 else x
        return slic_segments(
            x_u8, n_segments=n_segments,
            compactness=compactness, sigma=sigma, start_label=0
        )

    explanation = explainer.explain_instance(
        img_rgb_uint8,
        classifier_fn=classifier_fn,
        top_labels=2,
        hide_color=0,
        num_samples=int(num_samples),
        segmentation_fn=segmentation_fn,
    )

    # pick the class
    if isinstance(target_label, int):
        label = int(target_label)
    else:
        label = int(explanation.top_labels[0])

    # Start with K and grow until union area hits threshold
    H, W = img_rgb_uint8.shape[:2]
    total_px = H * W
    K = int(top_k_features)
    mask_union = np.zeros((H, W), dtype=bool)

    # sorted by absolute weight; positive_only filters sign in get_image_and_mask()
    while True:
        _, mask_k = explanation.get_image_and_mask(
            label=label,
            positive_only=bool(positive_only),
            num_features=int(K),
            hide_rest=False,
        )
        mask_union = mask_k.astype(bool)
        if (min_total_area_ratio <= 0) or (mask_union.sum() / total_px >= min_total_area_ratio) or (K >= max_k):
            break
        K += 1

    # Outline ON the original image
    base = (img_rgb_uint8.astype(np.float32) / 255.0)
    outlined = mark_boundaries(base, mask_union, color=contour_color, mode=thickness_mode)
    overlay = (outlined * 255).astype(np.uint8)
    return mask_union, overlay
