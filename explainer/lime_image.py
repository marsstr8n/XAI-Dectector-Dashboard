import numpy as np
from typing import Optional, Union, Callable
from lime import lime_image
from skimage.segmentation import mark_boundaries
from .lime_segments import slic_segments

def _weights_to_heatmap(labels: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    """
    Convert LIME's superpixel weights (local_exp) into a dense heatmap in [-1, 1].
    """
    h, w = labels.shape
    heat = np.zeros((h, w), dtype=np.float32)
    if not weights:
        return heat
    # fill each superpixel id with its weight
    for sp_id, wgt in weights.items():
        heat[labels == sp_id] = float(wgt)
    # normalize to [-1,1] for visualization
    mx = np.max(np.abs(heat)) + 1e-8
    heat = heat / mx
    return heat

def lime_explain_image(
    img_rgb_uint8: np.ndarray,
    classifier_fn: Callable[[np.ndarray], np.ndarray],   # returns (N,C)
    *,
    target_label: Optional[Union[int, str]] = "auto",    # int class id or "auto"
    num_samples: int = 2000,
    n_segments: int = 80,
    compactness: float = 10.0,
    sigma: float = 1.0,
    top_k_features: int = 10,       # how many positive superpixels to keep in mask
    positive_only: bool = True,     # only show regions that increase the prob
    hide_rest: bool = False,        # mask out non-selected regions
    random_state: int = 42,
):
    """
    Returns:
      mask_bool: (H, W) mask of selected superpixels
      overlay_rgb: (H, W, 3) pretty overlay with boundaries
      heatmap_float: (H, W) per-pixel LIME weight map in [-1, 1]
    """
    explainer = lime_image.LimeImageExplainer(verbose=False, random_state=random_state)

    # Custom segmentation - for granularity tuning
    def segmentation_fn(x):
        # x is (H, W, 3) float [0,1] or uint8; convert to uint8 for slic
        x_u8 = np.clip(x * 255.0, 0, 255).astype(np.uint8) if x.dtype != np.uint8 else x
        return slic_segments(x_u8, n_segments=n_segments, compactness=compactness, sigma=sigma, start_label=0)

    explanation = explainer.explain_instance(
        img_rgb_uint8,
        classifier_fn=classifier_fn,   # expects (N,H,W,3) uint8 -> (N,C)
        top_labels=2,                  # get both labels; pick one below
        hide_color=0,
        num_samples=int(num_samples),
        segmentation_fn=segmentation_fn,
    )

    # pick the label to visualise
    if isinstance(target_label, int):
        label = int(target_label)
    else:
        label = int(explanation.top_labels[0])  # "auto": highest prob per LIME

    # pull superpixel importance and convert to dense heatmap
    # local_exp[label] is list of (superpixel_id, weight)
    sp_weights = dict(explanation.local_exp[label])
    labels_img = explanation.segments
    heatmap = _weights_to_heatmap(labels_img, sp_weights)  # [-1, 1]

    # get LIME's binary mask + pretty image
    temp, mask = explanation.get_image_and_mask(
        label=label,
        positive_only=bool(positive_only),
        num_features=int(top_k_features),
        hide_rest=bool(hide_rest),
    )
    # temp is float [0,1]; convert to uint8 RGB
    overlay_rgb = (np.clip(temp, 0, 1) * 255).astype(np.uint8)
    # draw boundaries on top for readability
    overlay_rgb = (mark_boundaries(overlay_rgb, labels_img, color=(1, 0, 0), mode='thick') * 255).astype(np.uint8)

    return mask.astype(bool), overlay_rgb, heatmap
