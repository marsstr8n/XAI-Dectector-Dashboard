from typing import Optional, Union
from .gradcam_pp import gradcam_pp
from .guided_backprop import guided_backprop
from .guided_gradcam import guided_gradcam
from .lime_contours import lime_explain_contours
from .lime_wrap import LimePredictor
from .shap_grid import shap_grid_explain
from .occlusion_map import occlusion_explain
from .occlusion_captum import occlusion_explain_captum
from .sobol_grid import sobol_grid_explain



# gradcam++
def explain_gradcampp(model, device: str, img_rgb_uint8,
                      target_category: Optional[Union[int, str]] = "auto",
                      eigen_smooth: bool = True,
                      aug_smooth: bool = True,
                      gamma: float = 1.25,
                      keep_percent: float = 30.0):
    return gradcam_pp(model, device, img_rgb_uint8,
                      target_category=target_category,
                      eigen_smooth=eigen_smooth,
                      aug_smooth=aug_smooth,
                      gamma=gamma, keep_percent=keep_percent)

# guided_grad_cam
def explain_guided_backprop(model, device: str, img_rgb_uint8,
                            target_category: Optional[Union[int, str]] = "auto"):
    return guided_backprop(model, device, img_rgb_uint8, target_category=target_category)

def explain_guided_gradcam(model, device: str, img_rgb_uint8,
                           target_category: Optional[Union[int, str]] = "auto",
                           mask_percent: float = 30.0,
                           alpha: float = 1.5,
                           background: str = "gray"):
    return guided_gradcam(model, device, img_rgb_uint8,
                          target_category=target_category,
                          mask_percent=mask_percent,
                          alpha=alpha,
                          background=background)


    
# LIME
def explain_lime_contours(
    img_rgb_uint8, predict_proba_fn, *,
    target_label="auto",
    num_samples=2000,
    n_segments=80,
    compactness=18.0,
    sigma=1.5,
    top_k_features=2,
    positive_only=True,
    min_total_area_ratio=0.12,
    max_k=12,
    random_state=42,
    thickness_mode="thick",
):
    clf = LimePredictor(predict_proba_fn=predict_proba_fn)
    return lime_explain_contours(
        img_rgb_uint8, clf,
        target_label=target_label,
        num_samples=num_samples,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        top_k_features=top_k_features,
        positive_only=positive_only,
        min_total_area_ratio=min_total_area_ratio,
        max_k=max_k,
        random_state=random_state,
        thickness_mode=thickness_mode,
    )


# shap component 
def explain_shap_grid(
    model, device, img_rgb_uint8, **kwargs
):
    return shap_grid_explain(model, device, img_rgb_uint8, **kwargs)



# occlusion component
def explain_occlusion(
    model, device, img_rgb_uint8, **kwargs
):
    return occlusion_explain(model, device, img_rgb_uint8, **kwargs)


def explain_occlusion_captum(model, device, img_rgb_uint8, **kwargs):
    """
    Captum standard implementation
    """
    return occlusion_explain_captum(model, device, img_rgb_uint8, **kwargs)


# sobol component
def explain_sobol_grid(model, device, img_rgb_uint8, **kwargs):
    """
    Sobol-based occlusion on a grid of superpixels.
    """
    return sobol_grid_explain(model, device, img_rgb_uint8, **kwargs)
