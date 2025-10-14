
import io, os, tempfile
import numpy as np
from PIL import Image
import streamlit as st
import torch

# --- detector & wrappers ---
from df_detectors.xception_min import model as xcep_model, device as xcep_device, predict_proba

# --- XAI APIs ---
from explainer.api import (
    explain_gradcampp,
    explain_guided_backprop,
    explain_guided_gradcam,
    explain_lime_contours,
    explain_shap_grid,
    explain_occlusion,          
    explain_sobol_grid,        
)

# UI sizing
IMG_W = 360 # adjust this part later



# --- face cropper (disk-based) ---
try:
    from face_crop.face_cropper_run import crop_face_from_image
    MEDIAPIPE_OK = True
except Exception:
    crop_face_from_image = None
    MEDIAPIPE_OK = False

st.set_page_config(page_title="Deepfake XAI", layout="wide")
st.title("Deepfake Detector + XAI")

@st.cache_resource(show_spinner=False)
def get_model_and_device():
    xcep_model.eval()
    return xcep_model, xcep_device

model, device = get_model_and_device()


def _predict_fake_prob(img_rgb_uint8: np.ndarray) -> float:
    # ensure RGB float in [0,1]
    if img_rgb_uint8.ndim == 2:
        img_rgb_uint8 = np.dstack([img_rgb_uint8] * 3)
    img01 = img_rgb_uint8.astype(np.float32)
    if img01.max() > 1.0:    # convert 0..255 -> 0..1
        img01 /= 255.0
    probs = predict_proba([img01]) # shape (1, 2) = [p(real), p(fake)]
    return float(probs[0, 1])

def _read_upload(upl) -> Image.Image:
    return Image.open(io.BytesIO(upl.getvalue())).convert("RGB")

def _maybe_crop_to_face(img_pil: Image.Image) -> np.ndarray:
    if not MEDIAPIPE_OK or crop_face_from_image is None:
        st.warning("MediaPipe not available — skipping face crop.")
        return np.array(img_pil)

    with tempfile.TemporaryDirectory() as td:
        in_path  = os.path.join(td, "in.png")
        out_path = os.path.join(td, "crop.png")
        img_pil.save(in_path)
        try:
            result = crop_face_from_image(
                in_path, out_path,
                min_score=0.6, strategy="largest", margin=0.25, min_side=160, model_selection=1
            )
            if result is None or not os.path.exists(out_path):
                st.warning("No face detected — using original image.")
                return np.array(img_pil)
            return np.array(Image.open(out_path).convert("RGB"))
        except Exception as e:
            st.warning(f"Face crop failed ({e}); using original image.")
            return np.array(img_pil)

# ---------------- 2-column layout ----------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Upload")
    upl = st.file_uploader("PNG / JPG only", type=["png", "jpg", "jpeg"])
    if not upl:
        st.info("Please upload an image to begin.")
        st.stop()

    orig = _read_upload(upl)
    st.image(orig, caption="Uploaded image", width=IMG_W)

    st.subheader("Preprocessing")
    img = _maybe_crop_to_face(orig)
    st.image(img, caption="Image used by detector & explainers", width=IMG_W)


with right:
    with st.spinner("Running detector..."):
        p_fake = _predict_fake_prob(img)
    st.metric("Detector", f"{'FAKE' if p_fake >= 0.5 else 'REAL'}", delta=f"p(fake)={p_fake:.2f}")

    st.subheader("Explanations")
    tabs = st.tabs([
        "Grad-CAM++", "Guided Backprop", "Guided Grad-CAM",
        "LIME (contours)", "SHAP (grid dots)",
        "Occlusion (custom)", "Sobol (grid)"
    ])

    with tabs[0]:
        with st.spinner("Grad-CAM++…"):
            heat, overlay = explain_gradcampp(
                model, device, img,
                target_category=1,
                eigen_smooth=True, aug_smooth=True,
                gamma=1.35, keep_percent=35.0,
            )
        st.image(overlay, caption="Grad-CAM++ overlay", width=IMG_W)

    with tabs[1]:
        with st.spinner("Guided Backprop…"):
            gb = explain_guided_backprop(model, device, img, target_category=1)
        st.image(gb, caption="Guided Backprop", width=IMG_W)

    with tabs[2]:
        with st.spinner("Guided Grad-CAM…"):
            gg = explain_guided_gradcam(
                model, device, img,
                target_category=1,
                mask_percent=25,   # 20–35 works well
                alpha=1.5,
                background="gray",
            )
        st.image(gg, caption="Guided Grad-CAM", width=IMG_W)

    with tabs[3]:
        with st.spinner("LIME…"):
            _, lime_img = explain_lime_contours(
                img, predict_proba,
                target_label=1,
                num_samples=3000,
                n_segments=100,
                compactness=18.0,
                sigma=1.5,
                top_k_features=2,
                min_total_area_ratio=0.12,
                max_k=8,
                thickness_mode="thick",
            )
        st.image(lime_img, caption="LIME (selected superpixels outlined)", width=IMG_W)

    with tabs[4]:
        with st.spinner("SHAP (grid)…"):
            shap_heat, shap_overlay = explain_shap_grid(
                model, device, img,
                target_label=1,
                cell=12, dots_only=True, tau=0.15, dots_on_white=False,
            )
        st.image(shap_overlay, caption="SHAP (blue=real, red=fake)", width=IMG_W)

    with tabs[5]:
        with st.spinner("Occlusion…"):
            occ_heat, occ_overlay = explain_occlusion(
                model, device, img,
                target_label=1,
                patch=24, stride=12,
                baseline="blur",
                normalize="robust", robust_pct=97.0,
            )
        st.image(occ_overlay, caption="Occlusion (red=↑fake, blue=↓fake)", width=IMG_W)

    with tabs[6]:
        with st.spinner("Sobol…"):
            sobol_heat, sobol_overlay, sobol_stats = explain_sobol_grid(
                model, device, img,
                target_label=1,
                cell=20, N=32,
                calc_second_order=False,
                baseline="gray",
                aggregate="logit",
                normalize="robust", robust_pct=97.0,
                batch=32,
            )
        st.image(sobol_overlay, caption="Sobol grid (red=↑fake, blue=↓fake)", width=IMG_W)
