# xception_min.py
# pip install timm torch torchvision opencv-python numpy
import torch, torch.nn as nn, numpy as np, cv2, timm

# Build a vanilla Xception backbone, 2-class head
model = timm.create_model('legacy_xception', pretrained=False, num_classes=2)
# --- remap keys so heads match and stray blocks are ignored ---
def _remap_state_dict_keys(sd: dict) -> dict:
    remapped = {}
    for k, v in sd.items():
        # strip common wrappers
        for pref in ("module.", "backbone.", "model.", "net."):
            if k.startswith(pref):
                k = k[len(pref):]
        # some xception checkpoints store pointwise conv as 1D: [C] -> [C,1,1]
        if "pointwise" in k and getattr(v, "ndim", 0) == 1:
            v = v.unsqueeze(-1).unsqueeze(-1)

        # map classifier name: last_linear.* --> fc.*
        if k.startswith("last_linear."):
            k = k.replace("last_linear.", "fc.", 1)

        # drop aux/adapter heads we don't have
        if k.startswith("adjust_channel.") or ".adjust_channel." in k:
            continue
        if k.startswith("classifier.") or ".classifier." in k:
            continue

        # keep everything else
        remapped[k] = v
    return remapped

# --- load DeepfakeBench checkpoint (fine-tuned detector) ---
ckpt = torch.load("pretrained_model/xception_best.pth", map_location="cpu", weights_only=True)
state = ckpt.get("state_dict", ckpt)
state = _remap_state_dict_keys(state)
missing, unexpected = model.load_state_dict(state, strict=False)
print("After remap -> Missing:", missing, "| Unexpected:", unexpected)  # should be []



# Key fixups so it loads into timm:
new_state = {}
for k, v in state.items():
    # strip common prefixes
    for pref in ("module.", "backbone.", "model.", "net."):
        if k.startswith(pref):
            k = k[len(pref):]
    # DeepfakeBench patches pointwise conv weights as [C] -> [C,1,1]
    if "pointwise" in k and v.ndim == 1:
        v = v.unsqueeze(-1).unsqueeze(-1)
    # Drop any 'fc.*' from a different head
    if k.startswith("fc.") or ".fc." in k:
        continue
    new_state[k] = v

model.eval()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

IMG_SIZE = 256  

MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)
STD  = torch.tensor([0.5, 0.5, 0.5]).view(1,3,1,1)

@torch.no_grad()
@torch.no_grad()
def predict_proba(images_rgb_01):
    arr = np.stack(images_rgb_01, 0).transpose(0,3,1,2).astype("float32")
    t = torch.from_numpy(arr)
    t = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE),
                                        mode="bilinear", align_corners=False)
    t = t.to(device, non_blocking=True)
    t = (t - MEAN.to(t.device)) / STD.to(t.device)

    logits = model(t)
    probs = torch.softmax(logits, dim=1)  # [P(real), P(fake)]
    return probs.cpu().numpy()

