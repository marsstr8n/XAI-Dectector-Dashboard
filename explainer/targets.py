import torch.nn as nn

def get_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Returns the last Conv2d layer found in model (how CAM is).
    """
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last
