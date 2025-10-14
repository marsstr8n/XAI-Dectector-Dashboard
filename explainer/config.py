# 1
import torch

# ensure this consistent with xception_min.py
IMG_SIZE = 256  

# [-1, 1] scaling used in detector (MEAN=STD=0.5)
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
