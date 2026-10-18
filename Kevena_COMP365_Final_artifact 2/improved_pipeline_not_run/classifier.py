"""Loads the final model exported by the notebook and predicts the top-k CIFAR-10 classes for one photo.
Used by app.py (web app) and test_phone_photos.py. Preprocessing matches the notebook's evaluation step."""
import json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from PIL import Image, ImageOps


class Classifier:
    def __init__(self, weights="outputs/final_model.pt", config=None):
        cfg = json.load(open(config or weights.replace(".pt", ".json")))
        self.cfg, self.classes, self.img_size = cfg, cfg["classes"], cfg["img_size"]
        self.mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def predict(self, img, top_k=3):
        """img: PIL image. Returns {class: probability} for the top_k classes."""
        img = ImageOps.exif_transpose(img).convert("RGB")         # respect phone-camera rotation
        x = np.array(img.resize((32, 32), Image.BILINEAR))         # shrink to CIFAR-10 size, like the training data
        x = torch.tensor(x).permute(2, 0, 1)[None].float() / 255.0
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        probs = F.softmax(self.model((x - self.mean) / self.std), dim=1)[0]
        top = probs.topk(top_k)
        return {self.classes[i]: float(p) for p, i in zip(top.values, top.indices)}
