"""
Object Image Classifier - web app prototype (COMP 365, Practical Checkpoint 1)
Ryan Kevena

Upload a photo; the fine-tuned ResNet-18 from the pilot study returns its top-3 CIFAR-10 classes.
Run:  pip install -r requirements.txt   then   python app.py [weights.pt] [--share]   (opens at http://127.0.0.1:7860)
"""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18
from PIL import Image
import gradio as gr

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
WEIGHTS = ARGS[0] if ARGS else "outputs/resnet18_finetune_n5000.pt"   # add --share for a public link (e.g. in Colab)
IMG_SIZE = 64                                   # must match the notebook setting
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()


@torch.no_grad()
def predict(img, top_k=3):
    if img is None:
        return {}
    # Same preprocessing as training: shrink to 32x32 (CIFAR-10 size), upsample, normalize
    x = np.array(img.convert("RGB").resize((32, 32), Image.BILINEAR))
    x = torch.tensor(x).permute(2, 0, 1)[None].float() / 255.0
    x = F.interpolate(x, size=IMG_SIZE, mode="bilinear", align_corners=False)
    x = (x - MEAN) / STD
    probs = F.softmax(model(x), dim=1)[0]
    top = probs.topk(top_k)
    return {CLASSES[i]: float(p) for p, i in zip(top.values, top.indices)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 prediction"),
    title="Object Image Classifier (pilot prototype)",
    description=("ResNet-18 fine-tuned from ImageNet on a 5,000-image CIFAR-10 subset. "
                 "Works best on a single object from: " + ", ".join(CLASSES) + "."),
)

if __name__ == "__main__":
    demo.launch(share="--share" in sys.argv)
