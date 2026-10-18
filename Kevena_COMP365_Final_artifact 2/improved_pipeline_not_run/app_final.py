"""
Object Image Classifier - web app (COMP 365 final project, Ryan Kevena)
Upload a photo; the final model chosen by the notebook returns its top-3 CIFAR-10 classes.
Run:  python app.py            (opens at http://127.0.0.1:7860)
      python app.py --share    (also creates a temporary public link, useful in Colab)
"""
import argparse
import gradio as gr
from classifier import Classifier  # (app_final.py: version of app.py for the improved pipeline)

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="outputs/final_model.pt")
ap.add_argument("--port", type=int, default=7860)
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

clf = Classifier(args.weights)


def predict(img):
    return {} if img is None else clf.predict(img, top_k=3)


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a photo"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 prediction"),
    title="Object Image Classifier",
    description=(f"Final model: {clf.cfg['method']} (ResNet-18, trained on {clf.cfg['n_train']:,} CIFAR-10 images). "
                 "Works best on a single object from: " + ", ".join(clf.classes) + "."),
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)
