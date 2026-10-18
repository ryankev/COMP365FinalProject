"""
Real-photo test for the final model (reported SEPARATELY from CIFAR-10 results).
Put your own photos in phone_photos/<class name>/ (for example phone_photos/cat/IMG_0001.jpg),
using the 10 CIFAR-10 class names, then run:   python test_phone_photos.py
Writes outputs/phone_photo_results.csv and outputs/phone_photo_summary.json.
"""
import argparse, glob, json, os, time
import pandas as pd
from PIL import Image
from classifier import Classifier

ap = argparse.ArgumentParser()
ap.add_argument("--photos", default="phone_photos")
ap.add_argument("--weights", default="outputs/final_model.pt")
args = ap.parse_args()

clf = Classifier(args.weights)
rows = []
for cls in clf.classes:
    files = sorted(f for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG") for f in glob.glob(f"{args.photos}/{cls}/*.{ext}"))
    for f in files:
        t0 = time.time(); top = clf.predict(Image.open(f), top_k=3)
        rows.append(dict(file=f, true=cls, top1=next(iter(top)), top3=list(top), top1_prob=round(max(top.values()), 3),
                         correct_top1=next(iter(top)) == cls, correct_top3=cls in top, latency_ms=round((time.time() - t0) * 1000)))

if not rows:
    raise SystemExit(f"No photos found. Put .jpg/.png files in {args.photos}/<class name>/ (class names: {', '.join(clf.classes)})")

df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/phone_photo_results.csv", index=False)
per_class = df.groupby("true")[["correct_top1", "correct_top3"]].mean().mul(100).round(1)
summary = dict(model=clf.cfg["method"], n_photos=len(df), top1_accuracy=round(df.correct_top1.mean() * 100, 1),
               top3_accuracy=round(df.correct_top3.mean() * 100, 1), photos_per_class=df.true.value_counts().to_dict(),
               per_class_top1=per_class.correct_top1.to_dict())
json.dump(summary, open("outputs/phone_photo_summary.json", "w"), indent=2)
print(per_class.to_string()); print(json.dumps(summary, indent=2))
print("Saved outputs/phone_photo_results.csv and outputs/phone_photo_summary.json")
