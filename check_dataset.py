import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

IMAGES_DIR = Path("./dataset/images")
LABELS_DIR = Path("./dataset/labels")

def main():
    label_files = sorted(LABELS_DIR.glob("*.json"))
    if not label_files:
        print("No labels found.")
        return

    sample = random.sample(label_files, k=min(10, len(label_files)))

    for lbl_path in sample:
        with open(lbl_path, "r", encoding="utf-8") as f:
            lbl = json.load(f)

        image_id = lbl["image_id"]
        img_path = IMAGES_DIR / f"{image_id}.npz"
        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        # Load image data from .npz
        img = np.load(img_path)["data"]

        objects = lbl.get("objects", [])
        print(f"{image_id}: {len(objects)} stars")

        plt.figure(figsize=(6, 5))
        plt.imshow(img, origin="lower", cmap="gray")

        if objects:
            xs = [o["x"] for o in objects]
            ys = [o["y"] for o in objects]
            plt.scatter(xs, ys, s=20)

        plt.title(image_id)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
