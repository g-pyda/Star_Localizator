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
        img_path = IMAGES_DIR / f"{image_id}.npy"
        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        img = np.load(img_path)
        obj = lbl["objects"][0]
        x, y = obj["x"], obj["y"]

        plt.figure()
        plt.imshow(img, origin="lower")
        plt.scatter([x], [y])
        plt.title(image_id)
        plt.show()

if __name__ == "__main__":
    main()
