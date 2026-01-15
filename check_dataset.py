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

        image_id = lbl.get("image_id")
        if image_id is None:
            print("Invalid label file:", lbl_path)
            continue

        img_path = IMAGES_DIR / f"{image_id}.npz"
        if not img_path.exists():
            print("Missing image:", img_path)
            continue

        # Load image data from .npz
        try:
            img = np.load(img_path)["data"]
        except Exception as e:
            print("Failed to load image:", img_path, e)
            continue

        objects = lbl.get("objects", [])
        print(f"{image_id}: {len(objects)} objects")

        plt.figure(figsize=(6, 5))
        plt.imshow(img, origin="lower", cmap="gray")

        if objects:
            # Separate by catalog for clarity
            xs_gaia, ys_gaia = [], []
            xs_hyg, ys_hyg = [], []

            for o in objects:
                try:
                    x = float(o["x"])
                    y = float(o["y"])
                except Exception:
                    continue

                if o.get("catalog") == "gaia_dr3":
                    xs_gaia.append(x)
                    ys_gaia.append(y)
                else:
                    xs_hyg.append(x)
                    ys_hyg.append(y)

            if xs_gaia:
                plt.scatter(xs_gaia, ys_gaia, s=8, c="red", label="Gaia DR3")

            if xs_hyg:
                plt.scatter(xs_hyg, ys_hyg, s=20, c="cyan", marker="+", label="HYG")

            plt.legend(loc="upper right", fontsize=8)

        plt.title(image_id)
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    main()
