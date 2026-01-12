import json
from pathlib import Path

LABELS_DIR = Path("dataset/labels")
SPLITS_DIR = Path("dataset/splits")

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_DIR.glob("*.json"))
    if not label_files:
        print("No label files found in dataset/labels")
        return

    train_ids = []
    val_ids = []

    for p in label_files:
        with open(p, "r", encoding="utf-8") as f:
            lbl = json.load(f)

        image_id = lbl.get("image_id") or p.stem

        # Prefer star_id from label; fallback to parsing "star<id>_" prefix
        sid = None
        try:
            objs = lbl.get("objects", [])
            if objs and "star_id" in objs[0]:
                sid = int(objs[0]["star_id"])
        except Exception:
            sid = None

        if sid is None:
            try:
                sid = int(image_id.split("_")[0].replace("star", ""))
            except Exception:
                sid = 0

        if sid % 10 == 0:
            val_ids.append(image_id)
        else:
            train_ids.append(image_id)

    (SPLITS_DIR / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    print(f"Wrote splits: train={len(train_ids)} val={len(val_ids)} from {len(label_files)} labels")

if __name__ == "__main__":
    main()
