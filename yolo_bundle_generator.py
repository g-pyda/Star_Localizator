import numpy as np
import cv2
import os
import json
import yaml
from sklearn.model_selection import train_test_split

# --- CONFIGURATION - INPUT ---- #
BASE_PATH = "dataset"                   # base path for unprocessed dataset
INPUT_IMAGES = f"{BASE_PATH}/images"    # path to .npz files
INPUT_LABELS = f"{BASE_PATH}/labels"    # path to .json files

# --- CONFIGURATION - OUTPUT --- #
OUTPUT_DIR = "yolo_dataset"             # output directory
STAR_BOX_SIZE = 16                       # box size of a star (for YOLO)

# helper function - converting the files from specific split
def process_split(files, split_name):
    print(f"Processing the set {split_name} ({len(files)} files)...")

    for json_name in files:
        # Reading the JSON
        with open(os.path.join(INPUT_LABELS, json_name), 'r') as f:
            data = json.load(f)

        img_id = data['image_id']
        h_orig, w_orig = data['image_shape']  # SDSS shape [H, W]

        # Localization of .npz file based on image_id
        npz_path = os.path.join(INPUT_IMAGES, f"{img_id}.npz")

        if not os.path.exists(npz_path):
            print(f"Ostrzeżenie: Brak pliku {npz_path} dla labelki {json_name}")
            continue

        # Reading and preprocession of the photo
        npz_data = np.load(npz_path)
        key = list(npz_data.keys())[0]
        img_array = npz_data[key]

        # Normalization to uint8 (0-255)
        img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Conversion to RGB (3 channels)
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        # Save of the picture in PNG
        cv2.imwrite(f"{OUTPUT_DIR}/{split_name}/images/{img_id}.png", img_array)

        # Creating YOLO .txt etiquette
        label_path = f"{OUTPUT_DIR}/{split_name}/labels/{img_id}.txt"
        with open(label_path, 'w') as f_out:
            for obj in data['objects']:
                # Fetching x, y from JSON
                x_px = obj['x']
                y_px = obj['y']

                # Coordinates normalization (0.0 - 1.0)
                x_center = x_px / w_orig
                y_center = y_px / h_orig
                width = STAR_BOX_SIZE / w_orig
                height = STAR_BOX_SIZE / h_orig

                # File writing: class=0, x, y, w, h
                f_out.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def prepare_yolo_dataset():
    # Creating the structure of YOLO-compatible dataset
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # Download of all JSON files and split on training and validation files
    json_files = [f for f in os.listdir(INPUT_LABELS) if f.endswith('.json')]

    if not json_files:
        print("[ERROR] No .json files found in input folder")
        return
    train_jsons, val_jsons = train_test_split(json_files, test_size=0.2, random_state=42)

    # Processing the split parts
    process_split(train_jsons, 'train')
    process_split(val_jsons, 'val')

    # Creating the YAML file
    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'star'}
    }

    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f_yaml:
        yaml.dump(yaml_data, f_yaml, default_flow_style=False)

    print(f"\nSuccess! Data saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_yolo_dataset()