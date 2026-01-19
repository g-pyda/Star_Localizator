import numpy as np
import cv2
import os
import json
import yaml
from sklearn.model_selection import train_test_split

# --- KONFIGURACJA ŚCIEŻEK WEJŚCIOWYCH ---
BASE_PATH = "dataset"  # Główny folder Twojego projektu
INPUT_IMAGES = f"{BASE_PATH}/images"  # Tu masz pliki .npz
INPUT_LABELS = f"{BASE_PATH}/labels"  # Tu masz pliki .json

# --- KONFIGURACJA WYJŚCIOWA ---
OUTPUT_DIR = "yolo_dataset"
STAR_BOX_SIZE = 8  # Rozmiar ramki wokół gwiazdy w pikselach (dla YOLO)


def prepare_yolo_dataset():
    # 1. Tworzenie struktury folderów YOLO
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # 2. Pobranie listy wszystkich plików JSON
    json_files = [f for f in os.listdir(INPUT_LABELS) if f.endswith('.json')]

    if not json_files:
        print("Błąd: Nie znaleziono plików JSON w folderze labels!")
        return

    # Podział na trening i walidację (80/20)
    train_jsons, val_jsons = train_test_split(json_files, test_size=0.2, random_state=42)

    def process_split(files, split_name):
        print(f"Przetwarzanie zbioru {split_name} ({len(files)} plików)...")

        for json_name in files:
            # Wczytanie JSON-a
            with open(os.path.join(INPUT_LABELS, json_name), 'r') as f:
                data = json.load(f)

            img_id = data['image_id']
            h_orig, w_orig = data['image_shape']  # SDSS shape [H, W]

            # Lokalizacja pliku .npz na podstawie image_id z JSONa
            npz_path = os.path.join(INPUT_IMAGES, f"{img_id}.npz")

            if not os.path.exists(npz_path):
                print(f"Ostrzeżenie: Brak pliku {npz_path} dla labelki {json_name}")
                continue

            # Wczytanie i przygotowanie obrazu
            npz_data = np.load(npz_path)
            # Wyciągamy tablicę (zazwyczaj pod 'arr_0' lub kluczem pasującym do band r/g/i)
            # Jeśli nie wiesz jaki jest klucz, bierzemy pierwszy dostępny
            key = list(npz_data.keys())[0]
            img_array = npz_data[key]

            # Normalizacja do uint8 (0-255)
            img_array = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            # Konwersja do RGB (3 kanały)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

            # Zapis obrazu jako PNG
            cv2.imwrite(f"{OUTPUT_DIR}/{split_name}/images/{img_id}.png", img_array)

            # Tworzenie etykiet YOLO .txt
            label_path = f"{OUTPUT_DIR}/{split_name}/labels/{img_id}.txt"
            with open(label_path, 'w') as f_out:
                for obj in data['objects']:
                    # Wyciągamy x, y z JSONa
                    x_px = obj['x']
                    y_px = obj['y']

                    # Normalizacja współrzędnych (0.0 - 1.0)
                    x_center = x_px / w_orig
                    y_center = y_px / h_orig
                    width = STAR_BOX_SIZE / w_orig
                    height = STAR_BOX_SIZE / h_orig

                    # Zapis: klasa=0, x, y, w, h
                    f_out.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Uruchomienie procesowania
    process_split(train_jsons, 'train')
    process_split(val_jsons, 'val')

    # 3. Tworzenie pliku data.yaml
    yaml_data = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'star'}
    }

    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f_yaml:
        yaml.dump(yaml_data, f_yaml, default_flow_style=False)

    print(f"\nSukces! Dane gotowe w folderze: {OUTPUT_DIR}")


if __name__ == "__main__":
    prepare_yolo_dataset()