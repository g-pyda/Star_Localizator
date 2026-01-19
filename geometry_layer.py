import numpy as np
import json
from scipy.spatial import distance
from ultralytics import YOLO
from astroquery.simbad import Simbad

# --- CONFIGURATION ---
MIN_CONFIDENCE = 0.5
# --- EXAMPLE DATA ---
CATALOG_PATH = "dataset/labels/run4822_cam6_frame535_r.json"
MODEL_PATH = "star_detection_project/yolo11s_stars_v2/best.pt"
IMAGE_PATH = "yolo_dataset/train/images/run4822_cam6_frame535_r.png"


def get_common_name(source_id):
    """
    Tries to get a common name of the star (eg. 'Betelgeuse') based on Gaia ID
    """
    try:
        # SIMBAD takes the Gaia identifier
        result_table = Simbad.query_object(f"Gaia DR3 {source_id}")

        if result_table is not None:
            # Returns common name
            return result_table['MAIN_ID'][0]
        # Star not found - no name
        return None
    except Exception:
        return None


def get_star_descriptors(points):
    """
    Created geometric signatures for three stars combinations
    Signature - sorted proportions of the triangle sides (resistant to scaling and rotation)
    """
    descriptors = []
    # Taking 10 the brightest stars in the picture (N^3 complexity)
    n = min(len(points), 10)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                dists = sorted([
                    distance.euclidean(p1, p2),
                    distance.euclidean(p2, p3),
                    distance.euclidean(p1, p3)
                ])
                # Proportions of the sides (middle/shortest , longest/shortest)
                if dists[2] > 0:
                    descriptors.append({
                        'sig': (dists[1] / dists[2], dists[0] / dists[2]),
                        'ids': (i, j, k)
                    })
    return descriptors


def identify_stars(image_path, model_path, catalog_json):
    # Downloading the model and detecting the stars
    model = YOLO(model_path)
    results = model.predict(image_path, conf=MIN_CONFIDENCE, imgsz=1280)[0]

    # Taking out the centers of the boxes (x_center, y_center)
    detected_points = results.boxes.xywh.cpu().numpy()[:, :2]
    print(f"Wykryto {len(detected_points)} gwiazd przez YOLO.")

    # Loading the training data
    with open(catalog_json, 'r') as f:
        catalog_data = json.load(f)

    catalog_stars = catalog_data['objects']
    catalog_points = np.array([[s['x'], s['y']] for s in catalog_stars])
    catalog_names = [s['source_id'] for s in catalog_stars]

    # Genereting the descriptors for the star set
    det_desc = get_star_descriptors(detected_points)
    cat_desc = get_star_descriptors(catalog_points)

    # Adjustment of the star
    matches = {}            # Dict: detection_id -> catalog_id
    threshold = 0.01        # Geometry error threshold

    for d in det_desc:
        for c in cat_desc:
            err = np.linalg.norm(np.array(d['sig']) - np.array(c['sig']))
            if err < threshold:
                # if triangles fit - assign them
                for i in range(3):
                    matches[d['ids'][i]] = catalog_names[c['ids'][i]]

    # Display the results
    print("\n--- IDENTIFIED STARS ---")
    for idx, name in matches.items():
        pos = detected_points[idx]
        proper_name = get_common_name(name)
        if proper_name is not None:
            name = proper_name
        print(f"Star in point {pos} is presumably: {name}")

    return matches

# --- RUN ---
if __name__ == "__main__":
    identify_stars(IMAGE_PATH, MODEL_PATH, CATALOG_PATH)