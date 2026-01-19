import numpy as np
import json
from scipy.spatial import distance
from ultralytics import YOLO
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

def check_id(source_id):
    # Ważne: wysyłamy ID jako string, aby uniknąć zaokrągleń w zapytaniu
    query = f"SELECT source_id, ra, dec FROM gaiaedr3.gaia_source WHERE source_id = {source_id}"
    job = Gaia.launch_job(query)
    res = job.get_results()

    if len(res) == 0:
        print(f"❌ ID {source_id} NIE istnieje w bazie Gaia DR3 (prawdopodobnie zaokrąglone).")
    else:
        print(f"✅ ID {source_id} jest poprawne!")
        print(res)


def get_common_name(source_id):
    """
    Próbuje znaleźć popularną nazwę gwiazdy (np. 'Betelgeuse') na podstawie Gaia ID.
    """
    try:
        # SIMBAD rozumie identyfikatory Gaia
        result_table = Simbad.query_object(f"Gaia DR3 {source_id}")

        if result_table is not None:
            # Zwraca główną nazwę z katalogu SIMBAD
            return result_table['MAIN_ID'][0]
        return None
    except Exception:
        return None


# --- KONFIGURACJA ---
CATALOG_PATH = "dataset/labels/run4822_cam6_frame535_r.json"  # Przykładowy wzorzec
MODEL_PATH = "star_detection_project/yolo11s_stars_v2/best.pt"
IMAGE_PATH = "yolo_dataset/train/images/run4822_cam6_frame535_r.png"
MIN_CONFIDENCE = 0.25


def get_star_descriptors(points):
    """
    Tworzy geometryczne sygnatury dla trójek gwiazd.
    Sygnatura to posortowane stosunki boków trójkąta (odporne na skalę i obrót).
    """
    descriptors = []
    # Bierzemy 10 najjaśniejszych, aby nie przeciążyć obliczeń (N^3)
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
                # Stosunki boków (np. średni/najdłuższy i najkrótszy/najdłuższy)
                if dists[2] > 0:
                    descriptors.append({
                        'sig': (dists[1] / dists[2], dists[0] / dists[2]),
                        'ids': (i, j, k)
                    })
    return descriptors


def identify_stars(image_path, model_path, catalog_json):
    # 1. Wczytaj model i wykryj gwiazdy
    model = YOLO(model_path)
    results = model.predict(image_path, conf=MIN_CONFIDENCE, imgsz=1280)[0]

    # Wyciągnij środki wykrytych ramek (x_center, y_center)
    detected_points = results.boxes.xywh.cpu().numpy()[:, :2]
    print(f"Wykryto {len(detected_points)} gwiazd przez YOLO.")

    # 2. Wczytaj dane z katalogu (Twoje dane treningowe)
    with open(catalog_json, 'r') as f:
        catalog_data = json.load(f)

    catalog_stars = catalog_data['objects']
    catalog_points = np.array([[s['x'], s['y']] for s in catalog_stars])
    catalog_names = [s['source_id'] for s in catalog_stars]  # Możesz tu dać realną nazwę

    # 3. Generuj deskryptory dla obu zbiorów
    det_desc = get_star_descriptors(detected_points)
    cat_desc = get_star_descriptors(catalog_points)

    # 4. Dopasowanie (prosty brute-force dla deskryptorów)
    matches = {}  # Słownik: id_wykrytej_gwiazdy -> nazwa_z_katalogu
    threshold = 0.01  # Tolerancja błędu geometrii

    for d in det_desc:
        for c in cat_desc:
            err = np.linalg.norm(np.array(d['sig']) - np.array(c['sig']))
            if err < threshold:
                # Jeśli trójkąty pasują, przypisz nazwy punktom
                for i in range(3):
                    matches[d['ids'][i]] = catalog_names[c['ids'][i]]

    # 5. Wyświetl wyniki
    print("\n--- ZIDENTYFIKOWANE GWIAZDY ---")
    for idx, name in matches.items():
        pos = detected_points[idx]
        proper_name = get_common_name(name)
        if proper_name is not None:
            name = proper_name
        print(f"Gwiazda w punkcie {pos} to prawdopodobnie: {name}")

    return matches

# --- URUCHOMIENIE ---
if __name__ == "__main__":
    identify_stars(IMAGE_PATH, MODEL_PATH, CATALOG_PATH)