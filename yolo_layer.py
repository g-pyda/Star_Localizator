from ultralytics import YOLO
import torch

def train_star_detector():
    # 1. Sprawdzenie dostępności GPU
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Używam urządzenia: {device}")

    # 2. Wybór modelu bazowego
    # YOLO11n (nano) - najszybszy, dobry do testów
    # YOLO11s (small) - nieco lepsza dokładność, nadal szybki
    model = YOLO("yolo11n.pt")

    # 3. Rozpoczęcie treningu
    # Parametry dostosowane do małych obiektów (gwiazd)
    model.train(
        data="data.yaml",                    # Ścieżka do Twojego pliku yaml
        epochs=100,                          # Liczba epok (przy 10k zdjęć 50-100 wystarczy)
        imgsz=1280,                          # Rozmiar obrazu - kluczowy dla małych gwiazd!
        batch=8,                             # Rozmiar paczki (zmniejsz do 8 lub 4, jeśli braknie VRAM)
        patience=10,                         # Early stopping: przerwij, jeśli brak poprawy przez 10 epok
        save=True,                           # Zapisuj wagi modelu
        device=device,                       # Użycie GPU lub CPU
        workers=8,                           # Liczba wątków do ładowania danych
        project="star_detection_project",    # Nazwa folderu projektu
        name="yolo11s_stars_v1",             # Nazwa konkretnego eksperymentu
        exist_ok=True,
        # Augmentacje
        mosaic=0.5,                          # Zmniejszamy mosaic, by nie "rozrywać" konstelacji za bardzo
        mixup=0.1                            # Może pomóc przy gęstych polach gwiazd
    )

    print("Trening zakończony!")

if __name__ == "__main__":
    train_star_detector()
    # # Podaj ścieżkę do Twojego najlepszego modelu
    # # Zazwyczaj jest to: runs/detect/star_detection_v1/weights/best.pt
    # model = YOLO("star_detection_project/yolo11s_stars_v1/weights/best.pt")
    #
    # # Uruchom detekcję
    # results = model.predict(source="yolo_dataset/train/images/run4152_cam6_frame66_r.png", imgsz=1280, conf=0.02)
    #
    # # Pokaż wyniki
    # results[0].show()
