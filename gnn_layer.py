import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
import json
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
import os
import torch
import numpy as np
import cv2
import pickle
from ultralytics import YOLO
from scipy.spatial import Delaunay
import pickle

# --- KONFIGURACJA ---
PATH_YOLO = "star_detection_project/yolo11s_stars_v2/best.pt"
PATH_GNN = "star_gnn_best.pth"
PATH_ENCODER = "label_encoder.pkl"  # Zapisany LabelEncoder
IMAGE_PATH = "yolo_dataset/train/images/run4822_cam6_frame535_r.png"
CONFIDENCE_THRESHOLD = 0.01
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Twój fix na błąd DLL


def predict_stars():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Wczytanie modeli i encodera
    yolo_model = YOLO(PATH_YOLO)

    with open(PATH_ENCODER, 'rb') as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    gnn_model = StarGAT(in_channels=2, hidden_channels=64, out_channels=num_classes)
    gnn_model.load_state_dict(torch.load(PATH_GNN, map_location=device))
    gnn_model.to(device)
    gnn_model.eval()

    # 2. Detekcja YOLO (Wyciąganie współrzędnych x, y)
    img = cv2.imread(IMAGE_PATH)
    h_orig, w_orig = img.shape[:2]
    results = yolo_model.predict(img, imgsz=1280, conf=0.25)[0]

    # Pobieramy środki ramek i normalizujemy
    boxes = results.boxes.xywh.cpu().numpy()
    if len(boxes) < 3:
        print("Zbyt mało gwiazd do zbudowania grafu!")
        return

    coords = boxes[:, :2]  # x, y
    x_norm = coords[:, 0] / w_orig
    y_norm = coords[:, 1] / h_orig
    normalized_points = np.stack([x_norm, y_norm], axis=1)

    # 3. Budowa grafu dla zdjęcia użytkownika
    tri = Delaunay(normalized_points)
    edges = []
    for s in tri.simplices:
        edges.extend([[s[0], s[1]], [s[1], s[2]], [s[2], s[0]]])

    x_tensor = torch.tensor(normalized_points, dtype=torch.float).to(device)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    # 4. Inferencja GNN (Identyfikacja)
    with torch.no_grad():
        out = gnn_model(x_tensor, edge_index)
        predictions = out.argmax(dim=1).cpu().numpy()
        confidence = torch.nn.functional.softmax(out, dim=1).max(dim=1).values.cpu().numpy()

    # 5. Dekodowanie ID i Wizualizacja
    gaia_ids = le.inverse_transform(predictions)

    print(f"\nZnaleziono {len(gaia_ids)} obiektów:")
    for i, g_id in enumerate(gaia_ids):
        if confidence[i] > CONFIDENCE_THRESHOLD:  # Filtr pewności modelu
            print(f"Gwiazda {i}: Gaia ID {g_id} (Pewność: {confidence[i]:.2f})")

            # Rysowanie na obrazie
            x, y = int(coords[i][0]), int(coords[i][1])
            cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(img, str(g_id)[-6:], (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print("Wyszło z pętli")
    cv2.imshow("Zidentyfikowane Gwiazdy", img)
    cv2.waitKey(0)


def train_gnn_model(loader, num_classes, epochs=100, lr=0.001):
    # 1. Sprawdzenie urządzenia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Rozpoczynam trening GNN na: {device}")

    # 2. Inicjalizacja modelu (zdefiniowanego w kroku 3.1)
    # in_channels=2 (bo mamy x i y), hidden_channels=64, out_channels=num_classes
    model = StarGAT(in_channels=2, hidden_channels=64, out_channels=num_classes).to(device)

    # 3. Definicja Optymalizatora i Funkcji Straty
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct = 0
        total_nodes = 0

        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            out = model(data.x, data.edge_index)

            # Obliczanie straty (porównujemy przewidziane klasy z etykietami y)
            loss = criterion(out, data.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Obliczanie celności (accuracy) w locie
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total_nodes += data.num_nodes

        # Logowanie wyników
        avg_loss = total_loss / len(loader)
        acc = correct / total_nodes

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoka: {epoch:03d}, Strata (Loss): {avg_loss:.4f}, Celność (Acc): {acc:.4f}')

    # 4. Zapisywanie modelu
    torch.save(model.state_dict(), 'star_gnn_best.pth')
    print("Trening zakończony. Model zapisany jako star_gnn_best.pth")
    return model


def create_training_data(json_dir, batch_size=32):
    """
    Konwertuje pliki JSON na obiekty Data (PyTorch Geometric) i tworzy DataLoader.
    """
    all_data_list = []
    all_source_ids = []

    # 1. Zbieramy wszystkie JSON-y i unikalne Gaia IDs do Label Encodingu
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    print(f"Wczytywanie {len(json_files)} plików etykiet...")

    for j_file in json_files:
        with open(os.path.join(json_dir, j_file), 'r') as f:
            content = json.load(f)
            for obj in content['objects']:
                all_source_ids.append(obj['source_id'])

    # Label Encoder zamienia Gaia ID (np. 1941019...) na klasy (0, 1, 2...)
    le = LabelEncoder()
    le.fit(all_source_ids)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("✅ Słownik nazw (LabelEncoder) został zapisany pomyślnie.")

    num_classes = len(le.classes_)
    print(f"Zidentyfikowano {num_classes} unikalnych gwiazd (klas).")

    # 2. Budujemy grafy dla każdego zdjęcia
    for j_file in json_files:
        with open(os.path.join(json_dir, j_file), 'r') as f:
            content = json.load(f)

        h, w = content['image_shape']
        objs = content['objects']

        if len(objs) < 3: continue  # Potrzebujemy min. 3 punktów do triangulacji

        # Wyciągamy x, y i normalizujemy do [0, 1]
        coords = np.array([[o['x'] / w, o['y'] / h] for o in objs])

        # Tworzymy krawędzie za pomocą triangulacji Delaunaya
        tri = Delaunay(coords)
        edges = []
        for s in tri.simplices:
            edges.extend([[s[0], s[1]], [s[1], s[2]], [s[2], s[0]]])

        # Konwersja na format PyTorcha
        x = torch.tensor(coords, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Kodujemy etykiety (source_id -> int class)
        y = torch.tensor(le.transform([o['source_id'] for o in objs]), dtype=torch.long)

        # Tworzymy obiekt grafu
        data = Data(x=x, edge_index=edge_index, y=y)
        all_data_list.append(data)

    # 3. Tworzymy DataLoader
    loader = DataLoader(all_data_list, batch_size=batch_size, shuffle=True)

    return loader, le, num_classes


class StarGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.3):
        super(StarGAT, self).__init__()

        # Pierwsza warstwa GAT (Multi-head Attention)
        # in_channels: 2 (x, y)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * heads)

        # Druga warstwa GAT (Hidden Layer)
        # Wejście to hidden_channels * heads, ponieważ konkatenujemy wyniki z głów
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * heads)

        # Trzecia warstwa GAT (Output Layer)
        # Dla klasyfikacji węzłów często ustawiamy concat=False w ostatniej warstwie
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # Pierwszy blok: Konwolucja -> Batch Norm -> Aktywacja -> Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)  # ELU jest standardem dla GAT
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Drugi blok
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Trzecia warstwa (wyjściowa)
        x = self.conv3(x, edge_index)

        # Zwracamy logity (użyjemy CrossEntropyLoss podczas treningu)
        return x


def build_star_graph(points, source_ids=None):
    """
    Zamienia punkty (x, y) na graf. Krawędzie tworzone są za pomocą triangulacji Delaunaya.
    """
    coords = torch.tensor(points, dtype=torch.float)

    # 1. Tworzenie krawędzi (Triangulacja Delaunaya łączy najbliższych sąsiadów)
    tri = Delaunay(points)
    edges = []
    for simplex in tri.simplices:
        edges.extend([[simplex[0], simplex[1]], [simplex[1], simplex[2]], [simplex[2], simplex[0]]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # 2. Cechy węzłów (np. znormalizowane x, y lub jasność jeśli ją masz)
    # W Big Data często dodaje się tu deskryptory lokalnej gęstości
    node_features = coords

    # 3. Etykiety (source_id z katalogu Gaia - do treningu)
    y = None
    if source_ids:
        y = torch.tensor(source_ids, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=y)

# detected_points = [
#     [559.88,  31.87],
#     [1205.6, 989.63],
#     [942.03, 84.174],
#     [1680.2, 1353.2],
#     [1908.9, 719.62],
#     [  1489, 902.81],
#     [387.24, 29.155],
#     [559.71, 1286.1],
#     [1263.7, 1346.2],
#     [785.26, 813.15],
# ]
#
# source_ids = [
#     2880159340182235264,
#     2880160882074990848,
#     2880159615060138368,
#     2880159340182235264,
#     2880160882074990848,
#     2880161088233420928,
#     2880160852010722432,
#     2880159340182235264,
#     2880160882074990848,
#     2880160817650983808,
# ]


if __name__ == "__main__":
    # 1. Przygotowanie danych i ENCODERA
    # Ta funkcja wczyta JSONy, zrobi fit() na LabelEncoderze i zapisze label_encoder.pkl
    loader, le, num_stars = create_training_data("dataset/labels")

    # 2. Trening modelu GNN
    # Przekazujemy num_stars, żeby sieć wiedziała, ile ma klas na wyjściu
    model_gnn = train_gnn_model(loader, num_stars)

    # 3. Wczytaj YOLO (gdzie są kropki?)
    yolo_model = YOLO(PATH_YOLO)

    # 4. Wczytaj ENCODER (co oznaczają numery klas?)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # 5. Wczytaj GNN (jaka to gwiazda?)
    num_classes = len(le.classes_)
    gnn_model = StarGAT(in_channels=2, hidden_channels=64, out_channels=num_classes)
    gnn_model.load_state_dict(torch.load(PATH_GNN))

    # 4. Wywołaj funkcję predict_stars(), którą napisaliśmy wcześniej
    predict_stars()

