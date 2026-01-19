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
from ultralytics import YOLO
from scipy.spatial import Delaunay
import pickle

# --- CONFIGURATION ---
PATH_YOLO = "star_detection_project/yolo11s_stars_v2/best.pt"
PATH_GNN = "star_gnn_best.pth"
PATH_ENCODER = "label_encoder.pkl"  # Saved LabelEncoder
CONFIDENCE_THRESHOLD = 0.01
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# --- EXAMPLES ---
IMAGE_PATH = "yolo_dataset/train/images/run4822_cam6_frame535_r.png"


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


if __name__ == "__main__":
    # Preparing the data and the encoder
    # fit() on LabelEncoder and save in label_encoder.pkl
    loader, le, num_stars = create_training_data("dataset/labels")

    # Training GNN model
    model_gnn = train_gnn_model(loader, num_stars)

