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
from scipy.spatial import Delaunay, KDTree
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
    # Getting the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting GNN training on: {device}")

    # Initialization
    # in_channels=2 (xy), hidden_channels=64, out_channels=num_classes
    model = StarGAT(in_channels=2, hidden_channels=64, out_channels=num_classes).to(device)

    # Optimizer and loss function definition
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
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

            # Loss count (comparison of predicted classes with y)
            loss = criterion(out, data.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy count
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total_nodes += data.num_nodes

        # Results logging
        avg_loss = total_loss / len(loader)
        acc = correct / total_nodes

        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}')

    # Saving
    torch.save(model.state_dict(), PATH_GNN)
    print(f"Training ended. Model saved at {PATH_GNN}")
    return model


def create_training_data(json_dir, train_dir, batch_size=32):
    """
    Converts JSON files to Data objects (PyTorch Geometric) and creates DataLoader.
    """
    all_data_list = []
    all_source_ids = []
    json_files = []

    # Fetching all training JSONs and unique Gaia IDs to Label Encoding
    json_files_unfiltered = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    train_labels = [f.removesuffix(".txt") for f in os.listdir(train_dir) if f.endswith('.txt')]

    for f in json_files_unfiltered:
        if f.removesuffix(".json") in train_labels:
            json_files.append(f)
    print(f"Loading {len(json_files)} label files...")

    for j_file in json_files:
        with open(os.path.join(json_dir, j_file), 'r') as f:
            content = json.load(f)
            for obj in content['objects']:
                all_source_ids.append(obj['source_id'])

    # Label Encoder changes Gaia ID (e.g. 1941019...) to classes (0, 1, 2...)
    le = LabelEncoder()
    le.fit(all_source_ids)

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("LabelEncoder was saved successfully.")

    num_classes = len(le.classes_)
    print(f"{num_classes} unique stars (classes) identified.")

    # Building graphs for each photo
    for j_file in json_files:
        with open(os.path.join(json_dir, j_file), 'r') as f:
            content = json.load(f)

        h, w = content['image_shape']
        objs = content['objects']

        if len(objs) < 3: continue  # We need min. 3 points for triangulation

        # Taking xy and normalizing to [0, 1]
        coords = np.array([[o['x'] / w, o['y'] / h] for o in objs])

        # Creating edges with Delaunay triangulation
        tri = Delaunay(coords)
        edges = []
        for s in tri.simplices:
            edges.extend([[s[0], s[1]], [s[1], s[2]], [s[2], s[0]]])

        # PyTorch format conversion
        x = torch.tensor(coords, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Label encoding (source_id -> int class)
        y = torch.tensor(le.transform([o['source_id'] for o in objs]), dtype=torch.long)

        # Graph object creation
        data = Data(x=x, edge_index=edge_index, y=y)
        all_data_list.append(data)

    # DataLoader creation
    loader = DataLoader(all_data_list, batch_size=batch_size, shuffle=True)

    return loader, le, num_classes


class StarGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.3):
        super(StarGAT, self).__init__()

        # First GAT layer (Multi-head Attention)
        # in_channels: 2 (xy)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * heads)

        # Second GAT layer (Hidden Layer)
        # Entrance to hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * heads)

        # Third GAT layer (Output Layer)
        # concat=False in the last layer
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # 1. Block: Convolution -> Batch Norm -> Activation -> Dropout
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)  # ELU - standard for GAT
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Block
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Block (out)
        x = self.conv3(x, edge_index)

        # Return of logity
        return x


def build_star_graph(points, source_ids=None):
    """
    Changes points (xy) to a graph. Edges are created with Delaunay triangulation.
    """
    coords = torch.tensor(points, dtype=torch.float)

    # Edge creation (Delaunay triangulation connects the closest neighbors)
    tri = Delaunay(points)
    edges = []
    for simplex in tri.simplices:
        edges.extend([[simplex[0], simplex[1]], [simplex[1], simplex[2]], [simplex[2], simplex[0]]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Spatial Density Descriptors (using KDTree)
    tree = KDTree(points)

    # Descriptor A: Local Count Density (number of stars within 50px radius)
    density_counts = tree.query_ball_point(points, r=50.0, return_length=True)
    density_counts = torch.tensor(density_counts, dtype=torch.float).view(-1, 1)

    # Descriptor B: Average Distance to 5-nearest neighbors
    # We use k=6 because the first result is always the point itself (distance 0)
    dist, _ = tree.query(points, k=6)
    avg_dist = torch.tensor(dist[:, 1:].mean(axis=1), dtype=torch.float).view(-1, 1)

    # Node characteristics
    # Combined features: [x, y, neighbor_count, mean_distance]
    node_features = torch.cat([coords, density_counts, avg_dist], dim=-1)

    # Labels (source_id from Gaia - for training)
    y = None
    if source_ids:
        y = torch.tensor(source_ids, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=y)


if __name__ == "__main__":
    # Preparing the data and the encoder
    # fit() on LabelEncoder and save in label_encoder.pkl
    loader, le, num_stars = create_training_data("dataset/labels", "yolo_dataset/train/labels")

    # Training GNN model
    model_gnn = train_gnn_model(loader, num_stars)

