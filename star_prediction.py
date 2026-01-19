from gnn_layer import create_training_data, train_gnn_model, StarGAT
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


# star prediction function
def predict_stars():
    # -----------------------------------------------
    #  1. LOADING THE MODELS
    # -----------------------------------------------

    # Defining the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # loading the YOLO model
    yolo_model = YOLO(PATH_YOLO)

    with open(PATH_ENCODER, 'rb') as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)

    # loading the GNN model
    gnn_model = StarGAT(in_channels=2, hidden_channels=64, out_channels=num_classes)
    gnn_model.load_state_dict(torch.load(PATH_GNN, map_location=device))
    gnn_model.to(device)
    gnn_model.eval()

    # -----------------------------------------------
    #  2. DATA RETRIEVAL
    # -----------------------------------------------

    # YOLO detection (retrieval of xy coordinates)
    img = cv2.imread(IMAGE_PATH)
    h_orig, w_orig = img.shape[:2]
    results = yolo_model.predict(img, imgsz=1280, conf=0.25)[0]

    # Getting boxes middles and normalization
    boxes = results.boxes.xywh.cpu().numpy()
    if len(boxes) < 3:
        print("Zbyt mało gwiazd do zbudowania grafu!")
        return

    coords = boxes[:, :2]  # x, y
    x_norm = coords[:, 0] / w_orig
    y_norm = coords[:, 1] / h_orig
    normalized_points = np.stack([x_norm, y_norm], axis=1)

    # Graph build for the picture
    tri = Delaunay(normalized_points)
    edges = []
    for s in tri.simplices:
        edges.extend([[s[0], s[1]], [s[1], s[2]], [s[2], s[0]]])

    x_tensor = torch.tensor(normalized_points, dtype=torch.float).to(device)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    # GNN Inference (identification)
    with torch.no_grad():
        out = gnn_model(x_tensor, edge_index)
        predictions = out.argmax(dim=1).cpu().numpy()
        confidence = torch.nn.functional.softmax(out, dim=1).max(dim=1).values.cpu().numpy()

    # -----------------------------------------------
    #  3. RESULTS VISUALIZATION
    # -----------------------------------------------

    # ID decoding and visualization
    gaia_ids = le.inverse_transform(predictions)

    print(f"\n{len(gaia_ids)} objects found:")
    for i, g_id in enumerate(gaia_ids):
        if confidence[i] > CONFIDENCE_THRESHOLD:  # Filtr pewności modelu
            print(f"Star {i}: Gaia ID {g_id} (Certainty: {confidence[i]:.2f})")

            # Sketching on the picture
            x, y = int(coords[i][0]), int(coords[i][1])
            cv2.circle(img, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(img, str(g_id)[-6:], (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Identified stars", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    # Preparing the data and the encoder
    # fit() on LabelEncoder and save in label_encoder.pkl
    loader, le, num_stars = create_training_data("dataset/labels")

    # Training GNN model
    model_gnn = train_gnn_model(loader, num_stars)

    # Loading YOLO
    yolo_model = YOLO(PATH_YOLO)

    # Loading encoder
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # Loading GNN
    num_classes = len(le.classes_)
    gnn_model = StarGAT(in_channels=2, hidden_channels=64, out_channels=num_classes)
    gnn_model.load_state_dict(torch.load(PATH_GNN))

    # Actual prediction process
    predict_stars()