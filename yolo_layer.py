from ultralytics import YOLO
import torch

# --- CONFIGURATION --- #
GOOGLE_DRIVE = False
GOOGLE_PROJECT_ADRESS = ""
YOLO_MODEL = "yolo11n.pt" # Choices: n - nano, s - small, m - medium
DATA_YAML_PATH = "data.yaml"


def train_star_detector():
    # Checking the GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Used evice: {device}")

    # Choice of the base model
    model = YOLO(YOLO_MODEL)

    # Training
    # Parametrization adjusted to small objects (stars)
    model.train(
        data=DATA_YAML_PATH,                    # Path to data.yaml file
        epochs=100,                             # Number of epochs
        imgsz=1280,                             # Picture size - crucial for star resolution
        batch=8,                                # Batch size (depends on VRAM possibilities)
        patience=10,                            # Early stopping: stop if {patience} epochs give no improvement
        save=True,                              # Save the model weights (important!)
        device=device,                          # Use of detected device
        workers=8,                              # Number of threads to operate on
        project="star_detection_project",       # Project name (will create this directory)
        name="yolo11s_stars_v1",                # Name of the experiment
        exist_ok=True,
        # Augmentations
        mosaic=0.5,                             # To not shred the constellations
        mixup=0.1                               # Helpful in dense star regions
    )

    print("Training finished!")

if __name__ == "__main__":
    train_star_detector()

