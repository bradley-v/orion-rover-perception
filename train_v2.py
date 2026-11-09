from ultralytics import YOLO
from pathlib import Path
import torch

# Use Apple's GPU if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Training device:", device)

# Load pretrained YOLO11 nano model
model = YOLO("yolo11n.pt")

# Train using the corrected dataset
model.train(
    data="dataset_v2/data.yaml",
    epochs=15,
    imgsz=320,
    batch=8,
    device=device,
    workers=0,
    project="training_v2",
    name="rocks",
    exist_ok=True
)

print("\nTRAINING COMPLETE")
print("Model saved to:")
print(Path("training_v2/rocks/weights/best.pt").resolve())
