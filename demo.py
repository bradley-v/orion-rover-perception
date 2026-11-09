from pathlib import Path
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/training_v2/rocks/weights/best.pt")

# Create demonstration folder on Desktop
output = Path.home() / "Desktop/ORION_Demo"
output.mkdir(exist_ok=True)

# Select 5 original images from the held-out test set
images = sorted(
    Path("dataset_v2/images/test").glob("*_raw.png")
)[:5]

print("Generating rock detection examples...")

for i, image in enumerate(images):
    result = model.predict(
        str(image),
        conf=0.30,
        iou=0.40,
        verbose=False
    )[0]

    annotated = result.plot()
    cv2.imwrite(
        str(output / f"rock_detection_{i+1}.jpg"),
        annotated
    )

    print(f"Image {i+1}: {len(result.boxes)} rocks detected")

# Evaluate on held-out test set
print("\nEvaluating model...")
metrics = model.val(
    data="dataset_v2/data.yaml",
    split="test",
    imgsz=320,
    device="mps",
    project=str(output),
    name="evaluation",
    exist_ok=True
)

summary = f"""OXFORD ORION — ROCK DETECTION PROTOTYPE

Model: YOLO11n
Task: Detect rocks in Martian terrain photographs
Dataset: MarsData-V2
Training: 100 images, 15 epochs
Validation: 100 images
Testing: 100 images

HELD-OUT TEST RESULTS
Precision: {metrics.box.mp:.3f}
Recall: {metrics.box.mr:.3f}
mAP@50: {metrics.box.map50:.3f}

Limitations:
- Only 10 original photographs per dataset split.
- Augmentations increase image count but not scene diversity.
- Segmentation masks converted into approximate bounding boxes.
- Detecting rocks does not establish whether terrain is traversable.

Next steps:
1. Improve annotation quality.
2. Investigate semantic segmentation.
3. Test on ERC Mars Yard imagery.
"""

(output / "results.txt").write_text(summary)

print("\nDEMO COMPLETE")
print("Saved to:", output)
