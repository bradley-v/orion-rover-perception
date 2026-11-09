from pathlib import Path
import random
import shutil
import cv2
from ultralytics import YOLO

# Dataset locations
root = Path.home() / "Downloads/MarsData-MarsData-V2"
output = Path("dataset")
random.seed(42)

# Read Mars images and their segmentation masks
image_dir = root / "img/train/train"
mask_dir = root / "label/train/train"

files = sorted(image_dir.glob("*.png"))
random.shuffle(files)
files = files[:100]

if not files:
    raise RuntimeError("No Mars images found. Check dataset path.")

counts = {"train": 0, "val": 0}
total_boxes = 0

for i, file in enumerate(files):
    subset = "train" if i < 80 else "val"

    image = cv2.imread(str(file))
    mask = cv2.imread(str(mask_dir / file.name))

    if image is None or mask is None:
        continue

    h, w = image.shape[:2]

    # Identify yellow rock regions
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv, (15, 70, 70), (45, 255, 255))

    # Convert regions to bounding boxes
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    labels = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        # Ignore very small rocks
        if bw * bh < 0.002 * w * h:
            continue

        labels.append(
            f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} "
            f"{bw/w:.6f} {bh/h:.6f}"
        )

    dest_img = output / "images" / subset
    dest_lbl = output / "labels" / subset
    dest_img.mkdir(parents=True, exist_ok=True)
    dest_lbl.mkdir(parents=True, exist_ok=True)

    shutil.copy2(file, dest_img / file.name)
    (dest_lbl / f"{file.stem}.txt").write_text("\n".join(labels))

    counts[subset] += 1
    total_boxes += len(labels)

print("Images prepared:", counts)
print("Rock boxes generated:", total_boxes, flush=True)

if min(counts.values()) == 0 or total_boxes == 0:
    raise RuntimeError("Dataset preparation failed: check mask colours.")

# Configure YOLO dataset
config = output.resolve() / "data.yaml"
config.write_text(
    f"path: {output.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n  0: rock\n"
)

# Train a small pretrained model
print("Starting training...", flush=True)
model = YOLO("yolo11n.pt")

model.train(
    data=str(config),
    epochs=5,
    imgsz=320,
    batch=8,
    device="cpu",
    workers=0,
    project="training",
    name="rocks",
    exist_ok=True
)

# Load trained weights
trained = YOLO("runs/detect/training/rocks/weights/best.pt")

# Test on a validation image
test_image = sorted((output / "images/val").glob("*.png"))[0]

results = trained.predict(
    source=str(test_image),
    conf=0.10,
    verbose=False
)

# Save the annotated result to Desktop
destination = Path.home() / "Desktop/rock_prediction.jpg"
cv2.imwrite(str(destination), results[0].plot())

print("\nSUCCESS!")
print("Prediction saved to:", destination)
print("Rocks detected:", len(results[0].boxes))
