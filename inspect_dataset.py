from pathlib import Path
import cv2
import numpy as np

root = Path.home() / "Downloads/MarsData-MarsData-V2"

image_dir = root / "img/train/train"
mask_dir = root / "label/train/train"

# Select one original photograph
image_path = sorted(image_dir.glob("*_raw.png"))[0]
mask_path = mask_dir / image_path.name

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path))

if image is None or mask is None:
    raise RuntimeError("Could not load image or mask.")

# Identify yellow rock regions
hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
binary = cv2.inRange(hsv, (15, 70, 70), (45, 255, 255))

# Overlay rock regions in red
overlay = image.copy()
overlay[binary > 0] = (0, 0, 255)

combined = np.hstack([
    image,
    mask,
    overlay
])

output = Path.home() / "Desktop/mars_labels_inspection.png"
cv2.imwrite(str(output), combined)

print("Saved:", output)
print("Unique mask colours:", len(np.unique(mask.reshape(-1, 3), axis=0)))
print("Rock pixel percentage:",
      round(100 * np.mean(binary > 0), 2))
