from pathlib import Path
import cv2
import shutil

root = Path.home() / "Downloads/MarsData-MarsData-V2"
output = Path("dataset_v2")

total_boxes = 0

for split in ["train", "val", "test"]:
    image_dir = root / "img" / split / split
    mask_dir = root / "label" / split / split

    images = sorted(image_dir.glob("*.png"))
    count = 0
    split_boxes = 0

    for image_path in images:
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_dir / image_path.name))

        if image is None or mask is None:
            continue

        h, w = image.shape[:2]

        # Identify yellow pixels representing rocks
        hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
        binary = cv2.inRange(
            hsv, (15, 70, 70), (45, 255, 255)
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        labels = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)

            # Ignore very small regions
            if bw * bh < 0.002 * w * h:
                continue

            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h

            labels.append(
                f"0 {xc:.6f} {yc:.6f} "
                f"{bw/w:.6f} {bh/h:.6f}"
            )

        dest_img = output / "images" / split
        dest_lbl = output / "labels" / split

        dest_img.mkdir(parents=True, exist_ok=True)
        dest_lbl.mkdir(parents=True, exist_ok=True)

        shutil.copy2(image_path, dest_img / image_path.name)

        (dest_lbl / f"{image_path.stem}.txt").write_text(
            "\n".join(labels)
        )

        count += 1
        split_boxes += len(labels)

    total_boxes += split_boxes
    print(f"{split}: {count} images, {split_boxes} rock boxes")

config = output.resolve() / "data.yaml"

config.write_text(
    f"path: {output.resolve()}\n"
    "train: images/train\n"
    "val: images/val\n"
    "test: images/test\n"
    "names:\n"
    "  0: rock\n"
)

print("\nDataset prepared successfully!")
print("Total rock boxes:", total_boxes)
