import os
import glob
import random
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Balance dataset by augmenting underrepresented classes"
    )
    parser.add_argument(
        "--data-dir", type=str, default="datasets/mask", help="Path to data directory"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
        help="Target number of samples for each class",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def read_yolo_labels(label_path):
    """Read YOLO format labels from a file"""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                boxes.append([cls_id, x_center, y_center, width, height])
    return np.array(boxes)


def write_yolo_labels(label_path, boxes):
    """Write YOLO format labels to a file"""
    with open(label_path, "w") as f:
        for box in boxes:
            f.write(f"{int(box[0])} {box[1]} {box[2]} {box[3]} {box[4]}\n")


def yolo_to_xyxy(box, img_width, img_height):
    """Convert YOLO format (cls, x_center, y_center, width, height) to (cls, x1, y1, x2, y2)"""
    cls_id = box[0]
    x_center = box[1] * img_width
    y_center = box[2] * img_height
    width = box[3] * img_width
    height = box[4] * img_height

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    return [cls_id, x1, y1, x2, y2]


def xyxy_to_yolo(box, img_width, img_height):
    """Convert (cls, x1, y1, x2, y2) to YOLO format (cls, x_center, y_center, width, height)"""
    cls_id = box[0]
    x1, y1, x2, y2 = box[1], box[2], box[3], box[4]

    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return [cls_id, x_center, y_center, width, height]


def augment_image_and_labels(
    image_path, label_path, output_img_path, output_label_path, aug_transforms
):
    """Apply augmentation to both image and bounding boxes"""
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]

    # Read labels
    boxes = read_yolo_labels(label_path)

    # Convert YOLO format to xyxy for albumentations
    bboxes = []
    for box in boxes:
        xyxy_box = yolo_to_xyxy(box, width, height)
        bboxes.append([xyxy_box[1], xyxy_box[2], xyxy_box[3], xyxy_box[4], xyxy_box[0]])

    # Apply augmentation
    try:
        transformed = aug_transforms(image=img, bboxes=bboxes)
        augmented_img = transformed["image"]
        augmented_bboxes = transformed["bboxes"]

        # Convert back to YOLO format
        augmented_boxes = []
        for bbox in augmented_bboxes:
            x1, y1, x2, y2, cls_id = bbox
            yolo_box = xyxy_to_yolo([cls_id, x1, y1, x2, y2], width, height)
            augmented_boxes.append(yolo_box)

        # Save augmented image
        augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_img_path, augmented_img)

        # Save augmented labels
        write_yolo_labels(output_label_path, augmented_boxes)
        return True
    except Exception as e:
        print(f"Error during augmentation: {e}")
        return False


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check if albumentations is installed
    try:
        import albumentations
    except ImportError:
        print("albumentations package is required for augmentation.")
        print("Install it with: pip install albumentations")
        return

    # Define paths
    train_dir = os.path.join(args.data_dir, "train")
    train_images_dir = os.path.join(train_dir, "images")
    train_labels_dir = os.path.join(train_dir, "labels")

    augmented_dir = os.path.join(args.data_dir, "augmented")
    augmented_images_dir = os.path.join(augmented_dir, "images")
    augmented_labels_dir = os.path.join(augmented_dir, "labels")

    # Create augmented directories
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_labels_dir, exist_ok=True)

    # Get all label files
    label_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))

    # Count samples per class
    class_counts = {
        0: 0,
        1: 0,
        2: 0,
    }  # 0: with_mask, 1: without_mask, 2: mask_weared_incorrect
    class_samples = {0: [], 1: [], 2: []}

    print("Analyzing dataset...")
    for label_file in tqdm(label_files):
        basename = os.path.basename(label_file)
        image_file = os.path.join(
            train_images_dir, os.path.splitext(basename)[0] + ".png"
        )

        # Check if image exists
        if not os.path.exists(image_file):
            continue

        # Read labels and count classes
        boxes = read_yolo_labels(label_file)

        # Add file to corresponding class lists
        for box in boxes:
            cls_id = int(box[0])
            if cls_id in class_counts:
                class_counts[cls_id] += 1
                class_samples[cls_id].append((image_file, label_file))

    print("\nInitial class distribution:")
    print(f"Class 0 (with_mask): {class_counts[0]} samples")
    print(f"Class 1 (without_mask): {class_counts[1]} samples")
    print(f"Class 2 (mask_weared_incorrect): {class_counts[2]} samples")

    # Define augmentation transforms
    aug_transforms = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RGBShift(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussNoise(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # Augment underrepresented classes
    for cls_id in [0, 1, 2]:
        if class_counts[cls_id] < args.target_count:
            needed_samples = args.target_count - class_counts[cls_id]
            print(
                f"\nAugmenting class {cls_id} to generate {needed_samples} additional samples"
            )

            # If we don't have any samples of this class, skip
            if len(class_samples[cls_id]) == 0:
                print(f"No samples found for class {cls_id}, skipping")
                continue

            # Select random samples to augment (with replacement if needed)
            samples_to_augment = random.choices(class_samples[cls_id], k=needed_samples)

            # Apply augmentation
            augmented_count = 0
            for i, (image_file, label_file) in enumerate(tqdm(samples_to_augment)):
                basename = os.path.basename(image_file)
                basename_no_ext = os.path.splitext(basename)[0]

                output_img_path = os.path.join(
                    augmented_images_dir, f"{basename_no_ext}_aug_{i}.png"
                )
                output_label_path = os.path.join(
                    augmented_labels_dir, f"{basename_no_ext}_aug_{i}.txt"
                )

                success = augment_image_and_labels(
                    image_file,
                    label_file,
                    output_img_path,
                    output_label_path,
                    aug_transforms,
                )
                if success:
                    augmented_count += 1

            print(f"Successfully augmented {augmented_count} images for class {cls_id}")

    # Copy augmented data to training set
    print("\nCopying augmented data to training set...")
    augmented_images = glob.glob(os.path.join(augmented_images_dir, "*.png"))

    for image_path in tqdm(augmented_images):
        basename = os.path.basename(image_path)
        label_path = os.path.join(
            augmented_labels_dir, os.path.splitext(basename)[0] + ".txt"
        )

        if os.path.exists(label_path):
            # Copy to training set
            shutil.copy(image_path, os.path.join(train_images_dir, basename))
            shutil.copy(
                label_path, os.path.join(train_labels_dir, os.path.basename(label_path))
            )

    # Count final distribution
    new_class_counts = {0: 0, 1: 0, 2: 0}
    all_label_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))

    for label_file in all_label_files:
        boxes = read_yolo_labels(label_file)
        for box in boxes:
            cls_id = int(box[0])
            if cls_id in new_class_counts:
                new_class_counts[cls_id] += 1

    print("\nFinal class distribution after augmentation:")
    print(f"Class 0 (with_mask): {new_class_counts[0]} samples")
    print(f"Class 1 (without_mask): {new_class_counts[1]} samples")
    print(f"Class 2 (mask_weared_incorrect): {new_class_counts[2]} samples")

    print("\nData balancing complete!")
    print("To train with the balanced dataset, run:")
    print("cd yolov5")
    print(
        "python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0"
    )


if __name__ == "__main__":
    main()
