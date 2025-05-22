import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join, exists
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import zipfile
from pathlib import Path
import numpy as np
import shutil
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare face mask detection dataset")
    parser.add_argument(
        "--data-dir", type=str, default="datasets", help="Path to data directory"
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="archive.zip",
        help="Name of the zip file containing data",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (0.0-1.0)",
    )
    parser.add_argument(
        "--download-url",
        type=str,
        default="",
        help="URL to download the dataset if not found",
    )
    return parser.parse_args()


def convert(size, box):
    """Convert VOC bbox coords to YOLO format"""
    dw = 1.0 / (size[0])
    dh = 1.0 / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(classes, input_path, output_path):
    """Convert VOC XML annotation to YOLO txt format"""
    try:
        basename = os.path.basename(input_path)
        basename_no_ext = os.path.splitext(basename)[0]

        in_file = open(input_path)
        out_file = open(output_path + "/" + basename_no_ext + ".txt", "w")

        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        for obj in root.iter("object"):
            difficult = obj.find("difficult").text
            cls = obj.find("name").text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")

        in_file.close()
        out_file.close()
        return True
    except Exception as e:
        print(f"Error converting annotation {input_path}: {e}")
        return False


def download_dataset(url, save_path):
    """Download dataset if not available locally"""
    if not url:
        print("No download URL provided. Please download the dataset manually.")
        print(
            "You can get it from: https://www.kaggle.com/andrewmvd/face-mask-detection"
        )
        return False

    try:
        import requests

        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            for data in tqdm(
                response.iter_content(chunk_size=1024),
                total=total_size // 1024,
                unit="KB",
            ):
                f.write(data)
        print(f"Download complete: {save_path}")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def main():
    # Parse arguments
    args = parse_args()

    # Define paths
    classes = ["with_mask", "without_mask", "mask_weared_incorrect"]
    ROOT_DIR = args.data_dir
    DATA_FILE = args.data_file
    IMAGE_EXT = ".png"

    DATA_DIR = join(ROOT_DIR, "mask")
    IMAGE_DIR = join(DATA_DIR, "images")
    LABEL_DIR = join(DATA_DIR, "annotations")
    PROCESSED_LABEL_DIR = join(DATA_DIR, "processed_annotations")
    TRAIN_DATA_DIR = join(DATA_DIR, "train")
    VALID_DATA_DIR = join(DATA_DIR, "valid")

    # Create root directory if it doesn't exist
    os.makedirs(ROOT_DIR, exist_ok=True)

    # Check if data file exists, if not try to download it
    data_file_path = join(ROOT_DIR, DATA_FILE)
    if not exists(data_file_path):
        print(f"Data file not found: {data_file_path}")
        if not download_dataset(args.download_url, data_file_path):
            sys.exit(1)

    try:
        # Extract zip file
        print(f"Extracting {data_file_path} to {DATA_DIR}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        with zipfile.ZipFile(data_file_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Extraction complete")

        # Create necessary directories
        for directory in [
            TRAIN_DATA_DIR,
            VALID_DATA_DIR,
            join(TRAIN_DATA_DIR, "images"),
            join(TRAIN_DATA_DIR, "labels"),
            join(VALID_DATA_DIR, "images"),
            join(VALID_DATA_DIR, "labels"),
            PROCESSED_LABEL_DIR,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Convert annotations
        print("Converting annotations from VOC to YOLO format...")
        image_files = glob.glob(IMAGE_DIR + "/*" + IMAGE_EXT)

        if not image_files:
            print(f"No image files found in {IMAGE_DIR}. Check your dataset structure.")
            sys.exit(1)

        conversion_results = []
        for image_file in tqdm(image_files):
            basename = os.path.basename(image_file)
            basename_no_ext = os.path.splitext(basename)[0]
            annotation_path = join(LABEL_DIR, basename_no_ext + ".xml")

            if not exists(annotation_path):
                print(f"Warning: Annotation file not found for {basename}")
                continue

            result = convert_annotation(classes, annotation_path, PROCESSED_LABEL_DIR)
            conversion_results.append(result)

        print(f"Converted {sum(conversion_results)} annotations successfully")

        # Split dataset
        print(f"Splitting dataset with validation ratio: {args.split_ratio}")
        train_images, valid_images = train_test_split(
            image_files, test_size=args.split_ratio, random_state=42
        )

        print(
            f"Training images: {len(train_images)}, Validation images: {len(valid_images)}"
        )

        # Copy files to train/valid directories
        print("Copying files to train/valid directories...")
        for image_set, target_dir in [
            (train_images, TRAIN_DATA_DIR),
            (valid_images, VALID_DATA_DIR),
        ]:
            for image_path in tqdm(image_set):
                basename = os.path.basename(image_path)
                basename_no_ext = os.path.splitext(basename)[0]

                # Copy image
                shutil.copy(image_path, join(target_dir, "images", basename))

                # Copy label if exists
                label_path = join(PROCESSED_LABEL_DIR, basename_no_ext + ".txt")
                if exists(label_path):
                    shutil.copy(
                        label_path, join(target_dir, "labels", basename_no_ext + ".txt")
                    )
                else:
                    print(f"Warning: Label file not found for {basename}")

        print("Dataset preparation complete!")
        print(f"Train images: {len(train_images)}")
        print(f"Valid images: {len(valid_images)}")
        print(f"Dataset structure:")
        print(f"  {DATA_DIR}/")
        print(f"  ├── train/")
        print(f"  │   ├── images/ ({len(train_images)} files)")
        print(
            f"  │   └── labels/ ({len(glob.glob(TRAIN_DATA_DIR + '/labels/*.txt'))} files)"
        )
        print(f"  └── valid/")
        print(f"      ├── images/ ({len(valid_images)} files)")
        print(
            f"      └── labels/ ({len(glob.glob(VALID_DATA_DIR + '/labels/*.txt'))} files)"
        )

    except Exception as e:
        print(f"Error during dataset preparation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
