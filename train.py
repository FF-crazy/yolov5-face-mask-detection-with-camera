#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train YOLOv5 model for face mask detection.

This script is a wrapper around the YOLOv5 train.py script, making it easier
to train the face mask detection model with the appropriate parameters.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv5 model for face mask detection")
    parser.add_argument(
        "--img", type=int, default=640, help="Image size for training (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs (default: 100)"
    )
    parser.add_argument(
        "--data", type=str, default="../mask_config.yaml", help="Path to data config file (default: ../mask_config.yaml)"
    )
    parser.add_argument(
        "--weights", type=str, default="yolov5s.pt", help="Initial weights path (default: yolov5s.pt)"
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Number of worker threads (default: 0)"
    )
    parser.add_argument(
        "--device", type=str, default="", help="Device to use (cuda device, i.e. 0 or 0,1,2,3 or cpu)"
    )
    parser.add_argument(
        "--project", type=str, default="runs/train", help="Save to project/name (default: runs/train)"
    )
    parser.add_argument(
        "--name", type=str, default="exp", help="Save to project/name (default: exp)"
    )
    parser.add_argument(
        "--exist-ok", action="store_true", help="Existing project/name ok, do not increment"
    )
    parser.add_argument(
        "--cache", action="store_true", help="Cache images for faster training"
    )
    parser.add_argument(
        "--no-prepare", action="store_true", help="Skip running prepare.py before training"
    )
    return parser.parse_args()


def main():
    """Main function to run the training process."""
    args = parse_args()

    # Check if YOLOv5 repository exists
    if not os.path.exists("yolov5"):
        print("YOLOv5 repository not found. Cloning it now...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/ultralytics/yolov5"],
                check=True
            )
            print("YOLOv5 repository cloned successfully.")

            # Install YOLOv5 dependencies
            print("Installing YOLOv5 dependencies...")
            subprocess.run(
                ["pip", "install", "-r", "yolov5/requirements.txt"],
                check=True
            )
            print("YOLOv5 dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to clone YOLOv5 repository or install dependencies: {e}")
            sys.exit(1)

    # Run prepare.py if not skipped
    if not args.no_prepare:
        print("Preparing dataset...")
        try:
            subprocess.run(["python", "prepare.py"], check=True)
            print("Dataset preparation complete.")
        except subprocess.CalledProcessError as e:
            print(f"Error: Failed to prepare dataset: {e}")
            sys.exit(1)

    # Change directory to YOLOv5
    os.chdir("yolov5")

    # Construct the training command
    cmd = [
        "python", "train.py",
        "--img", str(args.img),
        "--batch", str(args.batch),
        "--epochs", str(args.epochs),
        "--data", args.data,
        "--weights", args.weights,
        "--workers", str(args.workers),
    ]

    # Add optional arguments if specified
    if args.device:
        cmd.extend(["--device", args.device])
    if args.project:
        cmd.extend(["--project", args.project])
    if args.name:
        cmd.extend(["--name", args.name])
    if args.exist_ok:
        cmd.append("--exist-ok")
    if args.cache:
        cmd.append("--cache")

    # Run the training command
    print("Starting training with the following command:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Training failed: {e}")
        sys.exit(1)

    # Return to original directory
    os.chdir("..")

    print("Training process completed. The trained model is saved in yolov5/runs/train/")
    print("You can use the model for detection with:")
    print("  python webcam.py --model yolov5/runs/train/exp/weights/best.pt")


if __name__ == "__main__":
    main()
