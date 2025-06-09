import cv2
import torch
import numpy as np
import time
import argparse
import os
from pathlib import Path
import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Face Mask Detection using YOLOv5")
    parser.add_argument(
        "--model",
        type=str,
        default="models/mask_yolov5.pt",
        help="Path to YOLOv5 model",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
    parser.add_argument(
        "--width", type=int, default=640, help="Width of the frames in the video stream"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of the frames in the video stream",
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Check if model file exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print(
            f"Please make sure the model file exists or specify a different path with --model"
        )
        return

    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, force_reload=True
        )
        model.to(device)
        model.eval()
        print("Model loaded successfully")

        # Load class names from mask_config.yaml
        config_path = Path(__file__).parent / "mask_config.yaml"
        if not config_path.exists():
            print(f"Error: mask_config.yaml not found at {config_path}")
            # Fallback to hardcoded classes if config not found, or handle error differently
            class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]
            print("Warning: Using hardcoded class names.")
        else:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            class_names = config["names"]
            print(f"Loaded class names: {class_names}")

        # Set fixed colors for known classes
        colors = [
            (0, 255, 0),   # with_mask
            (0, 0, 255),   # without_mask
            (0, 255, 255)  # mask worn incorrectly
        ][: len(class_names)]

        # Fallback to random colors if number of classes differs
        if len(colors) != len(class_names):
            colors = []
            np.random.seed(42)  # for consistent colors
            for _ in range(len(class_names)):
                colors.append(
                    tuple(np.random.randint(0, 255, size=3).tolist())
                )

        # Open webcam
        print(f"Opening camera device {args.camera}...")
        cap = cv2.VideoCapture(args.camera)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not cap.isOpened():
            print(f"Error: Could not open camera device {args.camera}")
            return

        print("Camera opened successfully. Press 'q' to quit.")

        # Initialize FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0

        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                # Try to reconnect
                print("Attempting to reconnect...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(args.camera)
                if not cap.isOpened():
                    print("Reconnection failed. Exiting.")
                    break
                continue

            try:
                # Process frame
                start_time = time.time()

                # Inference
                results = model(frame)

                # Process results
                for *xyxy, conf, cls in results.xyxy[0]:  # xyxy, confidence, class
                    if conf > args.conf_thres:  # confidence threshold
                        # Get coordinates and class
                        x1, y1, x2, y2 = (
                            int(xyxy[0]),
                            int(xyxy[1]),
                            int(xyxy[2]),
                            int(xyxy[3]),
                        )
                        class_idx = int(cls)
                        if class_idx < len(class_names):  # Safety check
                            label = f"{class_names[class_idx]} {conf:.2f}"

                            # Draw bounding box
                            color = colors[class_idx]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                            # Add label
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2,
                            )

                end_time = time.time()

                # Calculate FPS
                process_time = end_time - start_time
                fps_frame_count += 1

                # Update FPS every second
                if (end_time - fps_start_time) > 1.0:
                    fps = fps_frame_count / (end_time - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = end_time

                # Add FPS to frame
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Display the resulting frame
                cv2.imshow("Face Mask Detection", frame)

                # Break the loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception as e:
                print(f"Error during inference: {e}")
                continue

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
