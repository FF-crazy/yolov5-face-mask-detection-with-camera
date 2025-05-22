# Face Mask Detection - Setup Guide

This document provides detailed instructions for setting up the Face Mask Detection project.

## System Requirements

- Python 3.6 or higher
- pip (Python package installer)
- Git
- Camera/webcam for real-time detection
- Recommended: CUDA-compatible GPU for faster training and inference

## Quick Setup

For a quick automated setup, run:

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The script will:

1. Create a virtual environment (recommended)
2. Install all required dependencies
3. Clone the YOLOv5 repository if needed
4. Check for the model and dataset files

## Manual Setup

If you prefer to set up manually or if the automated script doesn't work for you, follow these steps:

### 1. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install project dependencies
pip install -r requirements.txt

# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

### 3. Download the Dataset

For training your own model, download the Face Mask Detection dataset:

1. Go to [Kaggle Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)
2. Download the dataset zip file
3. Place the downloaded `archive.zip` file in the `datasets` directory

### 4. Prepare the Dataset

```bash
# Create necessary directories
mkdir -p datasets

# Run the preparation script
python prepare.py
```

This will:

- Extract the dataset from the ZIP file
- Convert annotations to YOLO format
- Split the dataset into training and validation sets

### 5. Train Your Model (Optional)

If you want to train your own model:

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0
cd ..
```

Training parameters:

- `--img 640`: Set input image size to 640x640 pixels
- `--batch 16`: Batch size (reduce if you encounter memory issues)
- `--epochs 100`: Number of training epochs
- `--workers 0`: Number of worker threads (0 is recommended for Windows)

### 6. Run the Face Mask Detection

For real-time detection using your webcam:

```bash
python webcam.py
```

Additional options:

```bash
python webcam.py --help  # Show available options
python webcam.py --model path/to/model.pt  # Use a custom model
python webcam.py --conf-thres 0.6  # Set custom confidence threshold
python webcam.py --camera 1  # Use a different camera (if you have multiple)
```

## Troubleshooting

### Common Issues

1. **Model not found error**:
   - Ensure the model file exists in the `models` directory
   - If training your own model, check that the training completed successfully

2. **Camera access issues**:
   - Ensure your webcam is properly connected
   - Try using a different camera ID with `--camera` flag
   - Check if another application is using the camera

3. **CUDA/GPU issues**:
   - Verify your CUDA installation with `nvidia-smi`
   - Ensure you have compatible PyTorch version for your CUDA version

4. **Performance is slow**:
   - If using CPU, expect lower FPS
   - Reduce resolution with `--width` and `--height` flags
   - Consider using a smaller YOLOv5 model variant

## Additional Resources

- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [Face Mask Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection)
- [OpenCV Documentation](https://docs.opencv.org/)
