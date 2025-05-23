# Face Mask Detection - Setup Guide

This document provides detailed instructions for setting up the Face Mask Detection project on **Windows**, **Linux**, and **macOS**.

## System Requirements

- **Python 3.6 or higher** (3.8+ recommended)
- **pip** (Python package installer)
- **Git**
- **Camera/webcam** for real-time detection
- **Recommended**: CUDA-compatible GPU for faster training and inference

### Windows-Specific Requirements

- **Python**: Download from [python.org](https://www.python.org/downloads/) or use Microsoft Store
- **Git**: Download from [git-scm.com](https://git-scm.com/download/win) or install via GitHub Desktop
- **Visual Studio Build Tools** (if installing packages that require compilation)

## Quick Setup

### Windows

Open Command Prompt or PowerShell and run:

```cmd
setup.bat
```

### Linux/macOS

Open Terminal and run:

```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

The setup scripts will:

1. Check for Python installation
2. Create a virtual environment (recommended)
3. Install all required dependencies
4. Clone the YOLOv5 repository if needed
5. Check for the model and dataset files

## Manual Setup

If you prefer to set up manually or if the automated script doesn't work, follow these platform-specific steps:

### 1. Create a Virtual Environment (Recommended)

**Windows (Command Prompt/PowerShell):**

```cmd
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\activate
```

**Windows (Git Bash):**

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/Scripts/activate
```

**Linux/macOS:**

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies

**All platforms:**

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

### Windows-Specific Troubleshooting

5. **Python command not found**:
   - Ensure Python is added to your PATH during installation
   - Try using `py` instead of `python` command
   - Reinstall Python from [python.org](https://www.python.org/downloads/) with "Add to PATH" checked

6. **Virtual environment activation fails**:
   - Use `python -m venv .venv` instead of `python3 -m venv .venv`
   - Try running Command Prompt as Administrator
   - Check Windows Execution Policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

7. **Git command not found**:
   - Install Git from [git-scm.com](https://git-scm.com/download/win)
   - Restart Command Prompt after installation
   - Use GitHub Desktop as an alternative

8. **Package compilation errors**:
   - Install Visual Studio Build Tools
   - Try pre-compiled packages: `pip install --only-binary=all package_name`
   - Use conda instead of pip for problematic packages

9. **Permission errors**:
   - Run Command Prompt as Administrator
   - Check antivirus software isn't blocking Python
   - Disable Windows Defender real-time protection temporarily during setup

10. **Training fails with multiprocessing errors**:
    - Always use `--workers 0` on Windows
    - Reduce batch size if memory issues occur
    - Close other applications to free up RAM

## Additional Resources

- [YOLOv5 Documentation](https://github.com/ultralytics/yolov5)
- [Face Mask Dataset](https://www.kaggle.com/andrewmvd/face-mask-detection)
- [OpenCV Documentation](https://docs.opencv.org/)
