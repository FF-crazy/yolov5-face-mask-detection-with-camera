# Face Mask Detection Project Setup Script for Windows PowerShell
# This script automates the setup process for the face mask detection project

Write-Host "==== Setting up Face Mask Detection Project ====" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH. Please install Python 3.6+ and try again." -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create and activate virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
try {
    python -m venv .venv
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} catch {
    Write-Host "Warning: Failed to create virtual environment. Continuing without virtual environment." -ForegroundColor Yellow
    Write-Host "We recommend installing dependencies in a virtual environment." -ForegroundColor Yellow
    $skipVenv = $true
}

# Activate virtual environment if created successfully
if (-not $skipVenv) {
    try {
        & ".\.venv\Scripts\Activate.ps1"
        Write-Host "Virtual environment activated." -ForegroundColor Green
    } catch {
        Write-Host "Warning: Failed to activate virtual environment. You may need to adjust execution policy." -ForegroundColor Yellow
        Write-Host "Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
        Write-Host "Continuing without virtual environment." -ForegroundColor Yellow
    }
}

# Install project dependencies
Write-Host "Installing project dependencies..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    Write-Host "Project dependencies installed successfully." -ForegroundColor Green
} catch {
    Write-Host "Error: Failed to install project dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if YOLOv5 is already cloned
if (-not (Test-Path "yolov5")) {
    Write-Host "Cloning YOLOv5 repository..." -ForegroundColor Yellow
    try {
        git clone https://github.com/ultralytics/yolov5
        Push-Location yolov5
        pip install -r requirements.txt
        Pop-Location
        Write-Host "YOLOv5 repository cloned and dependencies installed." -ForegroundColor Green
    } catch {
        Write-Host "Error: Failed to clone YOLOv5 repository. Please check your git installation and internet connection." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host "YOLOv5 repository already exists." -ForegroundColor Green
}

# Check if the model file exists
if (-not (Test-Path "models\mask_yolov5.pt")) {
    Write-Host "Model file not found in models directory." -ForegroundColor Yellow
    Write-Host "You need to either:" -ForegroundColor Yellow
    Write-Host "1. Train your own model using 'python prepare.py' and then:" -ForegroundColor Yellow
    Write-Host "   'cd yolov5; python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0; cd ..'" -ForegroundColor Yellow
    Write-Host "2. Download a pre-trained model from the project repository" -ForegroundColor Yellow
    
    # Create models directory if it doesn't exist
    if (-not (Test-Path "models")) {
        New-Item -ItemType Directory -Path "models" | Out-Null
    }
} else {
    Write-Host "Model file found in models directory." -ForegroundColor Green
}

# Check if the dataset is available
if (-not (Test-Path "datasets\archive.zip")) {
    Write-Host "Dataset not found. You'll need to download it to train the model." -ForegroundColor Yellow
    Write-Host "You can download it from: https://www.kaggle.com/andrewmvd/face-mask-detection" -ForegroundColor Yellow
    
    # Create datasets directory if it doesn't exist
    if (-not (Test-Path "datasets")) {
        New-Item -ItemType Directory -Path "datasets" | Out-Null
    }
} else {
    Write-Host "Dataset found." -ForegroundColor Green
}

Write-Host "==== Setup Complete ====" -ForegroundColor Green
Write-Host "To run the face mask detection on your webcam:" -ForegroundColor Green
Write-Host "python webcam.py" -ForegroundColor Cyan
Write-Host "For help with available options:" -ForegroundColor Green
Write-Host "python webcam.py --help" -ForegroundColor Cyan
Write-Host "To prepare the dataset for training:" -ForegroundColor Green
Write-Host "python prepare.py" -ForegroundColor Cyan
Write-Host "Enjoy!" -ForegroundColor Green

Read-Host "Press Enter to exit" 