@echo off
:: Face Mask Detection Project Setup Script for Windows
:: This script automates the setup process for the face mask detection project

echo ==== Setting up Face Mask Detection Project ====

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH. Please install Python 3.6+ and try again.
    pause
    exit /b 1
)

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Warning: Failed to create virtual environment. Continuing without virtual environment.
    echo We recommend installing dependencies in a virtual environment.
    goto :install_deps
)

:: Activate virtual environment
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Warning: Failed to activate virtual environment. Continuing without virtual environment.
    goto :install_deps
)

echo Virtual environment created and activated.

:install_deps
:: Install project dependencies
echo Installing project dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install project dependencies.
    pause
    exit /b 1
)

:: Check if YOLOv5 is already cloned
if not exist "yolov5" (
    echo Cloning YOLOv5 repository...
    git clone https://github.com/ultralytics/yolov5
    if %errorlevel% neq 0 (
        echo Error: Failed to clone YOLOv5 repository. Please check your git installation and internet connection.
        pause
        exit /b 1
    )
    
    cd yolov5
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error: Failed to install YOLOv5 dependencies.
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo YOLOv5 repository cloned and dependencies installed.
) else (
    echo YOLOv5 repository already exists.
)

:: Check if the model file exists
if not exist "models\mask_yolov5.pt" (
    echo Model file not found in models directory.
    echo You need to either:
    echo 1. Train your own model using 'python prepare.py' and then:
    echo    'cd yolov5 && python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0'
    echo 2. Download a pre-trained model from the project repository
    
    :: Create models directory if it doesn't exist
    if not exist "models" mkdir models
) else (
    echo Model file found in models directory.
)

:: Check if the dataset is available
if not exist "datasets\archive.zip" (
    echo Dataset not found. You'll need to download it to train the model.
    echo You can download it from: https://www.kaggle.com/andrewmvd/face-mask-detection
    
    :: Create datasets directory if it doesn't exist
    if not exist "datasets" mkdir datasets
) else (
    echo Dataset found.
)

echo ==== Setup Complete ====
echo To run the face mask detection on your webcam:
echo python webcam.py
echo For help with available options:
echo python webcam.py --help
echo To prepare the dataset for training:
echo python prepare.py
echo Enjoy!

pause 