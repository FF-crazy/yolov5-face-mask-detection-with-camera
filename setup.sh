#!/bin/bash

# Face Mask Detection Project Setup Script
# This script automates the setup process for the face mask detection project

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==== Setting up Face Mask Detection Project ====${NC}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python is installed
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed. Please install Python 3 and try again.${NC}"
    exit 1
fi

# Create and activate virtual environment if venv is available
echo -e "${YELLOW}Creating virtual environment...${NC}"
if python3 -m venv -h >/dev/null 2>&1; then
    python3 -m venv .venv
    
    # Activate virtual environment based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source .venv/Scripts/activate
    else
        source .venv/bin/activate
    fi
    
    echo -e "${GREEN}Virtual environment created and activated.${NC}"
else
    echo -e "${YELLOW}Warning: Python venv module not found. Continuing without virtual environment.${NC}"
    echo -e "${YELLOW}We recommend installing dependencies in a virtual environment.${NC}"
fi

# Install project dependencies
echo -e "${YELLOW}Installing project dependencies...${NC}"
pip install -r requirements.txt

# Check if YOLOv5 is already cloned
if [ ! -d "yolov5" ]; then
    echo -e "${YELLOW}Cloning YOLOv5 repository...${NC}"
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    cd ..
    echo -e "${GREEN}YOLOv5 repository cloned and dependencies installed.${NC}"
else
    echo -e "${GREEN}YOLOv5 repository already exists.${NC}"
fi

# Check if the model file exists
if [ ! -f "models/mask_yolov5.pt" ]; then
    echo -e "${YELLOW}Model file not found in models directory.${NC}"
    echo -e "${YELLOW}You need to either:${NC}"
    echo -e "${YELLOW}1. Train your own model using 'python prepare.py' and then 'cd yolov5 && python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0'${NC}"
    echo -e "${YELLOW}2. Download a pre-trained model from the project repository${NC}"
    
    # Create models directory if it doesn't exist
    mkdir -p models
else
    echo -e "${GREEN}Model file found in models directory.${NC}"
fi

# Check if the dataset is available
if [ ! -f "datasets/archive.zip" ]; then
    echo -e "${YELLOW}Dataset not found. You'll need to download it to train the model.${NC}"
    echo -e "${YELLOW}You can download it from: https://www.kaggle.com/andrewmvd/face-mask-detection${NC}"
    
    # Create datasets directory if it doesn't exist
    mkdir -p datasets
else
    echo -e "${GREEN}Dataset found.${NC}"
fi

echo -e "${GREEN}==== Setup Complete ====${NC}"
echo -e "${GREEN}To run the face mask detection on your webcam:${NC}"
echo -e "${YELLOW}python webcam.py${NC}"
echo -e "${GREEN}For help with available options:${NC}"
echo -e "${YELLOW}python webcam.py --help${NC}"
echo -e "${GREEN}To prepare the dataset for training:${NC}"
echo -e "${YELLOW}python prepare.py${NC}"
echo -e "${GREEN}Enjoy!${NC}" 