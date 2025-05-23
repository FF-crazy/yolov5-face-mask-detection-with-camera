# Windows Setup Guide for Face Mask Detection

This guide provides step-by-step instructions specifically for Windows users to set up and run the Face Mask Detection project.

## Prerequisites

### Required Software

1. **Python 3.6+** (Python 3.8+ recommended)
   - Download from [python.org](https://www.python.org/downloads/)
   - **Important**: Check "Add Python to PATH" during installation
   - Or install from Microsoft Store

2. **Git** (for cloning repositories)
   - Download from [git-scm.com](https://git-scm.com/download/win)
   - Or use GitHub Desktop as an alternative

3. **Camera/Webcam** for real-time detection

### Optional (Recommended)

- **Visual Studio Build Tools** (if packages need compilation)
- **CUDA-compatible GPU** for faster training and inference

## Quick Setup Methods

### Method 1: Command Prompt (Recommended)

1. Open **Command Prompt** or **PowerShell**
2. Navigate to the project directory:

   ```cmd
   cd path\to\yolov5-face-mask-detection
   ```

3. Run the setup script:

   ```cmd
   setup.bat
   ```

### Method 2: PowerShell

1. Open **PowerShell**
2. Navigate to the project directory:

   ```powershell
   cd path\to\yolov5-face-mask-detection
   ```

3. Adjust execution policy (if needed):

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. Run the setup script:

   ```powershell
   .\setup.ps1
   ```

## Manual Setup (Step-by-Step)

If the automated scripts don't work, follow these manual steps:

### Step 1: Create Virtual Environment

```cmd
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

### Step 2: Install Dependencies

```cmd
# Install project dependencies
pip install -r requirements.txt

# Clone YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

### Step 3: Download Dataset (Optional)

1. Go to [Kaggle Face Mask Detection](https://www.kaggle.com/andrewmvd/face-mask-detection)
2. Download `archive.zip`
3. Create `datasets` folder and place the zip file inside

### Step 4: Prepare Dataset

```cmd
python prepare.py
```

### Step 5: Train Model (Optional)

```cmd
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 --data ../mask_config.yaml --weights yolov5s.pt --workers 0
cd ..
```

**Important**: Always use `--workers 0` on Windows to avoid multiprocessing issues.

## Running the Application

### Real-time Detection

```cmd
python webcam.py
```

### With Custom Options

```cmd
# Use different camera
python webcam.py --camera 1

# Custom confidence threshold
python webcam.py --conf-thres 0.7

# Custom model
python webcam.py --model path\to\your\model.pt

# Get help
python webcam.py --help
```

## Common Windows Issues & Solutions

### Issue 1: "Python not found"

**Solution:**

- Reinstall Python with "Add to PATH" checked
- Try using `py` instead of `python`
- Restart Command Prompt after installation

### Issue 2: Virtual environment activation fails

**Solutions:**

- Run Command Prompt as Administrator
- For PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Use full path: `C:\path\to\.venv\Scripts\activate`

### Issue 3: Git not found

**Solutions:**

- Install Git from [git-scm.com](https://git-scm.com/download/win)
- Restart Command Prompt
- Add Git to PATH manually

### Issue 4: Package installation fails

**Solutions:**

- Install Visual Studio Build Tools
- Use: `pip install --only-binary=:all: package_name`
- Try conda instead: `conda install package_name`

### Issue 5: Permission errors

**Solutions:**

- Run Command Prompt as Administrator
- Check antivirus software
- Temporarily disable Windows Defender real-time protection

### Issue 6: Camera access issues

**Solutions:**

- Check camera permissions in Windows Settings
- Close other applications using the camera
- Try different camera ID: `--camera 1` or `--camera 2`

### Issue 7: Training fails with multiprocessing errors

**Solutions:**

- Always use `--workers 0`
- Reduce batch size: `--batch 8`
- Close other applications to free RAM

## GPU Setup (Optional)

For faster training and inference with NVIDIA GPUs:

1. Install CUDA Toolkit from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Install PyTorch with CUDA support:

   ```cmd
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. Verify CUDA is working:

   ```cmd
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Performance Tips

1. **Close unnecessary applications** before training
2. **Use smaller batch sizes** if you encounter memory issues
3. **Reduce image resolution** for faster inference
4. **Use GPU** if available for better performance
5. **Monitor task manager** for resource usage

## Alternative Installation Methods

### Using Anaconda/Miniconda

```cmd
# Create conda environment
conda create -n mask-detection python=3.8
conda activate mask-detection

# Install packages
conda install opencv pytorch torchvision -c pytorch -c conda-forge
pip install -r requirements.txt
```

### Using Docker (Advanced)

If you have Docker Desktop for Windows:

```cmd
# Build Docker image (create Dockerfile first)
docker build -t mask-detection .

# Run container
docker run -it --rm mask-detection
```

## Getting Help

If you encounter issues not covered here:

1. Check the main [SETUP.md](SETUP.md) file
2. Search existing issues on GitHub
3. Create a new issue with:
   - Windows version
   - Python version
   - Complete error message
   - Steps that led to the error

## Video Tutorial

For a visual guide, check out our [YouTube tutorial](link-to-tutorial) (if available).

---

**Happy coding! 🎭✨**
