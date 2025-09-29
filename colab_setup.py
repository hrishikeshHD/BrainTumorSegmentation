#!/usr/bin/env python3
"""
Google Colab Setup Script for MONAI-based Brain Tumor Segmentation
Handles installation, GPU optimization, and data preparation for A100 GPUs
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_dependencies():
    """Install required packages for the segmentation pipeline."""
    logger.info("Installing dependencies...")
    
    # Essential packages for Colab
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0",
        "monai[all]>=1.3.0",
        "nibabel>=5.0.0",
        "rarfile>=4.1",
        "patoolib>=1.12.0",
        "psutil>=5.9.0",
        "tensorboard>=2.13.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
    
    return True


def setup_gpu_environment():
    """Optimize GPU settings for A100."""
    logger.info("Setting up GPU environment for A100...")
    
    # Set environment variables for optimal performance
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # A100 architecture
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # Set memory management settings
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # A100 specific optimizations
        device = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {device.name}")
        logger.info(f"Memory: {device.total_memory / 1024**3:.1f} GB")
        
        if "A100" in device.name:
            logger.info("A100 GPU detected - enabling tensor core optimizations")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        return True
    else:
        logger.warning("No GPU available")
        return False


def setup_data_directories():
    """Create necessary data directories for the pipeline."""
    logger.info("Setting up data directories...")
    
    base_dir = Path("/content")
    data_dir = base_dir / "data"
    
    # Create directory structure
    directories = [
        data_dir,
        data_dir / "processed_data",
        data_dir / "processed_data" / "train", 
        data_dir / "processed_data" / "test",
        data_dir / "models",
        data_dir / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return str(data_dir)


def check_colab_environment():
    """Check if running in Google Colab and setup accordingly."""
    try:
        import google.colab
        logger.info("Running in Google Colab environment")
        return True
    except ImportError:
        logger.info("Not running in Google Colab")
        return False


def download_sample_data():
    """Download sample BraTS data for testing."""
    logger.info("Setting up sample data download...")
    
    print("""
    To use this segmentation pipeline, you need BraTS dataset in the following structure:
    
    /content/data/processed_data/
    ├── train/
    │   ├── BraTS20_Training_001/
    │   │   ├── BraTS20_Training_001_t1.nii.gz
    │   │   ├── BraTS20_Training_001_t1ce.nii.gz
    │   │   ├── BraTS20_Training_001_t2.nii.gz
    │   │   ├── BraTS20_Training_001_flair.nii.gz
    │   │   └── BraTS20_Training_001_seg.nii.gz
    │   └── ... (more cases)
    └── test/
        ├── BraTS20_Validation_001/
        │   ├── BraTS20_Validation_001_t1.nii.gz
        │   ├── BraTS20_Validation_001_t1ce.nii.gz
        │   ├── BraTS20_Validation_001_t2.nii.gz
        │   └── BraTS20_Validation_001_flair.nii.gz
        └── ... (more cases)
    
    You can:
    1. Upload your own BraTS data
    2. Extract from .zip/.rar archives 
    3. Download from official BraTS challenge
    """)


def main():
    """Main setup function for Google Colab."""
    logger.info("Starting Google Colab setup for MONAI Brain Tumor Segmentation...")
    
    # Check environment
    is_colab = check_colab_environment()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Setup GPU
    gpu_available = setup_gpu_environment()
    if not gpu_available:
        logger.warning("No GPU available - training will be very slow")
    
    # Setup directories
    data_dir = setup_data_directories()
    
    # Show data setup instructions
    download_sample_data()
    
    logger.info("Setup completed successfully!")
    logger.info(f"Data directory: {data_dir}")
    logger.info("You can now run the brain tumor segmentation script.")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)