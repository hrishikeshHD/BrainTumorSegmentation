#!/usr/bin/env python3
"""
Example usage of the MONAI-based 3D Brain Tumor Segmentation pipeline.
This script demonstrates how to configure and run the segmentation for BraTS data.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_tumor_segmentation import BrainTumorSegmentation, create_file_list

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Example usage of the brain tumor segmentation pipeline."""
    
    # Configuration for different scenarios
    
    # 1. High-performance configuration for A100 GPU
    a100_config = {
        "seed": 42,
        "max_epochs": 200,
        "batch_size": 4,  # Large batch for A100
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "roi_size": (128, 128, 128),
        "sw_batch_size": 8,
        "spacing": (1.0, 1.0, 1.0),
        "num_classes": 4,
        "model_name": "unet",  # or "segresnet", "unetr"
        "use_amp": True,  # Essential for A100 performance
        "cache_rate": 1.0,  # Cache all data for speed
        "num_workers": 8,
        "log_interval": 50,
    }
    
    # 2. Memory-conservative configuration for limited resources
    conservative_config = {
        "seed": 42,
        "max_epochs": 100,
        "batch_size": 1,  # Small batch
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "roi_size": (96, 96, 96),  # Smaller ROI
        "sw_batch_size": 2,
        "spacing": (1.0, 1.0, 1.0),
        "num_classes": 4,
        "model_name": "segresnet",  # More memory efficient
        "use_amp": True,
        "cache_rate": 0.2,  # Low cache to save memory
        "num_workers": 2,
        "log_interval": 25,
    }
    
    # 3. Quick testing configuration
    test_config = {
        "seed": 42,
        "max_epochs": 5,  # Just a few epochs for testing
        "batch_size": 1,
        "learning_rate": 1e-3,  # Higher LR for quick testing
        "weight_decay": 1e-5,
        "roi_size": (64, 64, 64),  # Small ROI for speed
        "sw_batch_size": 2,
        "spacing": (2.0, 2.0, 2.0),  # Lower resolution for speed
        "num_classes": 4,
        "model_name": "unet",
        "use_amp": True,
        "cache_rate": 0.1,
        "num_workers": 2,
        "log_interval": 10,
    }
    
    # Choose configuration based on your needs
    config = a100_config  # Change this to conservative_config or test_config as needed
    
    print("MONAI-based 3D Brain Tumor Segmentation")
    print("=" * 50)
    print(f"Configuration: {config['model_name']} model")
    print(f"Epochs: {config['max_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"ROI size: {config['roi_size']}")
    print(f"Mixed precision: {config['use_amp']}")
    print("=" * 50)
    
    # Data directory (modify this path according to your setup)
    data_root = "/content/data"  # Default for Google Colab
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    
    if not os.path.exists(data_root):
        print(f"⚠️  Data directory not found: {data_root}")
        print("Please provide the correct path to your BraTS data.")
        print("Usage: python example_usage.py /path/to/your/data")
        print("\nFor Google Colab, upload your data to /content/data/")
        return
    
    try:
        # Initialize the segmentation pipeline
        segmentation = BrainTumorSegmentation(config)
        
        # Setup data directories (handles archive extraction)
        train_dir, test_dir = segmentation.setup_data_directories(data_root)
        
        # Create file lists
        train_files = create_file_list(train_dir, "train")
        val_files = create_file_list(test_dir, "test")
        
        print(f"Found {len(train_files)} training cases")
        print(f"Found {len(val_files)} validation cases")
        
        if not train_files:
            print("❌ No training data found!")
            print("\nExpected data structure:")
            print("processed_data/")
            print("├── train/")
            print("│   ├── CaseName/")
            print("│   │   ├── CaseName_t1.nii.gz")
            print("│   │   ├── CaseName_t1ce.nii.gz")
            print("│   │   ├── CaseName_t2.nii.gz")
            print("│   │   ├── CaseName_flair.nii.gz")
            print("│   │   └── CaseName_seg.nii.gz")
            print("└── test/")
            print("    └── ... (similar structure)")
            return
        
        # Show example file structure
        print("\nExample training case:")
        print(f"  Image files: {train_files[0]['image']}")
        if 'label' in train_files[0]:
            print(f"  Label file: {train_files[0]['label']}")
        
        # Start training
        print("\n🚀 Starting training...")
        print("This may take several hours depending on your configuration.")
        
        segmentation.train(train_files, val_files)
        
        print("\n✅ Training completed successfully!")
        print("Best model saved as: best_metric_model.pth")
        
        # Show training summary
        print(f"\n📊 Training Summary:")
        print(f"Best validation Dice score: {segmentation.best_metric:.4f}")
        print(f"Best epoch: {segmentation.best_metric_epoch}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise


def show_colab_instructions():
    """Show instructions for running in Google Colab."""
    print("Google Colab Setup Instructions:")
    print("=" * 40)
    print("1. Make sure you have GPU runtime enabled:")
    print("   Runtime → Change runtime type → Hardware accelerator: GPU")
    print("\n2. Clone this repository:")
    print("   !git clone https://github.com/hrishikeshHD/BrainTumorSegmentation.git")
    print("   %cd BrainTumorSegmentation")
    print("\n3. Install dependencies:")
    print("   !python colab_setup.py")
    print("\n4. Upload your BraTS data to /content/data/")
    print("   - Can be .zip or .rar archives")
    print("   - Will be automatically extracted")
    print("\n5. Run the example:")
    print("   !python example_usage.py")
    print("\nAlternatively, use the Jupyter notebook:")
    print("   Brain_Tumor_Segmentation_Colab.ipynb")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        show_colab_instructions()
    else:
        main()