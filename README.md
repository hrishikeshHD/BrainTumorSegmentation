# MONAI-based 3D Brain Tumor Segmentation

A robust, production-ready 3D brain tumor segmentation pipeline using MONAI framework, optimized for Google Colab with A100 GPUs. This implementation addresses common issues found in brain tumor segmentation workflows and provides a complete solution for BraTS data processing.

## 🚀 Features

### ✅ **Issue Fixes Implemented**
- **ROI Size Issues**: Adaptive ROI sizing that automatically adjusts to smaller volumes
- **Tensor Dimension Mismatches**: Fixed tensor shape incompatibilities across transforms
- **Autocast Compatibility**: Proper automatic mixed precision (AMP) implementation for A100 GPUs
- **Training/Validation Process**: Robust pipeline with proper error handling and logging

### 🎯 **Core Capabilities**
- **Archive Support**: Automatic extraction of .rar/.zip files containing BraTS data
- **Folder Structure**: Works with `processed_data/{train|test}/` organization
- **GPU Optimization**: Specifically optimized for Google Colab A100 GPUs
- **Memory Management**: Efficient memory usage with caching and batch optimization
- **Multiple Models**: Support for UNet, SegResNet, and UNETR architectures

## 📁 Project Structure

```
BrainTumorSegmentation/
├── brain_tumor_segmentation.py      # Main segmentation script
├── colab_setup.py                   # Google Colab setup script
├── Brain_Tumor_Segmentation_Colab.ipynb  # Interactive Jupyter notebook
├── requirements.txt                 # Python dependencies
└── README.md                       # This file
```

## 🔧 Installation & Setup

### Option 1: Google Colab (Recommended)

1. **Open the notebook**: Upload `Brain_Tumor_Segmentation_Colab.ipynb` to Google Colab
2. **Select A100 GPU**: Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: A100
3. **Run setup**: Execute the setup cells in the notebook

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/hrishikeshHD/BrainTumorSegmentation.git
cd BrainTumorSegmentation

# Install dependencies
pip install -r requirements.txt

# Run setup (optional)
python colab_setup.py
```

## 📊 Data Preparation

### Expected Data Structure

Your BraTS data should be organized as follows:

```
processed_data/
├── train/
│   ├── BraTS20_Training_001/
│   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   └── BraTS20_Training_001_seg.nii.gz
│   └── ... (more training cases)
└── test/
    ├── BraTS20_Validation_001/
    │   ├── BraTS20_Validation_001_t1.nii.gz
    │   ├── BraTS20_Validation_001_t1ce.nii.gz
    │   ├── BraTS20_Validation_001_t2.nii.gz
    │   └── BraTS20_Validation_001_flair.nii.gz
    └── ... (more validation cases)
```

### Archive Extraction

The pipeline automatically handles:
- **ZIP files**: `.zip` archives containing BraTS data
- **RAR files**: `.rar` archives (requires `rarfile` package)

Simply place your archives in the data directory, and they will be extracted automatically.

## 🚀 Usage

### Quick Start (Google Colab)

```python
# 1. Setup (run once)
!python colab_setup.py

# 2. Import and configure
from brain_tumor_segmentation import BrainTumorSegmentation

config = {
    "max_epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "roi_size": (128, 128, 128),
    "model_name": "unet",
    "use_amp": True,
}

# 3. Initialize and train
segmentation = BrainTumorSegmentation(config)
segmentation.train(train_files, val_files)
```

### Command Line Usage

```bash
# Basic usage
python brain_tumor_segmentation.py /path/to/data

# The script will automatically:
# 1. Extract any .zip/.rar archives
# 2. Setup the expected folder structure
# 3. Start training with optimized settings
```

## ⚙️ Configuration Options

### Model Selection
- `unet`: Standard 3D U-Net (default, most stable)
- `segresnet`: SegResNet architecture (memory efficient)
- `unetr`: Vision Transformer-based U-Net (requires more memory)

### Memory Optimization
```python
# For limited memory (reduce these values)
config = {
    "batch_size": 1,           # Reduce from 2
    "roi_size": (96, 96, 96),  # Reduce from (128, 128, 128)
    "cache_rate": 0.2,         # Reduce from 0.5
    "sw_batch_size": 2,        # Reduce from 4
}
```

### A100 GPU Optimization
```python
# For maximum A100 performance
config = {
    "batch_size": 4,           # Increase if memory allows
    "use_amp": True,           # Enable automatic mixed precision
    "num_workers": 8,          # Increase for faster data loading
    "cache_rate": 1.0,         # Cache all data (if enough RAM)
}
```

## 🔧 Technical Details

### Problem Fixes Implemented

#### 1. ROI Size Issues
- **Problem**: Fixed ROI sizes causing errors with smaller volumes
- **Solution**: Implemented `AdaptiveROITransform` that dynamically adjusts ROI size based on input volume dimensions
- **Code**: See `AdaptiveROITransform` class in `brain_tumor_segmentation.py`

#### 2. Tensor Dimension Mismatches
- **Problem**: Inconsistent tensor shapes between transforms and model input
- **Solution**: Added proper dimension handling with `EnsureChannelFirstd` and `EnsureTyped` transforms
- **Code**: See `get_transforms()` method

#### 3. Autocast Compatibility
- **Problem**: AMP (Automatic Mixed Precision) not working properly with certain operations
- **Solution**: Conditional autocast usage with proper scaler handling for A100 GPUs
- **Code**: See `train_epoch()` method

#### 4. Training/Validation Process
- **Problem**: Unstable training with poor error handling
- **Solution**: Robust pipeline with proper metrics, checkpointing, and error recovery
- **Code**: See `train()` and `validate()` methods

### Performance Optimizations

1. **Memory Management**: Intelligent caching and batch sizing
2. **GPU Utilization**: Tensor core optimization for A100 GPUs
3. **Data Loading**: Optimized transforms and data loaders
4. **Mixed Precision**: Automatic scaling for faster training

## 📈 Results & Metrics

The pipeline tracks:
- **Dice Score**: Main segmentation quality metric
- **Training Loss**: DiceCE loss for stable training
- **Validation Loss**: Generalization monitoring
- **GPU Memory**: Real-time memory usage tracking

## 🐛 Troubleshooting

### Common Issues

#### "CUDA out of memory"
```python
# Reduce memory usage
config["batch_size"] = 1
config["roi_size"] = (96, 96, 96)
config["cache_rate"] = 0.1
```

#### "No training files found"
- Check data structure matches expected format
- Ensure files are in `.nii.gz` format
- Verify BraTS naming convention

#### "Tensor dimension mismatch"
- This should be fixed by the adaptive transforms
- If persisting, check input data format

#### Archive extraction fails
```bash
# Install additional dependencies
pip install patoolib
pip install rarfile
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [MONAI](https://monai.io/) - Medical Open Network for AI
- [BraTS Challenge](http://braintumorsegmentation.org/) - Brain Tumor Segmentation Challenge
- [PyTorch](https://pytorch.org/) - Deep learning framework

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the [Issues](https://github.com/hrishikeshHD/BrainTumorSegmentation/issues) page
3. Create a new issue with detailed information about your problem

---

**Note**: This implementation is designed for research and educational purposes. For clinical use, additional validation and regulatory approval may be required.