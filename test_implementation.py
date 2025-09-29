#!/usr/bin/env python3
"""
Test script for MONAI-based Brain Tumor Segmentation
Validates the implementation without requiring actual BraTS data
"""

import os
import sys
import tempfile
import numpy as np
import torch
import nibabel as nib
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_data(data_dir: str, num_cases: int = 2) -> bool:
    """
    Create synthetic BraTS-like data for testing.
    
    Args:
        data_dir: Directory to create test data in
        num_cases: Number of test cases to create
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        data_path = Path(data_dir)
        
        # Create directory structure
        train_dir = data_path / "processed_data" / "train"
        test_dir = data_path / "processed_data" / "test"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create synthetic data
        for i in range(num_cases):
            case_name = f"TestCase_{i+1:03d}"
            
            # Create training case
            train_case_dir = train_dir / case_name
            train_case_dir.mkdir(exist_ok=True)
            
            # Create test case
            test_case_dir = test_dir / case_name
            test_case_dir.mkdir(exist_ok=True)
            
            # Generate synthetic volumes (smaller size for testing)
            volume_shape = (64, 64, 64)
            
            # Create modality data
            modalities = ['t1', 't1ce', 't2', 'flair']
            for modality in modalities:
                # Generate realistic-looking brain data
                data = np.random.normal(100, 50, volume_shape).astype(np.float32)
                data = np.clip(data, 0, 255)
                
                # Add some structure to make it more brain-like
                center = np.array(volume_shape) // 2
                for x in range(volume_shape[0]):
                    for y in range(volume_shape[1]):
                        for z in range(volume_shape[2]):
                            dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                            if dist > 25:  # Outside "brain"
                                data[x, y, z] = 0
                
                # Save training data
                img = nib.Nifti1Image(data, np.eye(4))
                nib.save(img, train_case_dir / f"{case_name}_{modality}.nii.gz")
                
                # Save test data (without segmentation)
                nib.save(img, test_case_dir / f"{case_name}_{modality}.nii.gz")
            
            # Create segmentation mask for training data only
            seg_data = np.zeros(volume_shape, dtype=np.uint8)
            
            # Add some tumor regions
            # Background: 0, NCR/NET: 1, ED: 2, ET: 4 (BraTS convention)
            center = np.array(volume_shape) // 2
            
            # Enhancing tumor (label 4)
            for x in range(center[0]-5, center[0]+5):
                for y in range(center[1]-5, center[1]+5):
                    for z in range(center[2]-5, center[2]+5):
                        if 0 <= x < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= z < volume_shape[2]:
                            seg_data[x, y, z] = 4
            
            # Edema (label 2)
            for x in range(center[0]-10, center[0]+10):
                for y in range(center[1]-10, center[1]+10):
                    for z in range(center[2]-10, center[2]+10):
                        if 0 <= x < volume_shape[0] and 0 <= y < volume_shape[1] and 0 <= z < volume_shape[2]:
                            if seg_data[x, y, z] == 0:  # Only if not already ET
                                dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                                if dist < 8:
                                    seg_data[x, y, z] = 2
            
            # Save segmentation
            seg_img = nib.Nifti1Image(seg_data, np.eye(4))
            nib.save(seg_img, train_case_dir / f"{case_name}_seg.nii.gz")
        
        logger.info(f"Created {num_cases} synthetic test cases")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create synthetic data: {str(e)}")
        return False


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        import torch
        import monai
        from brain_tumor_segmentation import BrainTumorSegmentation, create_file_list, ArchiveExtractor
        logger.info("✅ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False


def test_archive_extractor():
    """Test archive extraction functionality."""
    logger.info("Testing archive extractor...")
    
    try:
        from brain_tumor_segmentation import ArchiveExtractor
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with a non-existent file (should handle gracefully)
            result = ArchiveExtractor.extract_archive("nonexistent.zip", temp_dir)
            if result:
                logger.error("❌ Archive extractor should return False for non-existent files")
                return False
            
            logger.info("✅ Archive extractor handles errors correctly")
            return True
            
    except Exception as e:
        logger.error(f"❌ Archive extractor test failed: {str(e)}")
        return False


def test_adaptive_roi_transform():
    """Test the adaptive ROI transform."""
    logger.info("Testing adaptive ROI transform...")
    
    try:
        from brain_tumor_segmentation import AdaptiveROITransform
        
        # Create test data
        test_data = {
            "image": torch.randn(1, 32, 32, 32),  # Small volume
            "label": torch.randint(0, 4, (1, 32, 32, 32))
        }
        
        # Create transform
        transform = AdaptiveROITransform(keys=["image"], target_size=(64, 64, 64), min_size=(32, 32, 32))
        
        # Apply transform
        result = transform(test_data)
        
        # Check that adaptive size was calculated
        if "image_adaptive_size" not in result:
            logger.error("❌ Adaptive ROI transform didn't create adaptive_size")
            return False
        
        logger.info("✅ Adaptive ROI transform working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive ROI transform test failed: {str(e)}")
        return False


def test_model_creation():
    """Test model creation with different architectures."""
    logger.info("Testing model creation...")
    
    try:
        from brain_tumor_segmentation import BrainTumorSegmentation
        
        # Test different model configurations
        models_to_test = ["unet", "segresnet"]  # Skip UNETR for faster testing
        
        for model_name in models_to_test:
            config = {
                "model_name": model_name,
                "num_classes": 4,
                "roi_size": (64, 64, 64),  # Smaller for testing
            }
            
            segmentation = BrainTumorSegmentation(config)
            model = segmentation.create_model()
            
            # Test forward pass with dummy data
            dummy_input = torch.randn(1, 4, 64, 64, 64)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                model = model.cuda()
            
            with torch.no_grad():
                output = model(dummy_input)
            
            if output.shape[1] != 4:  # num_classes
                logger.error(f"❌ Model {model_name} output shape incorrect: {output.shape}")
                return False
            
            logger.info(f"✅ Model {model_name} created and tested successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model creation test failed: {str(e)}")
        return False


def test_data_loading():
    """Test data loading with synthetic data."""
    logger.info("Testing data loading...")
    
    try:
        from brain_tumor_segmentation import BrainTumorSegmentation, create_file_list
        
        # Create temporary directory with synthetic data
        with tempfile.TemporaryDirectory() as temp_dir:
            if not create_synthetic_data(temp_dir, num_cases=2):
                return False
            
            # Test file list creation
            train_dir = os.path.join(temp_dir, "processed_data", "train")
            test_dir = os.path.join(temp_dir, "processed_data", "test")
            
            train_files = create_file_list(train_dir, "train")
            test_files = create_file_list(test_dir, "test")
            
            if len(train_files) != 2:
                logger.error(f"❌ Expected 2 training files, got {len(train_files)}")
                return False
            
            if len(test_files) != 2:
                logger.error(f"❌ Expected 2 test files, got {len(test_files)}")
                return False
            
            # Test data loader creation
            config = {
                "batch_size": 1,
                "cache_rate": 0.1,
                "num_workers": 0,
                "roi_size": (32, 32, 32),  # Small for testing
            }
            
            segmentation = BrainTumorSegmentation(config)
            train_loader, val_loader = segmentation.create_data_loaders(train_files, test_files)
            
            # Test loading a batch
            train_batch = next(iter(train_loader))
            
            if "image" not in train_batch or "label" not in train_batch:
                logger.error("❌ Data loader batch missing required keys")
                return False
            
            logger.info("✅ Data loading test successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {str(e)}")
        return False


def test_transforms():
    """Test data transforms."""
    logger.info("Testing transforms...")
    
    try:
        from brain_tumor_segmentation import BrainTumorSegmentation
        
        config = {
            "roi_size": (32, 32, 32),
            "spacing": (1.0, 1.0, 1.0),
        }
        
        segmentation = BrainTumorSegmentation(config)
        
        # Test both train and validation transforms
        train_transforms = segmentation.get_transforms("train")
        val_transforms = segmentation.get_transforms("val")
        
        if train_transforms is None or val_transforms is None:
            logger.error("❌ Transform creation failed")
            return False
        
        logger.info("✅ Transform creation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Transform test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    logger.info("Starting comprehensive test suite...")
    
    tests = [
        ("Imports", test_imports),
        ("Archive Extractor", test_archive_extractor),
        ("Adaptive ROI Transform", test_adaptive_roi_transform),
        ("Model Creation", test_model_creation),
        ("Transforms", test_transforms),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {str(e)}")
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:25} {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        logger.error(f"⚠️  {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)