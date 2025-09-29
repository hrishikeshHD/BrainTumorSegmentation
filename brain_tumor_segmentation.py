#!/usr/bin/env python3
"""
MONAI-based 3D Brain Tumor Segmentation Script
Optimized for Google Colab with A100 GPUs

This script implements a robust 3D brain tumor segmentation pipeline using MONAI
framework, designed to work with BraTS data and handle common issues like:
- ROI size adaptation for smaller volumes
- Tensor dimension mismatches
- Autocast compatibility
- Archive extraction (.rar/.zip)
"""

import os
import sys
import json
import zipfile
import rarfile
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import monai
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import (
    CacheDataset,
    DataLoader as MonaiDataLoader,
    Dataset,
    decollate_batch,
    load_decathlon_datalist,
)
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, UNETR
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)
from monai.utils import first, set_determinism

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class ArchiveExtractor:
    """Handle extraction of .rar and .zip archives containing BraTS data."""
    
    @staticmethod
    def extract_archive(archive_path: str, extract_to: str) -> bool:
        """
        Extract .rar or .zip archive to specified directory.
        
        Args:
            archive_path: Path to the archive file
            extract_to: Directory to extract files to
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            os.makedirs(extract_to, exist_ok=True)
            
            if archive_path.lower().endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                logger.info(f"Successfully extracted ZIP archive: {archive_path}")
                return True
            
            elif archive_path.lower().endswith('.rar'):
                with rarfile.RarFile(archive_path, 'r') as rar_ref:
                    rar_ref.extractall(extract_to)
                logger.info(f"Successfully extracted RAR archive: {archive_path}")
                return True
            
            else:
                logger.error(f"Unsupported archive format: {archive_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extract archive {archive_path}: {str(e)}")
            return False


class AdaptiveROITransform(MapTransform):
    """
    Custom transform to handle adaptive ROI sizing for smaller volumes.
    Addresses the ROI size issues mentioned in the problem statement.
    """
    
    def __init__(self, keys, target_size=(128, 128, 128), min_size=(64, 64, 64)):
        super().__init__(keys)
        self.target_size = target_size
        self.min_size = min_size
    
    def __call__(self, data):
        d = dict(data)
        
        for key in self.keys:
            if key in d:
                image_shape = d[key].shape[1:]  # Exclude channel dimension
                
                # Calculate adaptive ROI size based on image dimensions
                adaptive_size = []
                for i, (img_dim, target_dim, min_dim) in enumerate(
                    zip(image_shape, self.target_size, self.min_size)
                ):
                    # Use smaller of image dimension or target, but at least min_dim
                    roi_dim = max(min(img_dim, target_dim), min_dim)
                    adaptive_size.append(roi_dim)
                
                # Store adaptive size for use in other transforms
                d[f"{key}_adaptive_size"] = adaptive_size
                
        return d


class BrainTumorSegmentation:
    """
    Main class for 3D brain tumor segmentation using MONAI.
    Optimized for Google Colab with A100 GPUs.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the segmentation pipeline.
        
        Args:
            config: Configuration dictionary containing model and training parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler() if self.config.get("use_amp", True) else None
        
        # Set deterministic behavior
        set_determinism(seed=self.config.get("seed", 42))
        
        # Initialize metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.best_metric = -1
        self.best_metric_epoch = -1
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"MONAI version: {monai.__version__}")
        
    def setup_data_directories(self, data_root: str) -> Tuple[str, str]:
        """
        Setup data directories and extract archives if needed.
        
        Args:
            data_root: Root directory containing the data
            
        Returns:
            Tuple of (train_dir, test_dir) paths
        """
        data_path = Path(data_root)
        
        # Check for archives to extract
        for archive_file in data_path.glob("*.rar") + data_path.glob("*.zip"):
            extract_dir = data_path / "extracted"
            if ArchiveExtractor.extract_archive(str(archive_file), str(extract_dir)):
                # Update data_path to extracted directory if extraction successful
                if (extract_dir / "processed_data").exists():
                    data_path = extract_dir
        
        # Setup expected directory structure
        processed_data_dir = data_path / "processed_data"
        train_dir = processed_data_dir / "train"
        test_dir = processed_data_dir / "test"
        
        # Create directories if they don't exist
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Train directory: {train_dir}")
        logger.info(f"Test directory: {test_dir}")
        
        return str(train_dir), str(test_dir)
    
    def get_transforms(self, mode: str = "train") -> Compose:
        """
        Get data transforms for training or validation.
        Includes fixes for tensor dimension mismatches and ROI size issues.
        
        Args:
            mode: Either "train" or "val"
            
        Returns:
            Composed transforms
        """
        # Get adaptive ROI size
        roi_size = self.config.get("roi_size", (128, 128, 128))
        
        common_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=self.config.get("spacing", (1.0, 1.0, 1.0)),
                mode=("bilinear", "nearest"),
            ),
            # Custom adaptive ROI transform to handle size issues
            AdaptiveROITransform(keys=["image"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Use adaptive sizing to prevent ROI issues
            SpatialPadd(keys=["image", "label"], spatial_size=roi_size, mode="constant"),
        ]
        
        if mode == "train":
            train_transforms = common_transforms + [
                RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=roi_size,
                    random_size=False,
                ),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                ToTensord(keys=["image", "label"]),
            ]
            return Compose(train_transforms)
        else:
            val_transforms = common_transforms + [
                # For validation, use center crop to avoid dimension issues
                CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
                ToTensord(keys=["image", "label"]),
            ]
            return Compose(val_transforms)
    
    def create_data_loaders(self, train_files: List[Dict], val_files: List[Dict]) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Args:
            train_files: List of training file dictionaries
            val_files: List of validation file dictionaries
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        train_transforms = self.get_transforms("train")
        val_transforms = self.get_transforms("val")
        
        # Use caching for better performance
        cache_rate = self.config.get("cache_rate", 0.5)
        
        train_ds = CacheDataset(
            data=train_files,
            transform=train_transforms,
            cache_rate=cache_rate,
            num_workers=self.config.get("num_workers", 4),
        )
        
        val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_rate=cache_rate,
            num_workers=self.config.get("num_workers", 4),
        )
        
        train_loader = MonaiDataLoader(
            train_ds,
            batch_size=self.config.get("batch_size", 1),
            shuffle=True,
            num_workers=0,  # Set to 0 for Colab compatibility
            pin_memory=True,
        )
        
        val_loader = MonaiDataLoader(
            val_ds,
            batch_size=1,
            num_workers=0,  # Set to 0 for Colab compatibility
            pin_memory=True,
        )
        
        return train_loader, val_loader
    
    def create_model(self) -> nn.Module:
        """
        Create the segmentation model.
        Fixed tensor dimension issues and autocast compatibility.
        
        Returns:
            Segmentation model
        """
        model_name = self.config.get("model_name", "unet")
        num_classes = self.config.get("num_classes", 4)
        
        if model_name.lower() == "unet":
            model = UNet(
                spatial_dims=3,
                in_channels=4,  # T1, T1CE, T2, FLAIR
                out_channels=num_classes,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm=Norm.BATCH,
            )
        elif model_name.lower() == "segresnet":
            model = SegResNet(
                spatial_dims=3,
                init_filters=32,
                in_channels=4,
                out_channels=num_classes,
                dropout_prob=0.2,
            )
        elif model_name.lower() == "unetr":
            model = UNETR(
                in_channels=4,
                out_channels=num_classes,
                img_size=self.config.get("roi_size", (128, 128, 128)),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                res_block=True,
                dropout_rate=0.0,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model.to(self.device)
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: torch.optim.Optimizer, loss_function: nn.Module,
                   epoch: int) -> float:
        """
        Train for one epoch with autocast compatibility fixes.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            optimizer: Optimizer
            loss_function: Loss function
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        model.train()
        epoch_loss = 0
        step = 0
        
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(self.device),
                batch_data["label"].to(self.device),
            )
            
            optimizer.zero_grad()
            
            # Use autocast only if scaler is available and compatible
            if self.scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % self.config.get("log_interval", 100) == 0:
                logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        return epoch_loss / step
    
    def validate(self, model: nn.Module, val_loader: DataLoader, 
                loss_function: nn.Module) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            model: The model to validate
            val_loader: Validation data loader
            loss_function: Loss function
            
        Returns:
            Tuple of (average_loss, dice_score)
        """
        model.eval()
        val_loss = 0
        step = 0
        
        with torch.no_grad():
            for val_data in val_loader:
                step += 1
                val_inputs, val_labels = (
                    val_data["image"].to(self.device),
                    val_data["label"].to(self.device),
                )
                
                # Use sliding window inference to handle memory constraints
                roi_size = self.config.get("roi_size", (128, 128, 128))
                sw_batch_size = self.config.get("sw_batch_size", 4)
                
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model
                )
                
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                
                # Calculate Dice metric
                val_outputs = [
                    AsDiscrete(argmax=True, to_onehot=self.config.get("num_classes", 4))(i) 
                    for i in decollate_batch(val_outputs)
                ]
                val_labels = [
                    AsDiscrete(to_onehot=self.config.get("num_classes", 4))(i) 
                    for i in decollate_batch(val_labels)
                ]
                
                self.dice_metric(y_pred=val_outputs, y=val_labels)
        
        # Aggregate the final mean dice result
        metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        
        return val_loss / step, metric
    
    def train(self, train_files: List[Dict], val_files: List[Dict]):
        """
        Main training loop with all fixes applied.
        
        Args:
            train_files: List of training file dictionaries
            val_files: List of validation file dictionaries
        """
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_files, val_files)
        
        # Create model
        model = self.create_model()
        
        # Setup loss function and optimizer
        loss_function = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            jaccard=False,
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.get("max_epochs", 100)
        )
        
        # Training loop
        for epoch in range(self.config.get("max_epochs", 100)):
            logger.info(f"Epoch {epoch + 1}/{self.config.get('max_epochs', 100)}")
            
            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_function, epoch)
            
            # Validate
            val_loss, val_dice = self.validate(model, val_loader, loss_function)
            
            scheduler.step()
            
            logger.info(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
            )
            
            # Save best model
            if val_dice > self.best_metric:
                self.best_metric = val_dice
                self.best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_model.pth")
                logger.info(f"New best metric: {self.best_metric:.4f}")
        
        logger.info(
            f"Training completed. Best metric: {self.best_metric:.4f} "
            f"at epoch: {self.best_metric_epoch}"
        )


def create_file_list(data_dir: str, split: str = "train") -> List[Dict]:
    """
    Create file list for training or validation.
    Handles the processed_data/{train|test}/ folder structure.
    
    Args:
        data_dir: Directory containing the data
        split: Either "train" or "test"
        
    Returns:
        List of file dictionaries
    """
    data_path = Path(data_dir)
    file_list = []
    
    # Look for BraTS-style naming convention
    for case_dir in data_path.iterdir():
        if case_dir.is_dir():
            # Look for standard BraTS modalities
            t1_file = case_dir / f"{case_dir.name}_t1.nii.gz"
            t1ce_file = case_dir / f"{case_dir.name}_t1ce.nii.gz"
            t2_file = case_dir / f"{case_dir.name}_t2.nii.gz"
            flair_file = case_dir / f"{case_dir.name}_flair.nii.gz"
            
            # For training, look for segmentation mask
            if split == "train":
                seg_file = case_dir / f"{case_dir.name}_seg.nii.gz"
                
                if all(f.exists() for f in [t1_file, t1ce_file, t2_file, flair_file, seg_file]):
                    file_list.append({
                        "image": [str(t1_file), str(t1ce_file), str(t2_file), str(flair_file)],
                        "label": str(seg_file),
                    })
            else:
                if all(f.exists() for f in [t1_file, t1ce_file, t2_file, flair_file]):
                    file_list.append({
                        "image": [str(t1_file), str(t1ce_file), str(t2_file), str(flair_file)],
                    })
    
    logger.info(f"Found {len(file_list)} cases in {data_dir}")
    return file_list


def main():
    """Main function to run the brain tumor segmentation pipeline."""
    
    # Print MONAI configuration
    print_config()
    
    # Configuration optimized for Google Colab A100
    config = {
        "seed": 42,
        "max_epochs": 100,
        "batch_size": 2,  # Optimized for A100 memory
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "roi_size": (128, 128, 128),  # Adaptive sizing will handle smaller volumes
        "sw_batch_size": 4,
        "spacing": (1.0, 1.0, 1.0),
        "num_classes": 4,  # Background, NCR/NET, ED, ET
        "model_name": "unet",  # Options: unet, segresnet, unetr
        "use_amp": True,  # Automatic Mixed Precision for A100
        "cache_rate": 0.5,
        "num_workers": 4,
        "log_interval": 50,
    }
    
    # Setup data directories
    data_root = "/content/data"  # Default Colab path
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    
    segmentation = BrainTumorSegmentation(config)
    train_dir, test_dir = segmentation.setup_data_directories(data_root)
    
    # Create file lists
    train_files = create_file_list(train_dir, "train")
    val_files = create_file_list(test_dir, "test")  # Using test as validation
    
    if not train_files:
        logger.error("No training files found. Please check your data directory structure.")
        return
    
    # Start training
    logger.info("Starting training...")
    segmentation.train(train_files, val_files)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()