"""
augment_cct.py

Comprehensive augmentation pipeline specifically designed for camera trap datasets.
Supports both object detection (YOLO) and classification tasks with modular design.

Usage:
    from camera_trap_augmentations import get_augmentation_pipeline
    
    # For detection
    aug = get_augmentation_pipeline(task='detection', mode='train')
    
    # For classification  
    aug = get_augmentation_pipeline(task='classification', mode='train')
"""

import albumentations as A
import cv2
import numpy as np


class CameraTrapAugmentations:
    """
    Camera trap specific augmentation pipeline addressing key challenges:
    - Poor illumination (day/night variations)
    - Motion blur from fast-moving animals
    - Small ROI (distant/small animals)
    - Occlusion from vegetation
    - Perspective distortion (close animals)
    - Weather conditions (rain, dust, fog)
    - Camera malfunctions and sensor variations
    - Temporal/seasonal changes
    - Background variability
    """
    
    def __init__(self):
        self.augmentation_registry = {
            'illumination': self._get_illumination_augs,
            'geometric': self._get_geometric_augs,
            'weather_noise': self._get_weather_noise_augs,
            'occlusion': self._get_occlusion_augs,
            'sensor_variation': self._get_sensor_variation_augs,
            'motion_blur': self._get_motion_blur_augs,
            'perspective': self._get_perspective_augs,
            'seasonal': self._get_seasonal_augs,
        }
    
    def _get_illumination_augs(self, intensity: float = 1.0) -> list:
        """
        Handle poor illumination conditions common in camera traps.
        - Night images with poor visibility
        - Overexposed daylight images  
        - Harsh shadows and highlights
        - IR/thermal imaging variations
        """

        return [
            # Primary illumination adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.7 * intensity, 
                contrast_limit=0.5 * intensity, 
                p=0.7
            ),
            
            # Histogram equalization for low-light enhancement
            # Enhanced histogram equalization
            A.OneOf([
                A.CLAHE(
                    clip_limit=max(1.0, 5.0 * intensity),  # Increased from 3.0
                    tile_grid_size=(8, 8),
                    p=1.0 
                ),
                A.Equalize(p=1.0),
                # NEW: Adaptive histogram equalization
                A.CLAHE(
                    clip_limit=max(1.0, 2.0 * intensity),
                    tile_grid_size=(4, 4),  # Smaller tiles for local enhancement
                    p=1.0
                )
            ], p=0.5 * intensity),  # Increased from 0.3
            
            
            # Gamma correction for exposure variations
            A.RandomGamma(
                gamma_limit=(60, 140), 
                p=0.5 * intensity
            ),
            
            # Color space illumination changes
            A.HueSaturationValue(
                hue_shift_limit=int(20 * intensity),
                sat_shift_limit=int(35 * intensity), 
                val_shift_limit=int(40 * intensity),
                p=0.5
            ),
        ]
    
    def _get_geometric_augs(self, intensity: float = 1.0, task: str = 'detection') -> list:
        """
        Handle geometric variations while preserving camera trap characteristics.
        - Scale variations for animals at different distances
        - Minor rotations (cameras usually fixed)
        - Translations for edge cases
        """
        # Camera traps have limited geometric variation compared to general datasets
        return [
            # Horizontal flip is common (animal can approach from either side)
            A.HorizontalFlip(p=0.5),
        
            
            # Combined geometric transformations
            A.Affine(
            translate_percent={"x": (-0.03 * intensity, 0.03 * intensity), "y": (-0.03 * intensity, 0.03 * intensity)},
            scale=(1.0 - 0.05 * intensity, 1.0 + 0.05 * intensity),
            rotate=(-3 * intensity, 3 * intensity),
            border_mode=cv2.BORDER_REFLECT,
            p=0.3 * intensity
            ),
            
            # Scale variations for distant/close animals
            A.RandomScale(
                scale_limit=0.15 * intensity, 
                p=0.4 * intensity
            ),

            A.RandomResizedCrop(
                size=(640 ,640),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.2 * intensity
            ),
        ]
    
    def _get_weather_noise_augs(self, intensity: float = 1.0) -> list:
        """
        Simulate weather conditions and camera sensor noise.
        - Rain droplets on lens
        - Dust and debris
        - Fog and atmospheric conditions
        - Sensor noise in low light
        """
        return [
            # Sensor noise (common in low-light conditions)
            A.OneOf([
                A.GaussNoise(
                    std_range=(0.3 * intensity, 0.5 * intensity),
                    p=1.0
                ),
                A.ISONoise(
                    color_shift=(0.01*intensity, 0.08 * intensity), 
                    intensity=(0.1*intensity, 0.7 * intensity),
                    p=1.0
                ),
            ], p=0.3),
            
            # Weather-related noise
            A.OneOf([
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1), 
                    per_channel=True
                ),
                A.RandomRain(
                    brightness_coefficient=0.9, 
                    drop_width=1, 
                    blur_value=2, 
                    rain_type="default"
                ),
                A.RandomFog(
                    fog_coef_range=(0.1*intensity, 0.3*intensity), 
                    alpha_coef=0.1
                ),
            ], p=0.15 * intensity),
        ]
    
    def _get_motion_blur_augs(self, intensity: float = 1.0) -> list:
        """
        Simulate motion blur from fast-moving animals.
        - Different blur patterns for different animal behaviors
        - Directional blur for running animals
        """
        return [
            A.OneOf([
                # General motion blur
                A.MotionBlur(
                    blur_limit=(max(3,int(12 * intensity)),max(3,int(15 * intensity))), 
                    allow_shifted=True,
                    p=1.0
                ),
                # Radial blur for close animals
                A.MedianBlur(
                    blur_limit=2 * int(max(1, 3 * intensity)) + 1,
                    p=1.0
                ),
                # Gaussian blur for atmospheric effects
                A.GaussianBlur(
                    blur_limit=2 * int(max(1, 3 * intensity)) + 1,
                    sigma_limit=(2.0 * intensity, 6.0 * intensity),
                    p=1.0
                ),
                A.Defocus(
                    radius=(1, max(1, int(4 * intensity))),
                    alias_blur=(0.1, 0.5),
                    p=1.0
                ),
            ], p=0.5),
        ]
    
    def _get_occlusion_augs(self, intensity: float = 1.0) -> list:
        """
        Simulate occlusion patterns common in camera traps.
        - Vegetation growing in front of camera
        - Partial occlusion by trees/branches
        - Edge occlusion when animals are partially out of frame
        """
        return [
            A.OneOf([
                # Small vegetation patches
                A.CoarseDropout(
                    num_holes_range=(2, max(2, int(12 * intensity))),
                    hole_height_range=(8, max(8, int(50 * intensity))),
                    hole_width_range=(8, max(8, int(50 * intensity))),
                    fill=0,
                    p=1.0
                ),
                
                # Large vegetation/branch occlusion
                A.CoarseDropout(
                    num_holes_range=(1, max(1, int(3 * intensity))),
                    hole_height_range=(10, max(10, int(120 * intensity))),
                    hole_width_range=(10, max(10, int(120 * intensity))),
                    fill=0,
                    p=1.0
                ),

                
                # Grid-like occlusion (fence, vegetation pattern)
                A.GridDistortion(
                    num_steps=8,
                    distort_limit=0.2 * intensity,
                    border_mode=cv2.BORDER_REFLECT,
                    p=1.0
                ),
            ], p=0.4),
        ]
    
    def _get_perspective_augs(self, intensity: float = 1.0) -> list:
        """
        Handle perspective distortion when animals are very close or far.
        - Close-up perspective changes
        """
        return [
                
                # Perspective transformation
                A.Perspective(
                    scale=(0.01, 0.05 * intensity),
                    keep_size=True,
                    p=0.4*intensity
                )
        ]
    
    def _get_sensor_variation_augs(self, intensity: float = 1.0) -> list:
        """
        Simulate camera sensor variations and malfunctions.
        - Different camera models/sensors
        - Color calibration differences
        - Sensor degradation over time
        """
        return [
            # Color channel variations
            A.RGBShift(
                r_shift_limit=int(30 * intensity),
                g_shift_limit=int(30 * intensity), 
                b_shift_limit=int(30 * intensity),
                p=0.3
            ),
            
            
            # Color space distortions
            A.OneOf([
                A.FancyPCA(alpha=0.15 * intensity),
                A.ColorJitter(
                    brightness=0.3 * intensity,
                    contrast=0.3 * intensity, 
                    saturation=0.3 * intensity,
                    hue=0.08 * intensity
                ),
            ], p=0.3),

    

            
        ]
    
    def _get_seasonal_augs(self, intensity: float = 1.0) -> list:
        """
        Simulate seasonal and temporal changes in the environment.
        - Autumn/dry season color shifts
        - Seasonal lighting changes
        - Background vegetation changes
        """
        return [
            A.OneOf([
                # Autumn/dry season simulation
                A.ToSepia(p=1.0),
                

                # Different seasonal lighting
                A.Emboss(alpha=(0.2, 0.6), strength=(0.5, 1.2)),
                
            ], p=0.15 * intensity),
        ]
    
    def _get_advanced_augs(self, intensity: float = 1.0) -> list:
        """NEW: Advanced augmentations for better robustness"""
        return [
            # Sharpening/unsharpening
            A.OneOf([
                A.Sharpen(
                    alpha=(0.2, 0.5),
                    lightness=(0.5, 1.0),
                    p=1.0
                ),
                A.UnsharpMask(
                    blur_limit=(3, 7),
                    sigma_limit=(0.5, 2.0),
                    alpha=(0.2, 0.5),
                    threshold=10,
                    p=1.0
                ),
            ], p=0.25 * intensity),
            
            
            # Downscale simulation (low resolution cameras)
            A.Downscale(
                scale_range=(0.5, 0.8),
                p=0.3 * intensity
            ),
        ]
    

def get_detection_augmentation(mode='train', img_size=640, intensity=1.0, epoch=0, max_epochs=100):
    """
    Get augmentation pipeline for detection tasks
    """

    cct_augs = CameraTrapAugmentations()

    augmentations=[]
    
    if mode == 'train':

        # Calculate epoch-based intensity (if you want to override the passed intensity)
        # epoch_intensity = min(1.0, 0.5 + 0.5 * (epoch / (max_epochs * 0.3)))
        
        augmentations.extend ( [


            # All your custom augmentations
            *cct_augs._get_illumination_augs(intensity),
            *cct_augs._get_geometric_augs(intensity, task='detection'),
            *cct_augs._get_weather_noise_augs(intensity),
            *cct_augs._get_motion_blur_augs(intensity),
            *cct_augs._get_occlusion_augs(intensity),
            *cct_augs._get_sensor_variation_augs(intensity*0.4),
            *cct_augs._get_perspective_augs(intensity),
            *cct_augs._get_seasonal_augs(intensity*0.3),
            *cct_augs._get_advanced_augs(intensity*0.7),


             A.Resize(height=img_size, width=img_size, p=1.0),
        ])

        
        
        pipeline = A.Compose(
            augmentations,
            bbox_params=A.BboxParams(format='yolo', label_fields=['cls'],min_visibility=0.2,filter_invalid_bboxes=True)
        )
    else:
        # Validation/test - minimal augmentations with proper image size
        pipeline = A.Compose([
            A.Resize(height=img_size, width=img_size, p=1.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['cls']))
    
    return pipeline

