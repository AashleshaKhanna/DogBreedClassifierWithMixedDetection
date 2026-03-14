"""
Dog detection module using YOLOv5.
Stage 1 of the pipeline: Detect and localize dogs in images.
"""

import torch
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List
import warnings


class DogDetector:
    """Wrapper for YOLOv5 dog detection."""
    
    def __init__(
        self,
        model_name: str = 'yolov5s',
        confidence_threshold: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize dog detector.
        
        Args:
            model_name: YOLOv5 model variant ('yolov5s', 'yolov5m', etc.)
            confidence_threshold: Minimum confidence for detection
            device: Device to run inference on
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load YOLOv5 model from torch hub
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            warnings.warn(f"Failed to load YOLOv5: {e}. Using fallback mode.")
            self.model = None
    
    def detect(
        self,
        image: Image.Image,
        return_all: bool = False
    ) -> Tuple[Optional[Image.Image], float, Optional[List]]:
        """
        Detect dog in image and return cropped region.
        
        Args:
            image: PIL Image
            return_all: If True, return all dog detections
        
        Returns:
            cropped_image: Cropped dog region (None if no detection)
            confidence: Detection confidence score
            all_detections: List of all detections if return_all=True
        """
        if self.model is None:
            # Fallback: return original image
            return image, 1.0, None
        
        # Run detection
        results = self.model(image)
        
        # Filter for dog class (class 16 in COCO)
        detections = results.pandas().xyxy[0]
        dog_detections = detections[detections['name'] == 'dog']
        
        if len(dog_detections) == 0:
            return None, 0.0, None
        
        # Filter by confidence
        dog_detections = dog_detections[dog_detections['confidence'] >= self.confidence_threshold]
        
        if len(dog_detections) == 0:
            return None, 0.0, None
        
        if return_all:
            all_crops = []
            for _, det in dog_detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                crop = image.crop((x1, y1, x2, y2))
                all_crops.append((crop, det['confidence']))
            return None, 0.0, all_crops
        
        # Get highest confidence detection
        best_detection = dog_detections.iloc[0]
        confidence = best_detection['confidence']
        
        # Crop image
        x1 = int(best_detection['xmin'])
        y1 = int(best_detection['ymin'])
        x2 = int(best_detection['xmax'])
        y2 = int(best_detection['ymax'])
        
        cropped_image = image.crop((x1, y1, x2, y2))
        
        return cropped_image, confidence, None
    
    def preprocess_image(
        self,
        image: Image.Image,
        target_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[Optional[Image.Image], dict]:
        """
        Detect dog and prepare image for classification.
        
        Args:
            image: Input PIL Image
            target_size: Target size for classification model
        
        Returns:
            processed_image: Cropped and resized image (None if no dog)
            metadata: Dictionary with detection info
        """
        cropped, confidence, _ = self.detect(image)
        
        metadata = {
            'detected': cropped is not None,
            'confidence': confidence,
            'status': 'success' if cropped is not None else 'no_dog_detected'
        }
        
        if cropped is None:
            if confidence < 0.3:
                metadata['status'] = 'uncertain_non_dog'
            return None, metadata
        
        # Resize for classification
        processed = cropped.resize(target_size, Image.BILINEAR)
        
        return processed, metadata
    
    def handle_multiple_dogs(
        self,
        image: Image.Image
    ) -> List[Tuple[Image.Image, float]]:
        """
        Detect and return all dogs in image.
        
        Args:
            image: Input PIL Image
        
        Returns:
            List of (cropped_image, confidence) tuples
        """
        _, _, all_detections = self.detect(image, return_all=True)
        
        if all_detections is None:
            return []
        
        return all_detections


def test_detector():
    """Test dog detector on sample image."""
    detector = DogDetector()
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color='white')
    
    processed, metadata = detector.preprocess_image(dummy_image)
    
    print(f"Detection metadata: {metadata}")
    print(f"Processed image: {processed}")


if __name__ == '__main__':
    test_detector()
