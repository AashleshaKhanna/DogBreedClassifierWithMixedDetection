"""
Single image inference script.
Demonstrates full pipeline: detection → classification → calibration.
"""

import torch
from PIL import Image
import argparse
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.detector import DogDetector
from src.models.classifier import BreedClassifier
from src.models.calibration import TemperatureScaling, CalibratedClassifier
from src.data.augmentation import get_val_transforms


def load_model(checkpoint_path: Path, config: dict, device: str):
    """Load trained model with calibration."""
    # Load classifier
    num_classes = config['model']['num_classes']
    classifier = BreedClassifier(num_classes=num_classes, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    elif 'classifier_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device)
    classifier.eval()
    
    # Load temperature scaling if available
    if 'temperature' in checkpoint:
        temp_model = TemperatureScaling()
        temp_model.temperature.data = torch.tensor([checkpoint['temperature']])
        temp_model = temp_model.to(device)
        
        model = CalibratedClassifier(classifier, temp_model)
    else:
        model = classifier
    
    return model


def predict_image(
    image_path: Path,
    model,
    detector: DogDetector,
    transform,
    class_names: list,
    config: dict,
    device: str
):
    """
    Predict breed for a single image.
    
    Args:
        image_path: Path to image
        model: Classification model (with calibration)
        detector: Dog detector
        transform: Image transforms
        class_names: List of breed names
        config: Configuration dict
        device: Device to run on
    
    Returns:
        Dictionary with prediction results
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Stage 1: Dog detection
    processed_image, metadata = detector.preprocess_image(image)
    
    if not metadata['detected']:
        return {
            'status': metadata['status'],
            'message': 'No dog detected in image',
            'detection_confidence': metadata['confidence']
        }
    
    # Stage 2: Breed classification
    input_tensor = transform(processed_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if isinstance(model, CalibratedClassifier):
            # Calibrated probabilities
            probs = model(input_tensor)[0]
        else:
            # Uncalibrated
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0]
    
    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, min(5, len(class_names)))
    
    top5_probs = top5_probs.cpu().numpy()
    top5_indices = top5_indices.cpu().numpy()
    
    # Stage 3: Confidence-based decision
    max_confidence = top5_probs[0]
    
    thresholds = config['thresholds']
    
    if max_confidence >= thresholds['high_confidence']:
        status = 'confident'
        message = f'Predicted breed: {class_names[top5_indices[0]]}'
    elif max_confidence >= thresholds['medium_confidence']:
        status = 'uncertain'
        message = 'Multiple possible breeds (uncertain)'
    else:
        status = 'unknown'
        message = 'Unknown breed (low confidence)'
    
    # Prepare results
    results = {
        'status': status,
        'message': message,
        'detection_confidence': metadata['confidence'],
        'top_prediction': {
            'breed': class_names[top5_indices[0]],
            'confidence': float(top5_probs[0])
        },
        'top5_predictions': [
            {
                'breed': class_names[idx],
                'confidence': float(prob)
            }
            for idx, prob in zip(top5_indices, top5_probs)
        ]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict dog breed from image')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--breeds', type=str, help='Path to breed names file (one per line)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load breed names
    if args.breeds:
        with open(args.breeds, 'r') as f:
            class_names = [line.strip() for line in f]
    else:
        # Dummy names
        class_names = [f'Breed_{i}' for i in range(config['model']['num_classes'])]
    
    print(f"Loaded {len(class_names)} breed classes")
    
    # Initialize detector
    print("Loading dog detector...")
    detector = DogDetector(
        model_name=config['detection']['model'],
        confidence_threshold=config['detection']['confidence_threshold'],
        device=device
    )
    
    # Load classifier
    print("Loading breed classifier...")
    model = load_model(Path(args.model), config, device)
    
    # Get transforms
    transform = get_val_transforms(config['data']['image_size'])
    
    # Predict
    print(f"\nProcessing image: {args.image}")
    results = predict_image(
        Path(args.image),
        model,
        detector,
        transform,
        class_names,
        config,
        device
    )
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Status: {results['status'].upper()}")
    print(f"Message: {results['message']}")
    print(f"Detection Confidence: {results['detection_confidence']:.3f}")
    
    if 'top_prediction' in results:
        print(f"\nTop Prediction:")
        print(f"  Breed: {results['top_prediction']['breed']}")
        print(f"  Confidence: {results['top_prediction']['confidence']:.3f}")
        
        print(f"\nTop-5 Predictions:")
        for i, pred in enumerate(results['top5_predictions'], 1):
            print(f"  {i}. {pred['breed']}: {pred['confidence']:.3f}")
    
    print("="*60)


if __name__ == '__main__':
    main()
