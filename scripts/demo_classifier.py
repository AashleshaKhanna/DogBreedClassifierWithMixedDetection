"""
Demo script for testing the dog breed classifier.
Works with trained models to predict breeds from images.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse

# Import your models
from src.models.baseline import BaselineModel
from src.data.augmentation import get_val_transforms


def load_breed_names(breed_file='data/processed/breed_names_combined.txt'):
    """Load breed names from file."""
    with open(breed_file, 'r') as f:
        breeds = [line.strip() for line in f.readlines()]
    return breeds


def load_model(model_path, model_type='baseline', num_classes=93):
    """Load trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'baseline':
        model = BaselineModel(num_classes=num_classes)
    else:
        # Add primary model loading here when ready
        raise NotImplementedError("Primary model loading not yet implemented")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, device


def predict_image(image_path, model, device, breed_names, transform):
    """Predict breed for a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_breed = breed_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    
    # Get top 5 predictions
    top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
    top5_breeds = [(breed_names[idx.item()], prob.item() * 100) 
                   for idx, prob in zip(top5_idx[0], top5_prob[0])]
    
    return predicted_breed, confidence_pct, top5_breeds


def main():
    parser = argparse.ArgumentParser(description='Dog Breed Classifier Demo')
    parser.add_argument('--image', type=str, required=True, help='Path to dog image')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='baseline', 
                       choices=['baseline', 'primary'], help='Model type')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Confidence threshold for unknown detection')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DOG BREED CLASSIFIER DEMO")
    print("="*70)
    
    # Load breed names
    print("\n📋 Loading breed names...")
    breed_names = load_breed_names()
    print(f"✓ Loaded {len(breed_names)} breed names")
    
    # Load model
    print(f"\n🧠 Loading {args.model_type} model...")
    model, device = load_model(args.model, args.model_type, len(breed_names))
    print(f"✓ Model loaded on {device}")
    
    # Get transforms
    transform = get_val_transforms()
    
    # Predict
    print(f"\n🔍 Analyzing image: {args.image}")
    predicted_breed, confidence, top5 = predict_image(
        args.image, model, device, breed_names, transform
    )
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    if confidence >= args.threshold * 100:
        print(f"\n✅ PREDICTED BREED: {predicted_breed.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence:.2f}%")
    else:
        print(f"\n⚠️  UNKNOWN/MIXED BREED")
        print(f"   Top prediction: {predicted_breed.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence:.2f}% (below threshold {args.threshold*100:.0f}%)")
    
    print(f"\n📊 Top 5 Predictions:")
    for i, (breed, prob) in enumerate(top5, 1):
        print(f"   {i}. {breed.replace('_', ' ').title()}: {prob:.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
