"""
Mock demo to show what the classifier interface will look like.
Works WITHOUT trained models - uses random predictions for demonstration.
"""

import random
from pathlib import Path
import argparse


def load_breed_names(breed_file='data/processed/breed_names_combined.txt'):
    """Load breed names from file."""
    try:
        with open(breed_file, 'r') as f:
            breeds = [line.strip() for line in f.readlines()]
        return breeds
    except FileNotFoundError:
        # Fallback to some common breeds
        return ['golden_retriever', 'german_shepherd', 'labrador_retriever', 
                'beagle', 'bulldog', 'poodle', 'rottweiler', 'yorkshire_terrier']


def mock_predict(image_path, breed_names, scenario='high_confidence'):
    """Generate mock predictions for demonstration."""
    
    # Different scenarios for demonstration
    if scenario == 'high_confidence':
        # Simulate a clear purebred dog
        predicted_breed = random.choice(breed_names)
        confidence = random.uniform(85, 98)
        
        # Generate top 5
        top5 = [(predicted_breed, confidence)]
        remaining = [b for b in breed_names if b != predicted_breed]
        for _ in range(4):
            breed = random.choice(remaining)
            remaining.remove(breed)
            top5.append((breed, random.uniform(0.5, confidence - 10)))
        
    elif scenario == 'low_confidence':
        # Simulate a mixed breed or unclear case
        predicted_breed = random.choice(breed_names)
        confidence = random.uniform(25, 45)
        
        # Generate top 5 with similar probabilities
        top5 = [(predicted_breed, confidence)]
        remaining = [b for b in breed_names if b != predicted_breed]
        for _ in range(4):
            breed = random.choice(remaining)
            remaining.remove(breed)
            top5.append((breed, random.uniform(15, confidence - 5)))
    
    else:  # random
        predicted_breed = random.choice(breed_names)
        confidence = random.uniform(20, 95)
        top5 = [(predicted_breed, confidence)]
        remaining = [b for b in breed_names if b != predicted_breed]
        for _ in range(4):
            breed = random.choice(remaining)
            remaining.remove(breed)
            top5.append((breed, random.uniform(1, confidence - 5)))
    
    return predicted_breed, confidence, top5


def main():
    parser = argparse.ArgumentParser(description='Mock Dog Breed Classifier Demo')
    parser.add_argument('--image', type=str, required=True, 
                       help='Path to dog image (for display only)')
    parser.add_argument('--scenario', type=str, default='random',
                       choices=['high_confidence', 'low_confidence', 'random'],
                       help='Prediction scenario to simulate')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for unknown detection')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("DOG BREED CLASSIFIER - MOCK DEMO")
    print("(Using simulated predictions - train models for real predictions)")
    print("="*70)
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"\n❌ Error: Image not found: {args.image}")
        print("   (This is just a demo, so we'll continue anyway...)")
    else:
        print(f"\n✓ Image found: {args.image}")
    
    # Load breed names
    print("\n📋 Loading breed names...")
    breed_names = load_breed_names()
    print(f"✓ Loaded {len(breed_names)} breed names")
    
    # Generate mock prediction
    print(f"\n🔍 Analyzing image (MOCK MODE - scenario: {args.scenario})...")
    predicted_breed, confidence, top5 = mock_predict(
        args.image, breed_names, args.scenario
    )
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS (SIMULATED)")
    print("="*70)
    
    if confidence >= args.threshold * 100:
        print(f"\n✅ PREDICTED BREED: {predicted_breed.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence:.2f}%")
        print(f"   Status: HIGH CONFIDENCE - Pure breed detected")
    else:
        print(f"\n⚠️  UNKNOWN/MIXED BREED")
        print(f"   Top prediction: {predicted_breed.replace('_', ' ').title()}")
        print(f"   Confidence: {confidence:.2f}% (below threshold {args.threshold*100:.0f}%)")
        print(f"   Status: LOW CONFIDENCE - Likely mixed breed or unclear image")
    
    print(f"\n📊 Top 5 Predictions:")
    for i, (breed, prob) in enumerate(top5, 1):
        bar_length = int(prob / 2)
        bar = '█' * bar_length
        print(f"   {i}. {breed.replace('_', ' ').title():<25} {prob:5.2f}% {bar}")
    
    print("\n" + "="*70)
    print("💡 NOTE: These are SIMULATED predictions for demonstration.")
    print("   Train the models using 'python scripts/train_baseline.py'")
    print("   Then use 'python scripts/demo_classifier.py' for real predictions.")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
