# Dog Breed Classifier With Mixed-Breed Detection
This project goes beyond basic CNN image classification by addressing:
- Real-world robustness challenges (blur, occlusion, lighting variations)
- Open-set recognition (detecting out-of-distribution inputs)
- Multi-label classification for mixed breed identification
- Confidence calibration for reliable predictions
- Multi-stage pipeline design (detection → classification → mixed breed analysis → calibration)

# Problem Statement
Standard dog breed classifiers fail in real-world scenarios with messy phone photos (weird angles, partial dogs, low light, cluttered backgrounds) and often hallucinate breed predictions for non-dog images or mixed breeds. This project develops a robust dog breed identification system that handles real-world photo conditions, intelligently rejects uncertain predictions with an "unknown" classification, and identifies mixed breeds by predicting multiple contributing breeds with confidence scores.
