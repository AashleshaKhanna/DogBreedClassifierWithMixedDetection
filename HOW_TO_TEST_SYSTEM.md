# How to Test Your Dog Breed Classifier System

## 📊 Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Dataset | ✅ Ready | 48,338 images, 93 breeds |
| Code Infrastructure | ✅ Ready | All modules implemented |
| Training Scripts | ✅ Ready | Baseline & Primary |
| Models | ❌ Not Trained | Need to run training |
| Testing Scripts | ✅ Ready | Demo scripts created |

## 🎯 Two Ways to Test

### Option 1: Mock Demo (Works NOW - No Training Needed)

This shows you what the interface will look like with simulated predictions.

```bash
# High confidence scenario (purebred dog)
python scripts/mock_demo.py --image "any_image.jpg" --scenario high_confidence

# Low confidence scenario (mixed breed)
python scripts/mock_demo.py --image "any_image.jpg" --scenario low_confidence --threshold 0.6

# Random scenario
python scripts/mock_demo.py --image "any_image.jpg" --scenario random
```

**Output Example** (High Confidence):
```
✅ PREDICTED BREED: Scottish Deerhound
   Confidence: 93.63%
   Status: HIGH CONFIDENCE - Pure breed detected

📊 Top 5 Predictions:
   1. Scottish Deerhound        93.63% ███████████████████████████████
   2. Pekinese                  49.34% ████████████████████████
   3. Airedale                  30.46% ███████████████
```

**Output Example** (Low Confidence):
```
⚠️  UNKNOWN/MIXED BREED
   Top prediction: Gordon Setter
   Confidence: 33.14% (below threshold 60%)
   Status: LOW CONFIDENCE - Likely mixed breed

📊 Top 5 Predictions:
   1. Gordon Setter             33.14% ████████████████
   2. Bluetick                  15.03% ███████
```

### Option 2: Real Demo (After Training Models)

Once you train models, use the real demo:

```bash
python scripts/demo_classifier.py --image path/to/dog.jpg --model checkpoints/baseline/best_model.pth
```

## 🚀 Steps to Get Real Testing Working

### Step 1: Train Baseline Model

```bash
python scripts/train_baseline.py
```

**What happens**:
- Trains ResNet-50 on your 48,338 images
- Takes 2-4 hours with GPU (much longer without)
- Saves model to `checkpoints/baseline/best_model.pth`
- Creates training logs and learning curves

**Expected output**:
```
Epoch 1/30
Train Loss: 3.245, Train Acc: 15.2%
Val Loss: 2.987, Val Acc: 18.5%
✓ Saved checkpoint

Epoch 2/30
Train Loss: 2.876, Train Acc: 22.3%
Val Loss: 2.654, Val Acc: 25.1%
...
```

### Step 2: Test with Real Images

Once training completes:

```bash
# Test single image
python scripts/demo_classifier.py \
    --image path/to/dog.jpg \
    --model checkpoints/baseline/best_model.pth \
    --threshold 0.5
```

### Step 3: Test Multiple Scenarios

```bash
# Test with different images
python scripts/demo_classifier.py --image golden_retriever.jpg --model checkpoints/baseline/best_model.pth
python scripts/demo_classifier.py --image mixed_breed.jpg --model checkpoints/baseline/best_model.pth
python scripts/demo_classifier.py --image cat.jpg --model checkpoints/baseline/best_model.pth
```

## 📁 What You Need for Testing

### For Mock Demo (Available NOW):
- ✅ Any image file (doesn't even need to exist)
- ✅ Python environment with dependencies

### For Real Demo (After Training):
- ✅ Trained model checkpoint
- ✅ Test images (dog photos)
- ✅ GPU recommended (but CPU works)

## 🖼️ Getting Test Images

### Option 1: Use Images from Your Dataset

```bash
# Find some test images
python scripts/demo_classifier.py \
    --image data/processed/test_images/golden_retriever_001.jpg \
    --model checkpoints/baseline/best_model.pth
```

### Option 2: Download from Internet

Download a few dog images and save them to a `test_images/` folder:
- Golden Retriever
- German Shepherd
- Mixed breed dog
- Cat (to test rejection)
- Multiple dogs

### Option 3: Use Your Own Photos

Take photos of dogs you know and test them!

## 📊 Understanding the Output

### High Confidence (>50% default threshold):
```
✅ PREDICTED BREED: Golden Retriever
   Confidence: 94.2%
```
**Meaning**: Model is confident this is a purebred Golden Retriever

### Low Confidence (<50% threshold):
```
⚠️  UNKNOWN/MIXED BREED
   Top prediction: Labrador Retriever
   Confidence: 42.3%
```
**Meaning**: Model thinks it might be a Lab but isn't sure - likely mixed breed

### Top 5 Predictions:
Shows the 5 most likely breeds with probabilities. Useful for:
- Seeing if the correct breed is in top 5
- Understanding model confusion (similar breeds)
- Identifying mixed breed components

## 🎓 For Your Progress Report

**You DON'T need trained models for the progress report!**

What you CAN show:
1. ✅ Dataset statistics (already have)
2. ✅ Architecture diagrams (already have)
3. ✅ Code infrastructure (already have)
4. ✅ Mock demo showing interface
5. ✅ Training plan and approach

What you'll ADD later (final project):
- Actual training results
- Test set accuracy
- Confusion matrices
- Real prediction examples

## 🔧 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch torchvision
```

### "CUDA out of memory"
Reduce batch size in `configs/config.yaml`:
```yaml
batch_size: 16  # Try 8 or 4 if still failing
```

### "Training is too slow"
- Use GPU if available
- Reduce number of epochs for testing
- Use smaller model (baseline instead of primary)

### "Model file not found"
You need to train the model first:
```bash
python scripts/train_baseline.py
```

## 💡 Quick Commands Reference

```bash
# Mock demo (works now)
python scripts/mock_demo.py --image test.jpg --scenario high_confidence

# Train baseline model
python scripts/train_baseline.py

# Real demo (after training)
python scripts/demo_classifier.py --image test.jpg --model checkpoints/baseline/best_model.pth

# Check if model exists
ls checkpoints/baseline/

# View training logs
cat logs/training_log.csv
```

## 🎯 Next Steps

1. **For Progress Report** (Now):
   - Use mock demo to show interface
   - Include dataset statistics
   - Show architecture diagrams
   - Explain approach

2. **For Final Project** (Later):
   - Train baseline model
   - Train primary model
   - Evaluate on test set
   - Generate results visualizations
   - Create final demo

## 📝 Summary

- ✅ **Mock demo works NOW** - shows interface without trained models
- ⏳ **Real demo needs training** - 2-4 hours with GPU
- ✅ **Progress report ready** - don't need trained models yet
- 🎯 **Final project needs training** - plan accordingly

---

**Bottom Line**: You can test the interface RIGHT NOW with the mock demo. For real predictions, you need to train the models first (2-4 hours with GPU).
