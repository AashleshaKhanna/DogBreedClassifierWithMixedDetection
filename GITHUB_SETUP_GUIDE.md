# GitHub Setup Guide

## 🎯 Goal
Push your Dog Breed Classifier project to: https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection

## 📋 What Will Be Pushed

### ✅ Included (Important Project Files):
- `src/` - All source code (models, training, data processing)
- `scripts/` - Training, demo, and utility scripts
- `configs/` - Configuration files
- `data/processed/` - CSV annotations and metadata (small files)
- `progress_report_figures/` - Report figures and diagrams
- `README.md` - Project documentation
- `requirements.txt` - Dependencies
- `APS360_Proposal.pdf` - Project proposal
- `pipeline_diagram.png` - Architecture diagram
- Documentation files (guides, setup instructions)

### ❌ Excluded (Large/Unnecessary Files):
- `data/raw/` - Raw dataset images (too large for GitHub)
- `src/data/Kaggle_dataset/` - Dataset images
- `src/data/Stanford_dataset/` - Dataset images
- `checkpoints/` - Model weights (too large)
- `Lab*.pdf` - Course materials
- `Week*.pdf` - Lecture slides
- `.kiro/` - Kiro specs (optional)
- `DogBreedClassifierWithMixedDetection/` - Nested repo

## 🚀 Step-by-Step Instructions

### Step 1: Initialize Git (if not already done)

```bash
# Check if git is initialized
git status

# If not initialized, run:
git init
```

### Step 2: Add Remote Repository

```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection.git

# Verify remote was added
git remote -v
```

### Step 3: Stage All Project Files

```bash
# Add all files (respecting .gitignore)
git add .

# Check what will be committed
git status
```

### Step 4: Create Initial Commit

```bash
# Commit with descriptive message
git commit -m "Initial commit: Dog Breed Classifier with complete infrastructure

- Added source code (models, training, data processing)
- Added training scripts (baseline and primary models)
- Added demo and testing scripts
- Added configuration files
- Added progress report figures and documentation
- Added dataset processing scripts
- Complete project infrastructure ready for training"
```

### Step 5: Push to GitHub

```bash
# Push to main branch
git push -u origin main

# If main branch doesn't exist, try master:
git push -u origin master

# Or create and push to main:
git branch -M main
git push -u origin main
```

## 🔧 If Repository Already Has Content

If your GitHub repo already has files (like a README), you'll need to pull first:

```bash
# Pull existing content
git pull origin main --allow-unrelated-histories

# Resolve any conflicts if needed
# Then push
git push -u origin main
```

## 📊 Verify Upload

After pushing, check on GitHub:
1. Go to: https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection
2. Verify you see:
   - `src/` folder with all code
   - `scripts/` folder with training scripts
   - `README.md` with project description
   - `requirements.txt` with dependencies
   - `progress_report_figures/` with diagrams

## 🎨 Optional: Add GitHub README Badges

Add these to the top of your README.md:

```markdown
# Dog Breed Classifier with Mixed Detection

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

[Rest of your README...]
```

## 📝 Future Updates

When you make changes:

```bash
# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to GitHub
git push
```

## 🔍 Check What Will Be Pushed

Before committing, see what files will be included:

```bash
# See all files that will be added
git add -n .

# See status
git status

# See what's ignored
git status --ignored
```

## ⚠️ Important Notes

1. **Large Files**: Datasets are excluded (too large for GitHub)
   - Users will need to download datasets separately
   - Document this in README

2. **Model Checkpoints**: Excluded (too large)
   - Users will need to train models themselves
   - Or you can share via Google Drive/Dropbox

3. **Sensitive Info**: Make sure no API keys or passwords are in code

4. **Course Materials**: Lab PDFs and lecture slides are excluded

## 🎯 Quick Command Summary

```bash
# One-time setup
git init
git remote add origin https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection.git

# Push project
git add .
git commit -m "Initial commit: Complete project infrastructure"
git branch -M main
git push -u origin main

# Future updates
git add .
git commit -m "Update: description"
git push
```

## 🆘 Troubleshooting

### "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/AashleshaKhanna/DogBreedClassifierWithMixedDetection.git
```

### "failed to push some refs"
```bash
git pull origin main --rebase
git push -u origin main
```

### "large files detected"
```bash
# Check file sizes
git ls-files -s | awk '{print $4, $2}' | sort -n -r | head -20

# Remove large file from staging
git rm --cached path/to/large/file
```

### "Permission denied"
Make sure you're authenticated with GitHub:
```bash
# Using GitHub CLI
gh auth login

# Or set up SSH keys
# Or use personal access token
```

## ✅ Verification Checklist

After pushing, verify on GitHub:
- [ ] README.md displays correctly
- [ ] Source code is visible in `src/`
- [ ] Scripts are in `scripts/`
- [ ] Figures are in `progress_report_figures/`
- [ ] requirements.txt is present
- [ ] No large dataset files included
- [ ] No course materials included
- [ ] .gitignore is working correctly

---

**Ready to push? Run the commands in Step-by-Step Instructions!**
