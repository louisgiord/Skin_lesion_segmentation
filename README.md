# Skin Lesion Segmentation

Automated segmentation pipeline for dermatological skin lesion images using Otsu's method with preprocessing and postprocessing steps.

## Overview

This project implements a complete image processing pipeline to segment skin lesions from dermatological images. The pipeline consists of three main stages:

1. **Preprocessing**: Black frame removal and hair removal (dull razor)
2. **Segmentation**: Otsu's thresholding method
3. **Postprocessing**: Morphological operations and connected component analysis

## Project Structure

```
├── src/                      # Source code
│   ├── preprocessing/        # Black frame & hair removal
│   ├── segmentation/         # Otsu methods
│   ├── postprocessing/       # Morphological operations
│   └── utils/                # Display & evaluation tools
├── tests/                    # Test scripts
├── experiments/              # Experimental code
├── data/                     # Image datasets
├── report.ipynb              # Main report notebook
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install opencv-python numpy matplotlib scikit-image jupyter
```

## Usage

Open and run `report.ipynb` to see the complete pipeline and results. 



