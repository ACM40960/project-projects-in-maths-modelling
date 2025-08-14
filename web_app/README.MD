# Web Application

This folder contains a **Streamlit web application** that provides an interactive interface for exploring our wildlife camera-trap classification project. The app demonstrates the complete two-stage pipeline with live inference capabilities and comprehensive project documentation.

## Overview

The web app serves multiple purposes:
- **Project showcase**: Interactive presentation of methodology, results, and literature context
- **Live inference**: Real-time testing of the MegaDetector v6 + ConvNeXt pipeline on verification images
- **Results exploration**: Detailed metrics, confusion matrices, and performance comparisons
- **Educational tool**: Visual explanations of model architectures and domain shift challenges

## Quick Start

### Local Development
```bash
# From project root
streamlit run web_app/app.py
```

### Live Demo
🌐 **Deployed version**: [wildlife-cameratrap.streamlit.app](https://wildlife-cameratrap.streamlit.app/)

## File Structure

```
web_app/
├── main.py                 # Main Streamlit application
├── utils/
│   ├── predict.py         # ONNX inference pipeline
│   └── diagrams.py         # Model Architecture export script
├── assets/               # All figures and diagrams for poster and readme files
│   ├── figs/             # Model architecture diagrams
└── README.md             # This file
```

## Core Components

### `app.py` - Main Application

**Two-tab interface:**

#### Main Tab
- **Project Overview**: Problem statement, approach, and key results
- **Dataset Description**: CCT20 benchmark details, challenges, annotation format
- **Methodology**: Single-stage baselines, progressive augmentation, two-stage pipeline
- **Architecture Diagrams**: Visual explanations of MegaDetector v6 and ConvNeXt-Small
- **Results Dashboard**: Performance metrics, confusion matrices, domain gap analysis
- **Literature Context**: Related work and academic positioning

#### Test Tab
- **Live Inference**: Interactive species prediction on verification images
- **Ground Truth Comparison**: Side-by-side visualization of predictions vs annotations
- **Metrics Dashboard**: Per-class performance breakdown for CIS and TRANS splits
- **Species Selection**: Choose specific animals to test model performance

**Key Features:**
- **Responsive design** with mobile-friendly layout
- **Custom CSS styling** with glassmorphism effects and hover animations
- **Cached data loading** for optimal performance
- **Error handling** for missing models or data files

### `utils/predict.py` - Inference Pipeline

**WildlifePredictor Class:**
```python
predictor = WildlifePredictor(
    md_model_path="models/megadetectorv6.onnx",
    cls_model_path="models/convnext_classifier.onnx"
)

predictions = predictor.predict(image_path)
```

**Two-stage inference workflow:**
1. **Stage 1 - MegaDetector v6**: Binary animal/vehicle detection (threshold: 0.35)
2. **Stage 2 - ConvNeXt Classifier**: 13-species classification (threshold: 0.55)

**Technical implementation:**
- **ONNX Runtime** for cross-platform model inference
- **GPU acceleration** when available (CUDA → CPU fallback)
- **Preprocessing pipelines** matching training configurations
- **Confidence thresholds** optimized for recall-precision balance

**Output format:**
```python
[
    {
        "bbox": [x1, y1, x2, y2],
        "category": "bobcat",
        "category_id": 6,
        "conf": 0.847
    }
]
```

### `diagrams.py` - Architecture Visualization (Utility)

**Purpose**: Extract and document ONNX model architectures for educational presentation

**Usage:**
```python
export_onnx_diagram("models/convnext_classifier.onnx", "assets/figs/convnext_arch.txt")
```

**Output**: Text-based layer breakdown showing:
- Layer sequence and operations
- Node names and types
- Total model complexity

*Note: This script was used to generate architecture documentation shown in the app's methodology section.*

## Dependencies

**Core requirements:**
- `streamlit` - Web application framework
- `onnxruntime` - Model inference engine
- `opencv-python` - Image processing
- `matplotlib` - Visualization and bbox drawing
- `pandas` - Data handling for metrics tables
- `numpy` - Numerical operations

**Optional GPU support:**
- `onnxruntime-gpu` - CUDA acceleration

## Configuration

### Model Paths
Models are automatically loaded from:
```python
MODELS_DIR = PROJECT_ROOT / "models"
- megadetectorv6.onnx      # 97MB - YOLOv9-Compact detector
- convnext_classifier.onnx # 85MB - Species classifier
```

### Data Paths
```python
DATA_DIR = PROJECT_ROOT / "data"
- verification/                    # Verification image subset
- preprocessed/annotations/cleaned/verify_test.json  # Ground truth
```

### Asset Paths
```python
ASSETS = web_app / "assets"
- figs/           # Architecture diagrams, results plots
- background.jpg  # App background
- example_*.jpg   # Demo images
```

## Features

### Interactive Inference
- **Species selection** from 13 wildlife classes
- **Random image sampling** from verification set
- **Real-time prediction** with bbox visualization
- **Confidence scoring** for both detection and classification stages

### Visual Design
- **Dark theme** with camera-trap background imagery
- **Glassmorphism effects** with semi-transparent containers
- **Hover animations** on interactive elements
- **Responsive layout** adapting to screen size
- **Professional typography** with proper hierarchy

### Performance Optimization
- **Cached model loading** (`@st.cache_resource`)
- **Cached data parsing** (`@st.cache_data`)
- **Efficient image handling** with OpenCV
- **Memory management** for large models

## Development

### Adding New Features

**New prediction models:**
1. Add ONNX model to `models/` directory
2. Update `WildlifePredictor` class in `predict.py`
3. Modify confidence thresholds and preprocessing as needed

**Additional visualizations:**
1. Generate plots/diagrams and save to `assets/figs/`
2. Reference in `app.py` using `fig()` helper function
3. Add explanatory text in the methodology section

**New datasets:**
1. Update verification JSON paths in `app.py`
2. Modify category mappings in `predict.py`
3. Update species selection options

### Styling Customization

The app uses custom CSS for visual appeal:
- **Background**: Configurable via `BG_IMAGE_PATH`
- **Color scheme**: Modify CSS variables in the `<style>` block
- **Animations**: Adjust hover effects and transitions
- **Responsive breakpoints**: Update `@media` queries for different screen sizes

## Deployment

### Streamlit Cloud
The app is deployed on Streamlit Cloud with automatic updates from the repository:
- **URL**: [wildlife-cameratrap.streamlit.app](https://wildlife-cameratrap.streamlit.app/)
- **Requirements**: Managed via `requirements.txt`
- **Secrets**: Model files handled via Git LFS

### Local Deployment
For development or custom deployment:
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure models are available
git lfs pull

# Run locally
streamlit run web_app/app.py --server.port 8501
```

## Troubleshooting

**Common issues:**

1. **Models not loading**
   - Ensure Git LFS is installed: `git lfs install`
   - Pull large files: `git lfs pull`
   - Check model file sizes (should be >50MB each)

2. **Images not displaying**
   - Verify asset paths in `assets/` directory
   - Check file permissions and formats (JPG/PNG)

3. **Prediction errors**
   - Confirm ONNX runtime installation
   - Test with CPU provider if GPU fails
   - Validate input image formats and sizes

4. **Performance issues**
   - Clear Streamlit cache: `streamlit cache clear`
   - Restart application
   - Check available memory for model loading

## Future Enhancements

**Potential improvements:**
- **Batch processing** for multiple images
- **Model comparison** interface (A/B testing)
- **Export functionality** for predictions (JSON/CSV)
- **Advanced filtering** by confidence, species, or location
- **Performance analytics** (inference time, memory usage)
- **User feedback collection** for model improvement
