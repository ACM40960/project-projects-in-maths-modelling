# Wildlife Camera-Trap Species Classification & Domain-Shift Study

> **Robust wildlife species recognition in challenging camera-trap imagery with focus on domain shift**

![Project Banner](assets/banner_image.jpg) <!-- Suggested: A compelling image showing camera-trap setup or wildlife montage -->

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-green.svg)](https://onnxruntime.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

##  Table of Contents
- [ Project Overview](#-project-overview)
- [ Quick Start](#-quick-start)
- [ Dataset](#-dataset)
- [ Methodology](#-methodology)
- [ Results](#-results)
- [ Running the Project](#-running-the-project)
- [ Project Structure](#-project-structure)
- [ Web Application](#-web-application)
- [ Documentation](#-documentation)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Author](#-author)

##  Project Overview

### The Challenge
Camera-traps generate millions of images annually for wildlife monitoring, but manual sorting is slow and expensive. **Domain shift** — the accuracy drop when moving from *seen* to *unseen* camera locations — remains a critical challenge due to changes in backgrounds, lighting, and species appearance.

### Our Solution
A **two-stage pipeline** that achieves state-of-the-art cross-domain performance:

```
Raw Image →  MegaDetector v6 →  ConvNeXt Classifier → 🏷 Species Label
```

### Key Innovation
- **Stage 1**: Binary animal/vehicle detection with high recall
- **Stage 2**: Fine-grained 13-species classification on cropped regions
- **Result**: 27.5% error reduction on unseen locations vs. best single-stage baseline

![Pipeline Overview](web_app/assets/figs/pipeline.png)

###  Performance Summary

| Metric | CIS-Test (Seen) | TRANS-Test (Unseen) | Domain Gap |
|--------|-----------------|---------------------|------------|
| **F1 Score** | 0.903 | 0.773 | 0.13 |
| **Precision** | 0.904 | 0.750 | 0.15 |
| **Recall** | 0.902 | 0.816 | 0.09 |

*Smallest domain gap achieved among all tested configurations*

##  Quick Start

### Prerequisites
- Python 3.11+
- requirements.txt dependencies

### Installation
```bash
# Install Git LFS (only once on your system)
git lfs install

# Clone the repository
git clone https://github.com/ACM40960/project-projects-in-maths-modelling.git
cd project-projects-in-maths-modelling

# Pull large files (models, assets)
git lfs pull

# Install dependencies
pip install -r requirements.txt

```

### Quick Demo
```bash
# Run the web application
streamlit run web_app/app.py

# Or run evaluation-verification notebook
evaluation.ipynb
verification.ipynb
```

## Dataset

### CCT20 Benchmark Subset
The CCT20 benchmark (Beery et al., ECCV 2018) is a curated subset of the Caltech Camera Traps dataset, containing over 51,000 downsized images (max edge ≤ 1024 px) from 20 camera locations.  
- 13 wildlife species + vehicle  
- CIS: Seen locations (train/val/test)  
- TRANS: Unseen locations (test only), used to evaluate real-world generalization  

Our pipeline is trained only on CIS train/val and evaluated on both CIS-test and TRANS-test.

The full Caltech Camera Traps dataset contains approximately 243,000 images from 140 locations. CCT20 includes 57,864 images with bounding boxes.

### Download Links
| Resource | Size | Link |
|----------|------|------|
| Benchmark Images | 6 GB | [Download](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_all_images_sm.tar.gz) |
| Metadata & Splits | 3 MB | [Download](https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz) |

More details: [Caltech Camera Traps Project Page](https://beerys.github.io/CaltechCameraTraps/)

### Citation
If you use this dataset, please cite:
```bibtex
@inproceedings{DBLP:conf/eccv/BeeryHP18,
  author    = {Sara Beery and Grant Van Horn and Pietro Perona},
  title     = {Recognition in Terra Incognita},
  booktitle = {Proc. ECCV 2018},
  pages     = {472--489},
  year      = {2018},
  doi       = {10.1007/978-3-030-01270-0_28}
}
```

---

### Key Challenges
- **Illumination**: Night-time IR imagery with low contrast
- **Motion blur**: Fast-moving animals with slow shutters  
- **Occlusion**: Vegetation blocking key features
- **Class imbalance**: Long-tailed species distribution
- **Location bias**: Background-dependent features

<p align="center">
  <img src="web_app/assets/example_1.jpg" alt="Sample 1" width="45%"/>
  <img src="web_app/assets/example_2.jpg" alt="Sample 2" width="45%"/>
</p>


##  Methodology

# Experiment

### Architecture Overview

### Stage 1: MegaDetector v6 (YOLOv9-Compact)
- **Purpose**: High-recall detection of animals and vehicles
- **Architecture**: YOLOv9-Compact with FPN and SPPF modules
- **Training**: Binary classification (animal vs vehicle)
- **Performance**: F1 = 0.96+ on both domains

![MegaDetector Architecture](web_app/assets/figs/megadetectorv6.png)

### Stage 2: ConvNeXt-Small Classifier  
- **Purpose**: Fine-grained species classification
- **Features**: ViT-inspired CNN with 7×7 depthwise convolutions
- **Training**: Class-balanced focal loss + weighted sampling
- **Innovation**: Tail-aware augmentation for rare species

![ConvNeXt Architecture](web_app/assets/figs/convnext.png)

### Technical Innovations
- **Progressive Augmentation**: Cosine-scheduled intensity ramping
- **Class-Balanced Focal Loss**: Addresses long-tail distribution
- **Domain-Aware Training**: Freeze-unfreeze scheduling
- **Confidence Gating**: Two-threshold system for robustness

##  Results

### Domain Shift Performance

![Results Comparison](web_app/assets/domainshift_delta_f1.png) <!-- Suggested: Bar chart comparing single-stage vs two-stage -->

### Single-Stage Baselines

| Model | CIS F1 | TRANS F1 | Domain Gap |
|-------|---------|----------|------------|
| YOLOv8 (baseline) | 0.65 | 0.40 | 0.25 |
| YOLOv8 (medium aug) | 0.77 | 0.52 | 0.25 |
| MegaDetector v6 | 0.79 | 0.63 | 0.16 |

### Two-Stage Pipeline

| Component | CIS F1 | TRANS F1 | Gap Reduction |
|-----------|---------|----------|---------------|
| **Full Pipeline** | **0.90** | **0.77** | **27.5% vs best single-stage** |

![Performance Metrics](assets/performance_charts.png) <!-- Your existing F1/Precision/Recall charts -->

### Per-Species Analysis

![Confusion Matrix](assets/confusion_matrices.png) <!-- Your existing confusion matrices -->

*Detailed per-class metrics available in the [evaluation notebook](evaluation.ipynb)*

##  Running the Project

### 1. Web Application Demo
```bash
streamlit run web_app/app.py
```
- **Interactive inference**: Upload images for real-time prediction
- **Species gallery**: Browse examples by species
- **Performance metrics**: Live comparison with ground truth

### 2. Evaluation Notebook

- **Complete model comparison**: Single-stage vs two-stage results
- **Training diagnostics**: Loss curves and validation metrics  
- **Domain shift analysis**: CIS vs TRANS performance breakdown

### 3. Verification Notebook  

- **End-to-end pipeline testing**: Pre-computed results on verification set
- **Visual inspection**: Ground truth vs prediction overlays
- **Threshold analysis**: Confidence score distributions

##  Project Structure

```
wildlife-camera-trap-classification/
├──  configs/                     # Training configurations
│   ├── megadetector_test/         # MegaDetector configs
│   ├── model/                     # Model hyperparameters  
│   └── test/                      # Test configurations
├──  data/                       # Dataset files
│   ├── preprocessed/              # Cleaned COCO annotations
│   └── verification/              # Verification image subset
├──  eval/                       # Evaluation results & metrics
│   ├── classifier_stage/          # Species classifier results
│   ├── single_stage/             # Single stage model results
│   ├── detector_stage/            # Object detector results
│   └── pipeline_results/          # End-to-end pipeline metrics
├──  models/                     # Exported ONNX models
│   ├── megadetectorv6.onnx       # Animal/vehicle detector
│   └── convnext_classifier.onnx   # Species classifier
├── 📁 notebooks/                  # Jupyter notebooks
│   └── eda_and_dataset_prep.ipynb # Exploratory data analysis
├── 📁 reports/                    # Generated evaluation reports of full pipeline (Verification set)
├── 📁 scripts/                    # Training & preprocessing scripts
│   ├── augmentation/              # Data augmentation pipelines
│   ├── dataset/                   # Dataset preparation utilities
│   └── train/                     # Model training scripts
├── 📁 web_app/                    # Streamlit web application
├── 📄 evaluation.ipynb            # Complete model evaluation
├── 📄 verification.ipynb          # Pipeline verification
├── 📄 requirements.txt            # Python dependencies
└── 📄 README.md                   # This file
└── Project_Poster.pdf              # Poster Presentation of project
```

*For detailed information about each component, see the individual README files in each folder.*

##  Web Application

### Features
- ** Live Inference**: Real-time species prediction with confidence scores
- ** Interactive Metrics**: Performance comparison across domains
- ** Species Browser**: Explore predictions by species type
- ** Visualization**: Ground truth vs prediction overlays

### Demo Screenshots

![Web App Demo](assets/webapp_demo.png) <!-- Suggested: Screenshot of your Streamlit app -->

### Launch Instructions
```bash
cd web_app/
streamlit run app.py
```

Navigate to `http://localhost:8501` to access the application.

OR

Go to this deployed website, easy. -> https://wildlife-cameratrap.streamlit.app/

##  Documentation

### Detailed Component Documentation
- **[Scripts](scripts/README.md)**: Training and preprocessing utilities
- **[Evaluation](eval/README.md)**: Metrics and analysis results  
- **[Web App](web_app/README.md)**: Application setup and features

### Key Notebooks
| Notebook | Purpose |
|----------|---------|
| `evaluation.ipynb` | **Complete model comparison & training analysis** |
| `verification.ipynb` | **End-to-end pipeline testing on verification set** |
| `notebooks/eda_and_dataset_prep.ipynb` | **Dataset exploration & preprocessing** |



##  Technical Approach

### Why Two-Stage Pipeline?
1. **Decoupled Optimization**: Detection and classification trained separately
2. **Background Invariance**: Cropping reduces location-specific bias
3. **Modular Design**: Components can be upgraded independently  
4. **Domain Robustness**: Locating "something alive" generalizes better than specific species recognition

### Novel Contributions
- **Progressive Augmentation Scheduling**: Gradual intensity increase during training
- **Tail-Aware Training**: Specialized augmentation banks for rare species
- **Cross-Domain Evaluation**: Comprehensive analysis of location generalization

##  Key Achievements

-  **State-of-the-art** cross-domain performance on CCT20 benchmark
-  **27.5% error reduction** on unseen locations vs. best baseline
-  **High recall maintenance** for rare species under domain shift
-  **Production-ready** ONNX models with web interface

##  Contributing


##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Citation

If you use this work in your research, please cite:

```bibtex
@misc{wildlife_camera_trap_2025,
  title={Wildlife Camera-Trap Species Classification with Domain-Shift Robustness},
  author={Sukru Deniz Cilek},
  year={2025},
  institution={University College Dublin},
  note={M.Sc. Project}
}
```

##  Author

**[Şükrü Deniz Çilek]**
-  Email: [ukru.cilek@ucdconnect.ie](mailto:ukru.cilek@ucdconnect.ie)
-  LinkedIn: [https://www.linkedin.com/in/denizcilek/](https://www.linkedin.com/in/denizcilek/)
-  Institution: [University College Dublin], M.Sc. Data & Computational Science

---

##  Acknowledgments

- **Microsoft AI for Earth** for MegaDetector v6
- **Caltech Camera Traps** team for the CCT20 dataset
- **Meta AI Research** for ConvNeXt architecture
- **Ultralytics** for YOLOv8 framework

---

##  Quick Navigation

**Want to dive right in?**
-  **[Run Web App](web_app/)** - Interactive demo
-  **[View Results](evaluation.ipynb)** - Complete analysis  
-  **[Test Pipeline](verification.ipynb)** - End-to-end verification
-  **[Read Methods](eval/README.md)** - Technical details

**For researchers:**
-  **[Training Scripts](scripts/README.md)** - Reproduce experiments
-  **[Evaluation Metrics](reports/)** - Detailed performance analysis

---

*This project demonstrates the effectiveness of two-stage architectures for robust wildlife monitoring under domain shift, contributing to automated conservation efforts worldwide.*
