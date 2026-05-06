# Rare Category Identification: Long-Tailed Object Detection Using Background Splitting

**BSc Computer and Internet Engineering — Final Year Dissertation**  
**Fareed Hakeem-Habeeb** | Supervisor: Dr Anjan Dutta | University of Surrey, 2026

## Overview

This project implements and evaluates **Background Splitting** ([Mullapudi et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Mullapudi_Background_Splitting_Finding_Rare_Classes_in_a_Sea_of_Background_CVPR_2021_paper.pdf)) for long-tailed object detection on COCO 2017.

Background Splitting decomposes the monolithic "background" class in two-stage detectors into semantically meaningful sub-categories. By training an auxiliary classifier to predict background scene types (e.g., indoor, outdoor, urban), the shared backbone learns finer-grained features that improve discrimination between rare foreground objects and their surroundings.

### Key Findings

| Model | Teacher | AP | AP50 | AP75 | APs | APm | APl |
|-------|---------|------|------|------|------|------|------|
| C4 Baseline | — | 21.03 | 36.83 | 21.13 | 7.80 | 22.39 | 31.19 |
| **FPN Baseline** | — | **21.57** | **39.58** | 21.19 | 10.36 | **23.21** | 29.26 |
| BG Split (DeepLabV3) | DeepLabV3 | 21.41 | 38.76 | 21.27 | 9.86 | 22.67 | 29.45 |
| **BG Split (Places365)** | Places365 | **21.58** | 39.21 | **21.41** | **10.49** | 22.61 | **29.40** |

- DeepLabV3 pseudo-labels (95.7% generic_background) **degrade** performance vs the fair FPN baseline (−0.16 AP)
- Places365 scene-level labels (43.3% max class) **recover** to baseline level (+0.01 AP, +0.22 AP75, +0.13 APs)
- **Pseudo-label quality is the critical factor** determining whether Background Splitting helps or hurts

## Repository Structure

```
rare-category-detection/
├── student/
│   ├── bg_split_roi_heads.py      # Custom ROI head with auxiliary BG classifier
│   └── bg_split_dataset.py        # Dataset mapper for loading pseudo-labels
├── teacher/
│   └── generate_pseudo_labels.py  # DeepLabV3 background pseudo-label generation
├── train_baseline.py              # Faster R-CNN C4 baseline (90K iters)
├── train_fpn_baseline.py          # Fair FPN baseline without BG Split (60K iters)
├── train_bg_split.py              # Background Splitting training
├── train_ablation.py              # Ablation study runner (λ and K sweeps)
├── generate_places365_labels.py   # Places365 scene-level pseudo-label generation
├── recluster_bg_labels.py         # Re-cluster labels for K ablation (K=3, K=14)
├── visualise_detections.py        # Qualitative comparison visualisations
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- NVIDIA GPU with ≥12 GB VRAM (tested on RTX 5070, Blackwell sm_120)
- CUDA 12.x
- Python 3.10+
- Conda (recommended)

### Installation

```bash
# Create environment
conda create -n rare_category python=3.10
conda activate rare_category

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install Detectron2 from source
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && pip install -e . && cd ..

# Install remaining dependencies
pip install -r requirements.txt

# Clone this repository
git clone https://github.com/YOUR_USERNAME/rare-category-detection.git
cd rare-category-detection
```

### Dataset Setup

```bash
# Download COCO 2017
mkdir -p datasets/coco
cd datasets/coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ../..
```

### ImageNet Weights

```bash
# Convert torchvision ResNet-50 weights to Detectron2 format
mkdir -p output
python detectron2/tools/convert-torchvision-to-d2.py \
    torchvision://resnet50 output/R-50_d2.pkl
```

## Running Experiments

### Step 1: Train the C4 Baseline (~9 hours)

```bash
python train_baseline.py --num-gpus 1 MODEL.WEIGHTS ./output/R-50_d2.pkl
```

### Step 2: Train the Fair FPN Baseline (~7 hours)

```bash
python train_fpn_baseline.py --num-gpus 1 MODEL.WEIGHTS ./output/R-50_d2.pkl
```

### Step 3: Generate DeepLabV3 Pseudo-Labels (~90 minutes)

```bash
python teacher/generate_pseudo_labels.py
```

Output: `./datasets/coco/bg_labels/bg_pseudo_labels.json`

### Step 4: Train BG Split with DeepLabV3 Labels (~7 hours)

```bash
python train_bg_split.py --num-gpus 1 MODEL.WEIGHTS ./output/R-50_d2.pkl
```

### Step 5: Run Ablation Studies (~7 hours each)

```bash
# λ ablation
python train_ablation.py --lambda-val 0.1 --num-bg-classes 7 MODEL.WEIGHTS ./output/R-50_d2.pkl
python train_ablation.py --lambda-val 1.0 --num-bg-classes 7 MODEL.WEIGHTS ./output/R-50_d2.pkl

# K ablation (generate K=3 labels first)
python recluster_bg_labels.py --K 3
python train_ablation.py --lambda-val 0.5 --num-bg-classes 3 \
    --bg-labels ./datasets/coco/bg_labels/bg_pseudo_labels_K3.json \
    MODEL.WEIGHTS ./output/R-50_d2.pkl
```

### Step 6: Generate Places365 Pseudo-Labels (~90 minutes)

```bash
python generate_places365_labels.py
```

Output: `./datasets/coco/bg_labels/bg_pseudo_labels_places365_K7.json`

### Step 7: Train BG Split with Places365 Labels (~7 hours)

```bash
python train_ablation.py --lambda-val 0.5 --num-bg-classes 7 \
    --bg-labels ./datasets/coco/bg_labels/bg_pseudo_labels_places365_K7.json \
    MODEL.WEIGHTS ./output/R-50_d2.pkl
```

### Step 8: Generate Qualitative Results (~10 minutes)

```bash
python visualise_detections.py \
    --baseline-weights ./output/baseline/model_final.pth \
    --fpn-baseline-weights ./output/fpn_baseline/model_final.pth \
    --bgsplit-weights ./output/ablation_lambda0.5_K7/model_final.pth
```

## Architecture

The system has three phases:

1. **Baseline** — Standard Faster R-CNN (C4 or FPN backbone) trained on COCO
2. **Teacher** — DeepLabV3-ResNet101 or Places365-ResNet50 generates background pseudo-labels for each training image
3. **Student** — Modified Faster R-CNN with an auxiliary background classification head (`BackgroundSplittingROIHeads`)

The auxiliary head adds a two-layer FC network (1024 → ReLU → Dropout → K classes) that predicts background sub-categories for negative proposals. The combined loss is:

```
L_total = L_cls + L_box + L_rpn + λ · L_bg
```

During inference, the auxiliary head is unused — no runtime cost.

## Results Summary

### Pseudo-Label Distribution Comparison

| Teacher | Max Class | Distribution |
|---------|-----------|-------------|
| DeepLabV3 | **95.7%** generic_background | Nearly uniform — trivial auxiliary task |
| Places365 | **43.3%** other | Well-balanced — challenging auxiliary task |

### λ Ablation (DeepLabV3 Labels)

| λ | AP | Trend |
|---|------|-------|
| 0 (FPN baseline) | **21.57** | — |
| 0.1 | 21.47 | ↓ |
| 0.5 | 21.41 | ↓ |
| 1.0 | 21.24 | ↓ |

Monotonic degradation confirms the auxiliary task is harmful with near-uniform labels.

## Hardware

All experiments were conducted on a single NVIDIA RTX 5070 (12 GB GDDR7, Blackwell sm_120) running Ubuntu 24.04 under WSL2.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{mullapudi2021background,
  title={Background Splitting: Finding Rare Classes in a Sea of Background},
  author={Mullapudi, Ravi Teja and Shyam, Pramit and Wang, William R and Keutzer, Kurt and Gonzalez, Joseph},
  booktitle={CVPR},
  pages={8043--8052},
  year={2021}
}
```

## License

This project was developed as a BSc dissertation at the University of Surrey. The code is provided for academic and research purposes.
