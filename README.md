# 🦅 Reproducing EAGLE & STEGO for Unsupervised Semantic Segmentation

This repository contains our implementation and reproduction of **EAGLE (Eigen Aggregation Learning)** and **STEGO (Self-supervised Transformer with Energy-based Graph Optimization)** for **Unsupervised Semantic Segmentation (USS)** on the **COCO-Stuff** dataset.

This project was completed as part of the **CSD454 – Computer Vision** course at **Shiv Nadar University**.

---

## 📌 Project Overview

Semantic segmentation traditionally relies on dense pixel-level annotations, making it expensive and time-consuming to build large labeled datasets.

Unsupervised Semantic Segmentation (USS) addresses this challenge by learning meaningful semantic regions directly from unlabeled images.

In this project, we:

- Reproduced the EAGLE framework
- Implemented STEGO for comparison
- Trained and evaluated both models on the COCO-Stuff dataset
- Compared quantitative and qualitative segmentation performance
- Explored object-centric representation learning using Vision Transformers

---

## 🚀 Features

- PyTorch implementation
- COCO-Stuff dataset support
- Reproduction of EAGLE
- STEGO implementation for comparison
- Training and evaluation scripts
- Qualitative visualization of segmentation outputs
- Quantitative comparison using multiple evaluation metrics

---

# 📊 Results

## Quantitative Results

| Metric | STEGO | EAGLE |
|:----------------|------:|------:|
| Linear mIoU | 41.0 | **43.9** |
| Linear Accuracy | 76.14 | **76.8** |
| Cluster mIoU | **28.2** | 27.2 |
| Cluster Accuracy | 57.0 | **64.2** |

Our reproduced implementation achieved results comparable to the official EAGLE implementation while outperforming STEGO in Linear mIoU and Cluster Accuracy.

---

## 🖼️ Qualitative Results

The following examples compare

**Input Image → Ground Truth → EAGLE Prediction**

<p align="center">
<img src="docs/figures/results.png" width="100%">
</p>

---

# 📄 Project Report

A detailed report describing the methodology, implementation, experiments, and results is available here.

📄 **[Project Report](docs/report.pdf)**

The report includes:

- Problem Statement
- Literature Review
- EAGLE Architecture
- STEGO Implementation
- Dataset Preparation
- Training Pipeline
- Experimental Results
- Comparative Analysis
- Future Work

---

# 📂 Repository Structure

```text
.
├── src_EAGLE/
│   ├── configs/
│   ├── datasets/
│   ├── models/
│   ├── train_segmentation_eigen.py
│   ├── eval_segmentation.py
│   └── ...
│
├── docs/
│   ├── report.pdf
│   └── figures/
│       ├── results.png
│       ├── architecture.png
│       └── pipeline.png
│
├── environment.yml
├── LICENSE
└── README.md
```

---

# ⚙️ Installation

## Create Conda Environment

```bash
conda env create -f environment.yml
conda activate EAGLE
```

---

## Download Pre-trained Models

```bash
cd src_EAGLE
python download_models.py
```

---

## Download COCO-Stuff Dataset

Modify the dataset directory inside the configuration file.

Run

```bash
python download_datasets.py
```

Extract the downloaded archive.

```bash
cd /YOUR/DATA/DIR
unzip cocostuff.zip
```

---

# 🏋️ Training

Create cropped images

```bash
python crop_datasets.py
```

Train EAGLE

```bash
cd src_EAGLE

python train_segmentation_eigen.py
```

Training hyperparameters can be modified in

```
src_EAGLE/configs/train_config_cocostuff.yml
```

---

# 📈 Evaluation

Evaluate the trained model

```bash
cd src_EAGLE

python eval_segmentation.py
```

Evaluation configuration

```
src_EAGLE/configs/eval_config.yml
```

---

# 📚 Technologies Used

- Python
- PyTorch
- Vision Transformers (ViT)
- DINO
- EAGLE
- STEGO
- COCO-Stuff
- Unsupervised Semantic Segmentation

---

# 👨‍💻 Team

**Shiv Nadar University**

- Bavineni Navneet
- Korrapati Jaideep
- Sapparapu Karthikeya
- Shyam Suchit Reddy

---

# 🙏 Acknowledgements

This project reproduces and builds upon the following research works.

**EAGLE**
> Kim et al., *Revisiting Unsupervised Segmentation: Object-Level Understanding in Vision Transformers*, 2024.

**STEGO**
> Hamilton et al., *Unsupervised Semantic Segmentation by Distilling Feature Correspondences*, ICLR 2022.

We sincerely thank the authors for making their code and research publicly available.

---

# ⭐ If you found this repository useful

Please consider giving it a ⭐ on GitHub.

It helps others discover the project and motivates future work.
