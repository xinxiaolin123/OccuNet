# OccuNet

OccuNet is a two-stage deep-learning framework for femoral-neck fracture detection on pelvic or hip radiographs. This repository provides code and model-configuration files related to the Radiology manuscript:

**Improving Recognition Performance and Reader Efficiency for Femoral-Neck Fracture Detection on Pelvic or Hip Radiographs Using Artificial Intelligence**

OccuNet was developed for research use in a retrospective multicenter diagnostic-accuracy study. The framework includes artifact-robust contrastive pretraining followed by detector fine-tuning for femoral-neck fracture localization.

## Important notice

This repository is provided for research transparency and reproducibility. It is **not** a clinically approved diagnostic device and must not be used for clinical decision-making, triage, or patient management without prospective validation, regulatory review, and appropriate institutional approval.

Patient-level imaging data are not publicly released because of institutional privacy, ethics approval, and data-use restrictions.

## Repository contents

| File | Description |
|---|---|
| `Occunet.yaml` | OccuNet detector architecture configuration. |
| `stage1_contrastive_learning(Pretraining).py` | Stage 1 contrastive pretraining script for artifact-robust radiographic feature learning. |
| `stage1_model_weights.pt` | Stage 1 pretrained model weights used to initialize detector fine-tuning. |
| `stage2_Occunet_Training.py` | Stage 2 detector fine-tuning script for femoral-neck fracture localization. |
| `Test_radiograph.png` | Example de-identified radiograph for code demonstration. |

## Method overview

OccuNet uses a two-stage training paradigm:

1. **Stage 1: Artifact-robust contrastive pretraining**  
   Paired original and artifact-augmented radiographs are used to learn artifact-invariant radiographic representations.

2. **Stage 2: Detector fine-tuning**  
   The pretrained backbone is transferred to a detector trained for femoral-neck fracture localization on pelvic or hip radiographs.

At inference, a radiograph is classified as positive when at least one femoral-neck detection reaches the prespecified confidence threshold.

## Software environment

The manuscript experiments were run with the following environment:

- Python 3.10
- PyTorch 2.8.0
- torchvision 0.23.0
- Ultralytics 8.0.196
- OpenCV 4.12.0.88
- NumPy 2.2.6
- Matplotlib 3.10.5
- PyYAML 6.0.2
- pytorch-grad-cam 1.5.5
- CUDA 12.9
- NVIDIA GeForce RTX 4060 GPU

A similar CUDA-enabled Python environment is recommended.

## Installation

Clone the repository:

```bash
git clone https://github.com/xinxiaolin123/OccuNet.git
cd OccuNet
