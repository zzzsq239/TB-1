# TB-DLossNet

Official implementation of TB-DLossNet for fine-grained agricultural lesion segmentation.

This repository contains the complete source code and dataset used in the manuscript.

---

## Repository Structure
├── TBLossNet/ # Source code
│ ├── data processing/ # Data preprocessing scripts
│ ├── model/ # Model architecture
│ ├── train/ # Training scripts
│ ├── test/ # Testing scripts
│
├── VOC2007/ # Complete dataset
│ ├── JPEGImages/ # Input images
│ ├── SegmentationClass/ # Pixel-level masks
│ ├── ImageSets/ # Data splits
│
├── images/ # Figures used in the paper
│
├── requirements
└── README.md

## Environment

- Python >= 3.8
- PyTorch >= 1.10.0

Install dependencies:

```bash
pip install -r requirements

Training
To train the model:
cd TBLossNet/train
python train_multimodal_vmamba.py

Testing
To evaluate the model:
cd TBLossNet/test
python test_multimodal_vmamba.py
