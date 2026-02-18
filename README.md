# TBDLossNet

## Project Structure

```
TBDLossNet/                # Source code
│
├── data_processing/      # Data preprocessing scripts
├── model/                # Model architecture
├── train/                # Training scripts
├── test/                 # Testing scripts
│
VOC2007/                  # Full dataset
│
├── JPEGImages/           # Input images
├── SegmentationClass/    # Pixel-level masks
├── ImageSets/            # Data splits
│
images/                   # Images used in the paper
│
├── requirements.txt
└── README.md
```

---

## Environment Requirements

- Python >= 3.8  
- PyTorch >= 1.10.0  

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Navigate to the training directory:

```bash
cd TBLossNet/train
```

Run the training script:

```bash
python train_multimodal_vmamba.py
```

---

## Testing

Navigate to the testing directory:

```bash
cd TBLossNet/test
```

Run the testing script:

```bash
python test_multimodal_vmamba.py
```

---

## Dependencies

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
tqdm>=4.64.0
scipy>=1.7.0
Pillow>=9.0.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
transformers>=4.30.0
sentencepiece>=0.1.99
huggingface-hub>=0.16.0
einops>=0.6.0
```
