# Image Classification on CIFAR-10 Using a Convolutional Neural Network in PyTorch

A mini project that builds and trains a custom **Convolutional Neural Network (CNN)** in **PyTorch** to classify images from the **CIFAR-10** dataset into 10 object categories.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)

---

## Overview

This project walks through a complete image classification pipeline:

1. Loading and preprocessing the CIFAR-10 dataset
2. Building a CNN from scratch with `torch.nn`
3. Training the model with the Adam optimizer
4. Tracking training and validation loss across epochs
5. Saving the best params for the model
6. Evaluating the final model on the test set

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) contains **60,000 color images (32×32 pixels)** across **10 classes**:

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

| Split    | Images |
| -------- | ------ |
| Training | 50,000 |
| Test     | 10,000 |

The dataset is downloaded automatically through `torchvision.datasets.CIFAR10`.

**Preprocessing:** images are converted to tensors and normalized per channel with mean = 0.5 and std = 0.5, which scales pixel values to the range [-1, 1].

## Model Architecture

The network has three convolutional blocks followed by two fully connected layers.

| Stage         | Layer                                     | Output Shape |
| ------------- | ----------------------------------------- | ------------ |
| Input         | RGB image                                 | 3 × 32 × 32  |
| Block 1       | Conv2d(3→32, 3×3) + ReLU + MaxPool(2×2)   | 32 × 16 × 16 |
| Block 2       | Conv2d(32→64, 3×3) + ReLU + MaxPool(2×2)  | 64 × 8 × 8   |
| Block 3       | Conv2d(64→128, 3×3) + ReLU + MaxPool(2×2) | 128 × 4 × 4  |
| Flatten       |                                           | 2048         |
| FC 1          | Linear(2048→256) + ReLU                   | 256          |
| FC 2 (output) | Linear(256→10)                            | 10           |

## Training Configuration

| Setting       | Value                                |
| ------------- | ------------------------------------ |
| Framework     | PyTorch                              |
| Loss function | Cross-Entropy Loss                   |
| Optimizer     | Adam (default learning rate)         |
| Batch size    | 64                                   |
| Epochs        | 10                                   |
| Validation    | Test set evaluated after every epoch |

## Results

After training, the notebook plots **training loss vs. validation loss** per epoch and prints the final test accuracy.

> **Test Accuracy:** `76.34`

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch and torchvision
- matplotlib
- pandas

### Installation

```bash
git clone https://github.com/theabdullah05-web/cifar10-cnn-pytorch.git
cd cifar10-cnn-pytorch
pip install torch torchvision matplotlib pandas jupyter
```

### Run the notebook

```bash
jupyter notebook CNN_for_CIFAR10.ipynb
```

Or open it directly in **Google Colab**.

> **Note:** The notebook begins by mounting Google Drive (`drive.mount`). If you are running it locally, remove or skip that first cell.

## Project Structure

```
cifar10-cnn-pytorch/
├── CNN_for_CIFAR10.ipynb   # Full pipeline: data, model, training, evaluation
└── README.md
```

## Tech Stack

- Python
- PyTorch / torchvision
- Matplotlib
- Pandas
- Google Colab

## Author

**theabdullah05-web**
GitHub: [@theabdullah05-web](https://github.com/theabdullah05-web)

## License

This project is open source and available for educational use.
