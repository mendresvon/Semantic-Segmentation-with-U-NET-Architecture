# Semantic Segmentation with U-Net Architecture

This repository contains a complete end-to-end deep learning pipeline for semantic segmentation using a U-Net architecture. The model is trained on the Cambridge-driving Labeled Video Database (CamVid) dataset to classify driving scenes pixel-by-pixel into distinct semantic categories (such as roads, vehicles, and pedestrians).

## Project Overview

The project leverages the **fastai** framework and **PyTorch** to implement semantic segmentation. It transitions from a frozen encoder baseline to a fully unfrozen, fine-tuned network using differential learning rates.

### Key Features
* **Architecture:** U-Net utilizing a pre-trained **ResNet34** backbone.
* **Optimizations:** Implements self-attention layers for improved spatial awareness and the **Mish** activation function for smoother gradient flow.
* **Custom Evaluation:** Features a custom accuracy metric (`acc_camvid`) that explicitly ignores "Void" (unlabeled) boundary pixels to ensure completely honest performance tracking.
* **Deployment Ready:** The trained weights are serialized into a standard FP32 `export.pkl` file, optimized for lightweight CPU inference and ready for integration with a Gradio web application or Hugging Face Spaces.

---

## Model Architecture

The core of this repository relies on a classic encoder-decoder U-Net design. To fulfill structural neural network transparency, the implementation includes explicit native PyTorch blocks:

```python
class PyTorchUnetDoubleConv(nn.Module):
    """
    A standard native PyTorch Neural Network Module representing
    the essential Double Convolution block inside a U-Net architecture.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)
