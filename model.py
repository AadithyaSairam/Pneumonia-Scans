"""
The CNN.

Five convolutional blocks, then a classifier head. Each block is
conv -> batchnorm -> ReLU -> (dropout) -> 2x2 max-pool, so the spatial size
halves five times: 224 -> 112 -> 56 -> 28 -> 14 -> 7. With 256 channels at
the end, the flattened feature vector is 256 * 7 * 7 = 12544.

That arithmetic is done in `__init__` rather than discovered during the first
forward pass. Discovering it lazily is tempting and it is a trap: the layer
would be created *after* the optimizer had already been handed
`model.parameters()`, so its weights would never appear in any update step
and would stay at their random initialization for the whole run.
"""

from __future__ import annotations

import torch
from torch import nn

# Input images are center-cropped to 224x224 (see data.py).
INPUT_SIZE = 224
N_POOLS = 5
FINAL_CHANNELS = 256

# 224 / 2^5 = 7
SPATIAL = INPUT_SIZE // (2**N_POOLS)
FLAT_FEATURES = FINAL_CHANNELS * SPATIAL * SPATIAL


def _block(in_ch, out_ch, dropout=0.0):
    """conv -> BN -> ReLU -> (dropout) -> pool.

    BatchNorm goes before the activation, which is the order the original
    paper uses and what the torchvision models do. Padding of 1 with a 3x3
    kernel keeps the spatial size unchanged, so the only thing that changes
    it is the pool, which keeps the size arithmetic above honest.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return layers


class PneumoniaCNN(nn.Module):
    """Binary classifier over chest radiographs: NORMAL vs PNEUMONIA."""

    def __init__(self, n_classes=2, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            *_block(3, 32),
            *_block(32, 64, dropout=0.1),
            *_block(64, 64),
            *_block(64, 128, dropout=0.2),
            *_block(128, FINAL_CHANNELS, dropout=0.2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLAT_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        """Returns raw logits, shape (batch, n_classes).

        Logits, not probabilities: `nn.CrossEntropyLoss` applies log-softmax
        itself, and applying softmax here as well would flatten the gradients.
        """
        return self.classifier(self.features(x))


def build(n_classes=2, device=None):
    """Construct the model on the right device."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PneumoniaCNN(n_classes=n_classes).to(device), device
