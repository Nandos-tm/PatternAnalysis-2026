""Run held-out evaluation for the COMP3710 ISIC 2020 classifiers.

The script follows the notebook evaluation pattern: use the validation split
to choose an operating threshold, then report the held-out test performance.
The threshold is never selected from test labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# This script can run on Colab or from a terminal. Use a non-interactive
# plotting backend so saving figures does not require a display window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from dataset import ISICImageDataset, index_images, make_transform
from modules import ResNet18Baseline, SiameseNetWithClassifier
