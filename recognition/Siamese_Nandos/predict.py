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

def parse_args() -> argparse.Namespace:
    # Keep paths explicit: the image archive and trained weights can live in
    # Drive, while extracted JPEGs and result files can live on local storage.
    parser = argparse.ArgumentParser(
        description="Predict on an ISIC split and evaluate using a validation-selected threshold."
    )
    parser.add_argument(
        "--model",
        choices=("baseline", "siamese_aux"),
        required=True,
        help="Use the ResNet-18 baseline or the Siamese auxiliary classifier head.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    parser.add_argument("--splits", required=True, help="Directory containing val.csv and test.csv")
    parser.add_argument("--images", required=True, help="Directory containing extracted JPEGs")
    parser.add_argument("--output", required=True, help="Directory for CSV results and plots")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    # 224 pixels and ImageNet normalisation match both notebook classifiers.
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--threshold-policy",
        choices=("val_recall70_max_precision", "val_specificity80_max_sensitivity"),
        default="val_recall70_max_precision",
        help="Notebook operating point, selected using validation predictions only.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use a previously chosen cutoff instead of selecting one from validation.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG visualisations")
    return parser.parse_args()
