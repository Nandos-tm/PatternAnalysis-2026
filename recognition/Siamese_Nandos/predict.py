"""Run held-out evaluation for the COMP3710 ISIC 2020 classifiers.

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


def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    """Build the requested architecture and load either training script's checkpoint."""
    # Load onto CPU first. This avoids briefly allocating a second copy of the
    # weights on a GPU and works even when a checkpoint was saved on another
    # machine or accelerator.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary saved by the training notebook/script.")

    config = checkpoint.get("config") or {}
    if model_name == "baseline":
        # The baseline trainer saves weights under `state_dict`; the modular
        # trainer may also use `model`. Both contain the same ResNet parameters.
        model = ResNet18Baseline(pretrained=False)
        state = checkpoint.get("state_dict", checkpoint.get("model"))
    else:
        # The notebook and modular trainer use slightly different names for
        # the embedding size, so accept either. The classifier head and encoder
        # must still match the checkpoint exactly when strict loading is used.
        embedding_dim = int(config.get("embedding_dim", config.get("embed_dim", 128)))
        model = SiameseNetWithClassifier(dim=embedding_dim, pretrained=False)
        state = checkpoint.get("model", checkpoint.get("state_dict"))

    if state is None:
        raise ValueError(
            f"Could not find model weights in {checkpoint_path}; expected 'model' or 'state_dict'."
        )
    model.load_state_dict(state, strict=True)
    # Evaluation mode disables training-only layer behaviour, such as dropout
    # and batch-normalisation updates, before predictions are generated.
    model.to(device).eval()
    return model, checkpoint


def infer(model, model_name: str, frame: pd.DataFrame, image_paths, args, device):
    """Return ordered logits and labels for one official split manifest."""
    # Reuse dataset.py rather than duplicating image loading and transforms.
    # `train=False` is important here: validation and test images should not
    # receive the random flips and rotations used to augment training images.
    dataset = ISICImageDataset(
        frame,
        image_paths,
        transform=make_transform(args.image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # Keep the manifest order. The saved CSVs can then be joined directly
        # to the official split by image name, and the check below catches any
        # accidental shuffle or mismatch.
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    # Store model outputs as logits first. Sigmoid is applied once later, after
    # concatenation, so inference and thresholding use the same score format.
    all_logits, all_labels, all_names = [], [], []
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            # Both models return one melanoma logit per image. The Siamese
            # network was trained with pairs, but its auxiliary classifier
            # predicts from a single image at evaluation time, just as in its
            # notebook. The pairwise embedding head is not used for this result.
            # Mixed precision mirrors notebook inference on GPU; CPU inference
            # stays in full precision because autocast is disabled there.
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                if model_name == "baseline":
                    logits = model(images)
                else:
                    logits = model.classify(images)
            all_logits.extend(logits.float().cpu().numpy().reshape(-1).tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_names.extend(names)

    # The dataset returns normalised image IDs (without .jpg). Confirm each
    # batch came back in precisely the manifest order before associating labels
    # or writing predictions. This also guards against silent ID/score swaps.
    expected_names = [Path(str(name).strip()).stem for name in frame["image_name"]]
    if all_names != expected_names:
        raise ValueError("Inference order does not match the order in the split CSV.")
    return np.asarray(all_logits, dtype=float), np.asarray(all_labels, dtype=int), all_names


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, policy: str):
    """Select the notebook's operating point from validation scores."""
    # A threshold can only be chosen when validation includes examples from
    # both classes. Test labels are deliberately not passed to this function.
    if len(np.unique(y_true)) != 2:
        raise ValueError("Threshold selection requires both classes in the validation split.")

    if policy == "val_recall70_max_precision":
        # This is the baseline notebook's operating rule: among cutoffs that
        # reach at least 70% melanoma recall, keep the one with highest
        # precision. A higher threshold is used as the tie-breaker.
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        # sklearn adds one terminal precision/recall point without a threshold.
        precision, recall = precision[:-1], recall[:-1]
        eligible = np.flatnonzero(recall >= 0.70)
        if not len(eligible):
            raise ValueError("No validation threshold reaches 70% melanoma recall.")
        index = max(eligible, key=lambda i: (precision[i], thresholds[i]))
        return float(thresholds[index]), {
            "validation_recall": float(recall[index]),
            "validation_precision": float(precision[index]),
            "validation_specificity": float("nan"),
        }

    # The auxiliary-head notebook also explored a fixed-specificity rule.
    # Keeping every ROC point matters here because an intermediate threshold
    # could be the best cutoff satisfying the 80% specificity requirement.
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    specificity = 1.0 - fpr
    eligible = np.flatnonzero(specificity >= 0.80)
    if not len(eligible):
        raise ValueError("No validation threshold reaches 80% specificity.")
    # At tied sensitivity, the notebook takes the first ROC point, which has
    # the higher cutoff and therefore avoids extra false positives.
    best_sensitivity = tpr[eligible].max()
    index = eligible[np.flatnonzero(tpr[eligible] == best_sensitivity)[0]]
    return float(thresholds[index]), {
        "validation_recall": float(tpr[index]),
        "validation_precision": float("nan"),
        "validation_specificity": float(specificity[index]),
    }


def summarize(y_true: np.ndarray, scores: np.ndarray, threshold: float):
    # Ranking metrics (AP and AUROC) use continuous scores; precision, recall,
    # specificity and the confusion matrix use the selected operating cutoff.
    if len(np.unique(y_true)) != 2:
        raise ValueError("Metric reporting requires both classes in the evaluated split.")
    # Label 1 is melanoma throughout the dataset and these reports.
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "auroc": float(roc_auc_score(y_true, scores)),
        "melanoma_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "melanoma_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_plots(y_true: np.ndarray, scores: np.ndarray, threshold: float, output: Path):
    """Save the same compact confusion-matrix, PR and ROC views used in the notebooks."""
    # Save figures instead of displaying them so this works in notebook cells,
    # headless Colab sessions and ordinary command-line runs alike.
    predicted = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Test confusion matrix (cutoff = {threshold:.4f})")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], labels=["Benign", "Melanoma"])
    ax.set_yticks([0, 1], labels=["Benign", "Melanoma"])
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, f"{value:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # The dashed prevalence line is the expected precision of a no-skill
    # ranking model, which gives context for this imbalanced dataset.
    precision, recall, _ = precision_recall_curve(y_true, scores)
    prevalence = float(y_true.mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Average precision = {average_precision_score(y_true, scores):.3f}")
    ax.axhline(prevalence, color="grey", linestyle="--", label=f"Prevalence = {prevalence:.3f}")
    ax.set(title="Precision–recall curve", xlabel="Recall (sensitivity)", ylabel="Precision")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "precision_recall_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # The red vertical marker shows the test false-positive rate at the
    # validation-selected cutoff. It is descriptive; the cutoff is not
    # adjusted after seeing this test result.
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUROC = {roc_auc_score(y_true, scores):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="No-skill reference")
    ax.axvline(1.0 - summarize(y_true, scores, threshold)["specificity"], color="red", linestyle=":")
    ax.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    
