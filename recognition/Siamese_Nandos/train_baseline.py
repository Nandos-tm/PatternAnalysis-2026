"""Train the ResNet-18 baseline and save evaluator-compatible checkpoints."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from dataset import ISICImageDataset, index_images, make_transform
from modules import ResNet18Baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, output = Path(args.splits), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_df, val_df = pd.read_csv(splits / "train.csv"), pd.read_csv(splits / "val.csv")
    paths = index_images(args.images)
    required = set(train_df.image_name.astype(str)) | set(val_df.image_name.astype(str))
    missing = required - paths.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} split images missing; examples: {list(missing)[:5]}")

    train_ds = ISICImageDataset(train_df, paths, make_transform(args.image_size, train=True))
    val_ds = ISICImageDataset(val_df, paths, make_transform(args.image_size))
    loader_args = {
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **loader_args)
    model = ResNet18Baseline(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    )
    best_ap, history = -1.0, []
    best_path = output / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        loss_sum, seen, started = 0.0, 0, time.time()
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            seen += len(y)

        model.eval()
        logits_all, labels, names = [], [], []
        with torch.inference_mode():
            for x, y, image_names in val_loader:
                logits = model(x.to(device, non_blocking=True)).float()
                logits_all.extend(logits.cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())
                names.extend(image_names)
        ap = float(average_precision_score(labels, logits_all))
        auc = float(roc_auc_score(labels, logits_all))
        row = {"epoch": epoch + 1, "train_loss": loss_sum / max(seen, 1),
               "val_ap": ap, "val_auroc": auc, "seconds": time.time() - started}
        history.append(row)
        probabilities = 1 / (1 + np.exp(-np.clip(np.asarray(logits_all), -40, 40)))
        predictions = pd.DataFrame({
            "image_name": names, "target": labels,
            "logit": logits_all, "probability": probabilities,
        })
        if ap > best_ap:
            best_ap = ap
            torch.save({"architecture": "resnet18", "state_dict": model.state_dict(),
                        "epoch": epoch + 1, "val_ap": ap, "config": vars(args)}, best_path)
            predictions.to_csv(output / "best_val_predictions.csv", index=False)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(row)
    print(f"Best validation AP: {best_ap:.6f}; checkpoint: {best_path}")


if __name__ == "__main__":
    main()
