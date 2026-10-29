"""Train the auxiliary-head Siamese model using the notebook's loss and schedule."""

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

from dataset import ISICImageDataset, PairDataset, index_images, make_transform
from modules import SiameseNetWithClassifier, contrastive_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", required=True, help="Directory containing train.csv and val.csv")
    parser.add_argument("--images", required=True, help="Directory containing extracted JPEGs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64, help="Pairs per batch")
    parser.add_argument("--pairs-per-epoch", type=int, default=24000)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3710)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.pairs_per_epoch % args.batch_size:
        raise ValueError("pairs-per-epoch must be divisible by batch-size.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, output = Path(args.splits), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(splits / "train.csv")
    val_df = pd.read_csv(splits / "val.csv")
    if "patient_id" in train_df and "patient_id" in val_df:
        if not set(train_df.patient_id).isdisjoint(val_df.patient_id):
            raise ValueError("Patient overlap between training and validation splits.")
    paths = index_images(args.images)
    required = set(train_df.image_name.astype(str)) | set(val_df.image_name.astype(str))
    missing = required - paths.keys()
    if missing:
        raise FileNotFoundError(f"{len(missing)} split images missing; examples: {list(missing)[:5]}")

    pair_ds = PairDataset(train_df, paths, args.pairs_per_epoch, args.seed)
    train_loader = DataLoader(
        pair_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )
    val_ds = ISICImageDataset(val_df, paths, make_transform(args.image_size))
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    model = SiameseNetWithClassifier(args.embedding_dim, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    best_ap, start_epoch, history = -1.0, 0, []
    last_path, best_path = output / "last.pt", output / "best.pt"
    if args.resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_ap, history = checkpoint["best_ap"], checkpoint["history"]

    for epoch in range(start_epoch, args.epochs):
        pair_ds.set_epoch(epoch)
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        started = time.time()
        for a, b, same, y_a, y_b in train_loader:
            a, b = a.to(device), b.to(device)
            same, y_a, y_b = same.to(device), y_a.to(device), y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                z_a, z_b, logit_a, logit_b = model(a, b)
                metric_loss = contrastive_loss(z_a, z_b, same, args.margin)
                class_loss = 0.5 * (criterion(logit_a, y_a) + criterion(logit_b, y_b))
                loss = metric_loss + args.class_loss_weight * class_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals += np.array([loss.item(), metric_loss.item(), class_loss.item()]) * len(same)

        model.eval()
        logits, labels, names = [], [], []
        with torch.inference_mode():
            for x, y, image_names in val_loader:
                x = x.to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    score = model.classify(x).float()
                logits.extend(score.cpu().numpy().tolist())
                labels.extend(y.numpy().tolist())
                names.extend(image_names)
        ap = float(average_precision_score(labels, logits))
        auc = float(roc_auc_score(labels, logits))
        row = {
            "epoch": epoch + 1,
            "train_total_loss": totals[0] / len(pair_ds),
            "train_contrastive_loss": totals[1] / len(pair_ds),
            "train_classification_loss": totals[2] / len(pair_ds),
            "val_ap": ap,
            "val_auroc": auc,
            "seconds": time.time() - started,
        }
        history.append(row)
        probabilities = 1 / (1 + np.exp(-np.clip(np.asarray(logits), -40, 40)))
        predictions = pd.DataFrame({
            "image_name": names, "target": labels, "score_logit": logits,
            "score_probability": probabilities,
        })
        if ap > best_ap:
            best_ap = ap
            torch.save({
                "model": model.state_dict(), "epoch": epoch, "val_ap": ap,
                "config": vars(args),
            }, best_path)
            predictions.to_csv(output / "best_val_scores.csv", index=False)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "epoch": epoch, "best_ap": best_ap,
            "history": history,
        }, last_path)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(row)
    print(f"Best validation AP: {best_ap:.6f}; checkpoint: {best_path}")


if __name__ == "__main__":
    main()

