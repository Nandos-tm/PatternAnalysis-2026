    model = ResNet18Baseline(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    # Supply raw logits to BCEWithLogitsLoss; it combines sigmoid and BCE in a
    # numerically stable operation. pos_weight can increase the melanoma loss.
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
        # Average precision is the selection metric. Use logits directly for
        # ranking metrics; sigmoid probabilities are saved for later reporting.
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
        # Preserve the checkpoint format expected by the baseline evaluator.
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
