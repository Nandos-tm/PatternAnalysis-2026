def index_images(image_dir: str | Path) -> dict[str, Path]:
    """Map image stem to path; rejects duplicate stems to avoid silent misalignment."""
    # The split CSVs contain image IDs rather than full paths. Index the image
    # tree once so each dataset lookup is quick and tied to the correct row.
    paths: dict[str, Path] = {}
    for path in Path(image_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
            if path.stem in paths:
                raise ValueError(f"Duplicate image ID {path.stem}: {paths[path.stem]} and {path}")
            paths[path.stem] = path
    return paths


class ISICImageDataset(Dataset):
    """One row per image; returns (tensor, target, image_name)."""

    def __init__(self, frame: pd.DataFrame, image_paths: dict[str, Path], transform=None):
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.transform = transform or make_transform()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        # Match the manifest row to its JPEG. Converting to RGB gives the model
        # a consistent three-channel input for every image.
        name = Path(str(row["image_name"]).strip()).stem
        with Image.open(self.image_paths[name]) as image:
            x = self.transform(image.convert("RGB"))
        return x, int(row["target"]), name


class PairDataset(Dataset):
    """Balanced four-way pair sampler, with different-patient endpoints."""

    # Cycle through same-benign, same-melanoma and both different-class orders.
    # This avoids letting the raw benign/melanoma imbalance dictate pair types.
    PAIR_TYPES = ((0, 0), (1, 1), (0, 1), (1, 0))

    def __init__(
        self,
        frame: pd.DataFrame,
        image_paths: dict[str, Path],
        n_pairs: int,
        seed: int = 3710,
        transform=None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.image_paths = image_paths
        self.n_pairs = int(n_pairs)
        self.seed = int(seed)
        self.epoch = 0
        self.transform = transform or make_transform(train=True)
        # Save row indices by class now instead of filtering the DataFrame for
        # every pair sampled during an epoch.
        self.by_class = {
            c: self.frame.index[self.frame["target"].astype(int) == c].to_numpy()
            for c in (0, 1)
        }
        if any(len(indices) < 2 for indices in self.by_class.values()):
            raise ValueError("Each class must have at least two examples for pair sampling.")
        if "patient_id" not in self.frame:
            raise ValueError("Pair sampling requires patient_id to prevent same-patient pairs.")

    def set_epoch(self, epoch: int):
        # The epoch contributes to each pair's seed: reruns are reproducible,
        # while later epochs draw fresh pairs and fresh random augmentations.
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_pairs

    def _load(self, row):
        key = Path(str(row["image_name"]).strip()).stem
        with Image.open(self.image_paths[key]) as image:
            return self.transform(image.convert("RGB"))

    def __getitem__(self, index):
        # A per-item RNG avoids depending on the order in which DataLoader
        # workers happen to request samples.
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        class_a, class_b = self.PAIR_TYPES[index % len(self.PAIR_TYPES)]
        index_a = int(rng.choice(self.by_class[class_a]))
        patient_a = self.frame.at[index_a, "patient_id"]
        # Keep endpoints from different patients so the network cannot rely as
        # easily on patient-specific appearance to identify a pair.
        for _ in range(1000):
            index_b = int(rng.choice(self.by_class[class_b]))
            if self.frame.at[index_b, "patient_id"] != patient_a:
                break
        else:
            raise RuntimeError("Could not sample a pair from different patients.")
        row_a, row_b = self.frame.iloc[index_a], self.frame.iloc[index_b]
        # `same` supervises the embedding distance; y_a and y_b supervise the
        # auxiliary classifier on each image separately.
        return (
            self._load(row_a),
            self._load(row_b),
            torch.tensor(float(class_a == class_b), dtype=torch.float32),
            torch.tensor(float(class_a), dtype=torch.float32),
            torch.tensor(float(class_b), dtype=torch.float32),
        )
