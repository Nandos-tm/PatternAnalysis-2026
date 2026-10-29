
    def embed_features(self, features: torch.Tensor) -> torch.Tensor:
        # Unit-length vectors make pair distances less sensitive to feature scale.
        return F.normalize(self.projector(features), p=2, dim=1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        # These are logits, not probabilities; BCEWithLogitsLoss applies the
        # sigmoid operation in a numerically stable way during training.
        return self.classifier(features).squeeze(1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_features(self.encode_features(x))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.encode_features(x))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # Compute each endpoint once, then use both heads on the resulting
        # features. The return order is (embedding A, embedding B, logit A, logit B).
        features_a = self.encode_features(a)
        features_b = self.encode_features(b)
        return (
            self.embed_features(features_a),
            self.embed_features(features_b),
            self.classify_features(features_a),
            self.classify_features(features_b),
        )


def contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, same: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Pull matching-class embeddings together and separate different-class pairs."""
    # `same` is 1 for a matching-class pair. Its squared distance is minimised.
    # A different-class pair is penalised only while it is closer than `margin`.
    # Cast to float32 so this calculation stays stable when the forward pass
    # uses mixed precision.
    distance = F.pairwise_distance(z1.float(), z2.float())
    same = same.float()
    return (same * distance.square() + (1.0 - same) * F.relu(margin - distance).square()).mean()
