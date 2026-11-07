
"""Visualize CIFAKE frequency features using PCA."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.frequency_features import (
    create_frequency_dataloaders,
    extract_loader_features,
)


def main():
    # 1. Load the same dataset used by our frequency model.
    loaders, class_to_idx = create_frequency_dataloaders(
        Path("data"),
        batch_size=128,
        seed=42,
        image_size=32,
    )

    # 2. Extract 240 FFT/DCT features from each image.
    train_features, _ = extract_loader_features(loaders["train"])
    test_features, test_labels = extract_loader_features(loaders["test"])

    print("Training features:", train_features.shape)
    print("Test features:", test_features.shape)

    # 3. Learn scaling from training data only.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    # 4. Learn two principal components from training data.
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_scaled)

    # 5. Project held-out test images into two dimensions.
    test_pca = pca.transform(test_scaled)

    # Use a balanced sample for a readable scatter plot.
    rng = np.random.default_rng(42)
    selected = []

    for label in (class_to_idx["fake"], class_to_idx["real"]):
        indices = np.flatnonzero(test_labels == label)
        selected.extend(
            rng.choice(indices, size=min(1000, len(indices)), replace=False)
        )

    selected = np.asarray(selected)

    # 6. Plot REAL and FAKE images.
    plt.figure(figsize=(9, 7))

    for name, color in [("fake", "tab:orange"), ("real", "tab:blue")]:
        mask = test_labels[selected] == class_to_idx[name]
        plt.scatter(
            test_pca[selected[mask], 0],
            test_pca[selected[mask], 1],
            s=12,
            alpha=0.45,
            label=name.upper(),
            color=color,
        )

    variance = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({variance[0]:.1f}% variance)")
    plt.ylabel(f"PC2 ({variance[1]:.1f}% variance)")
    plt.title("PCA of FFT + DCT Features — CIFAKE Test Set")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    output = Path("results/frequency_pca.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()

    print("Explained variance:", variance)
    print(f"PCA visualization saved to {output}")


if __name__ == "__main__":
    main()
