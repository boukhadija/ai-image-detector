from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# A REAL-class example with a blank randomized Grad-CAM
input_file = Path(
    "results/gradcam_prerelu_80/raw_TN_021_layer2_4.npy"
)

raw = np.load(input_file)
after_relu = np.maximum(raw, 0)

# Same color scale for both maps
limit = float(np.max(np.abs(raw)))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, data, title in zip(
    axes,
    [raw, after_relu],
    ["Before ReLU: signed values", "After ReLU: clipped values"],
):
    image = ax.imshow(
        data,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.axis("off")

fig.colorbar(
    image,
    ax=axes,
    label="Raw Grad-CAM value",
    shrink=0.8,
)

fig.suptitle("Why a Randomized REAL-Class Grad-CAM Heatmap Becomes Blank")
# Leave space below the plots for the caption
fig.subplots_adjust(bottom=0.25)

caption = (
    "Figure: Pre- and post-ReLU Grad-CAM for a CIFAKE true-negative image,\n"
    "with ResNet18 layers 2–4 randomized and the explanation target fixed to REAL.\n"
    "Raw values range from −0.001926 to −0.000514; ReLU clips all values to zero."
)

fig.text(
    0.5, 0.03,
    caption,
    ha="center",
    va="bottom",
    fontsize=9,
    color="black",
    linespacing=1.5,
)

output = Path("results/figures/gradcam_prerelu_diagnostic.png")
output.parent.mkdir(parents=True, exist_ok=True)

fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Raw minimum:", raw.min())
print("Raw maximum:", raw.max())
print("After ReLU all zero:", np.all(after_relu == 0))
print("Figure saved:", output)