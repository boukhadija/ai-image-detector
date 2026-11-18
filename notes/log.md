cat > notes/log.md <<'EOF'
# AI Image Forensics — Research Log

## November 17, 2026

### Experiment 1: Shared Preprocessing

Objective:
Centralize image preprocessing without changing ResNet18 predictions.

Completed:
- Created src/preprocessing.py.
- Updated gradcam.py.
- Updated gradcam_batch_review.py.
- Verified both scripts produce identical tensors.
- Verified original and updated model outputs match.

Results:
- Image size: 224 x 224
- Maximum tensor difference: 0.0
- Maximum prediction difference: 0.0
- Predicted class: FAKE

Conclusion:
The preprocessing refactor preserved model behavior on the tested CIFAKE image.

### Experiment 2: Grad-CAM Sanity Check

Status: Not started.

Research question:
Do Grad-CAM explanations change when model parameters are progressively randomized?

Planned work:
- Fix the target class across comparisons.
- Randomize model layers progressively.
- Measure heatmap similarity.
- Generate a CSV and comparison chart.
EOF
## November 18, 2026 — Grad-CAM Parameter Randomization

### Objective
Evaluate whether Grad-CAM explanations are sensitive to changes in trained model parameters.

### Method
- Model: Fine-tuned ResNet18 trained on CIFAKE.
- Sample: 80 test images (20 TP, 20 TN, 20 FP, 20 FN).
- Random seed: 42.
- Target class: Fixed to the trained model's predicted class.
- Randomization: Cumulative, beginning with layer4, then layer3, layer2, and layer1.
- Comparisons: 320 image-stage combinations.
- Metrics: Pearson correlation, heatmap mean absolute difference, and all-zero heatmap frequency.

### Results

| Randomized layers | Mean Pearson* | Mean MAE | All-zero heatmaps |
|---|---:|---:|---:|
| Layer 4 | -0.118 | 0.317 | 1/80 |
| Layers 3–4 | -0.058 | 0.300 | 36/80 |
| Layers 2–4 | 0.042 | 0.290 | 40/80 |
| Layers 1–4 | 0.007 | 0.297 | 40/80 |

*Mean Pearson correlation excludes undefined values.

At the final two randomization stages, all 40 TN/FN heatmaps collapsed to zero, while none of the 40 TP/FP heatmaps did.

### Interpretation
Grad-CAM explanations are sensitive to parameter randomization in this experiment. The observed heatmap collapse is strongly associated with the explained class. All-zero heatmaps make Pearson correlation undefined, so correlation must be interpreted alongside heatmap difference and collapse frequency.

### Limitations
- One randomization seed.
- Stratified subset of 80 CIFAKE test images.
- Initial convolutional stem and classifier were not randomized.
- The mechanism causing class-dependent collapse has not been directly verified.
- This experiment alone does not establish explanation faithfulness.

### Artifacts
- results/gradcam_sanity_cumulative_80/sanity_metrics.csv
- results/figures/gradcam_zero_heatmaps.png
