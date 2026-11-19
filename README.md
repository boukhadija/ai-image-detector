# ai-image-detector

Baseline project for classifying CIFAKE images as real or AI-generated.

## Dataset preparation

Install the project dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download and extract CIFAKE outside the project. The source directory should contain the official training and test splits, each with `REAL` and `FAKE` folders:

```text
CIFAKE/
	train/REAL/
	train/FAKE/
	test/REAL/
	test/FAKE/
```

Prepare project splits with a fixed seed. The script stratifies validation images from the official training data and leaves the official test data untouched. It hard-links images when possible and copies them otherwise.

```bash
python src/dataset.py --source-dir /path/to/CIFAKE --output-dir data --val-fraction 0.1 --seed 42
```

The resulting folders are `data/{train,val,test}/{fake,real}`. Labels are `fake=0` and `real=1`. Running the command again with the same options is safe; to change the seed or validation fraction, first move or remove the previously generated project splits.

## Train and evaluate

Train the tuned baseline. The options below control learning rate, weight decay, dropout, and early stopping; the best validation checkpoint is saved under the ignored `checkpoints/` directory:

```bash
python train.py --model baseline --epochs 15 --batch-size 128 --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.3 --patience 3 --checkpoint checkpoints/tuned_baseline.pt
```

Try ImageNet-pretrained ResNet-18 with two stages. Start with its backbone frozen:

```bash
python train.py --model resnet18 --freeze-backbone --epochs 8 --batch-size 64 --learning-rate 1e-3 --weight-decay 1e-4 --patience 3 --checkpoint checkpoints/resnet18_frozen.pt
```

If frozen-backbone validation improves, run a lower learning-rate fine-tuning experiment:

```bash
python train.py --model resnet18 --epochs 12 --batch-size 64 --learning-rate 1e-5 --weight-decay 1e-4 --patience 3 --checkpoint checkpoints/resnet18_finetuned.pt
```

Evaluate each checkpoint once on the held-out test split and run single-image inference:

```bash
python evaluate.py --checkpoint checkpoints/tuned_baseline.pt --device auto
python evaluate.py --checkpoint checkpoints/resnet18_finetuned.pt --device auto
python predict.py path/to/image.jpg --checkpoint checkpoints/resnet18_finetuned.pt --device auto
```

Train and evaluate a classical classifier using both FFT radial-spectrum and low-frequency DCT features:

```bash
python train_frequency.py
python evaluate_frequency.py --checkpoint checkpoints/frequency_features.joblib
```

The frequency pipeline resizes RGB images to 32×32, summarizes each channel's FFT magnitudes into radial bands, and appends the low-frequency 8×8 DCT coefficients. A standardized logistic-regression model is fit on the training split, with validation accuracy reported and the model plus feature settings saved to the checkpoint. Tune the feature resolution and classifier with `--image-size`, `--fft-bins`, `--dct-size`, and `--regularization`. The test split is only used by `evaluate_frequency.py`.

To create and open a visualization for a dataset image, run:

```bash
python visualize_frequency.py data/test/fake/your_image.jpg --show
```

Replace the example path with an actual image file. The command saves `frequency_visualization.png` in the current directory; the PNG shows the resized input, log-magnitude FFT, radial FFT feature profile, and low-frequency DCT coefficients. Omit `--show` to save the PNG without opening it.

## Frequency-model robustness

After training the frequency classifier, evaluate clean test accuracy against paired JPEG compression, downscale/upscale, Gaussian blur, and synthetic screenshot-style transformations:

```bash
python evaluate_robustness.py --checkpoint checkpoints/frequency_features.joblib --data-dir data --output reports/frequency_robustness.csv
```

The script uses the same held-out test images and labels for every condition. It prints clean accuracy and accuracy drop in percentage points, and writes all results to CSV. Default levels are JPEG quality 95/80/60/40, resize scale 0.75/0.5/0.25, blur radius 0.5/1/2 input pixels, and screenshot display scale 0.9/0.75/0.6. Screenshot levels simulate an image framed on a neutral 16:9 display and resized to model input; they are not real screenshot captures. Results depend on the trained checkpoint and test split, so run this command to obtain actual measurements.

The baseline uses 32×32 images; ResNet-18 uses 224×224 images with ImageNet normalization. `--device auto` selects CUDA, Apple MPS, or CPU when available. For a valid comparison, keep split seed, epochs, data augmentation, and test data unchanged and report validation/test accuracy, F1, and ROC-AUC for each model.
