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

Train the baseline CNN. The best validation-accuracy checkpoint is saved under the ignored `checkpoints/` directory:

```bash
python train.py --epochs 10 --batch-size 128 --device auto
```

Evaluate the saved checkpoint once on the held-out test split:

```bash
python evaluate.py --checkpoint checkpoints/baseline_cnn.pt --device auto
```

The model resizes images to 32×32, uses horizontal flips for training augmentation, and reports test accuracy, F1, and ROC-AUC. `--device auto` selects CUDA, Apple MPS, or CPU when available.
