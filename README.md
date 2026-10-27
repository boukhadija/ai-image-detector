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

The baseline uses 32×32 images; ResNet-18 uses 224×224 images with ImageNet normalization. `--device auto` selects CUDA, Apple MPS, or CPU when available. For a valid comparison, keep split seed, epochs, data augmentation, and test data unchanged and report validation/test accuracy, F1, and ROC-AUC for each model.
