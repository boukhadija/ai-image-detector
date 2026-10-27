"""Train the CIFAKE baseline and save its best validation checkpoint."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn

from models.detector import create_model
from src.dataset import create_dataloaders


def choose_device(requested: str) -> torch.device:
	if requested != "auto":
		device = torch.device(requested)
		if device.type == "cuda" and not torch.cuda.is_available():
			raise RuntimeError("CUDA was requested but is not available")
		if device.type == "mps" and not torch.backends.mps.is_available():
			raise RuntimeError("MPS was requested but is not available")
		return device
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
	is_training = optimizer is not None
	model.train(is_training)
	loss_total = 0.0
	correct_total = 0
	sample_total = 0

	context = torch.enable_grad() if is_training else torch.inference_mode()
	with context:
		for images, labels in loader:
			images = images.to(device)
			labels = labels.to(device)
			if is_training:
				optimizer.zero_grad(set_to_none=True)

			logits = model(images)
			loss = criterion(logits, labels)
			if is_training:
				loss.backward()
				optimizer.step()

			sample_total += labels.size(0)
			loss_total += loss.item() * labels.size(0)
			correct_total += (logits.argmax(dim=1) == labels).sum().item()

	if sample_total == 0:
		raise ValueError("A data loader is empty")
	return loss_total / sample_total, correct_total / sample_total


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a baseline CIFAKE CNN")
	parser.add_argument("--data-dir", type=Path, default=Path("data"))
	parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/baseline_cnn.pt"))
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
	parser.add_argument("--model", choices=("baseline", "resnet18"), default="baseline")
	parser.add_argument("--image-size", type=int, default=None)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--freeze-backbone", action="store_true")
	parser.add_argument("--no-pretrained", action="store_true")
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--patience", type=int, default=3)
	args = parser.parse_args()
	if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
		parser.error("epochs and batch-size must be positive, and learning-rate must be greater than 0")

	random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(args.seed)
	device = choose_device(args.device)
	is_pretrained_model = args.model == "resnet18"
	image_size = args.image_size or (224 if is_pretrained_model else 32)
	loaders, class_to_idx = create_dataloaders(
		args.data_dir,
		args.batch_size,
		args.num_workers,
		args.seed,
		image_size=image_size,
		imagenet_normalization=is_pretrained_model,
	)

	model = create_model(
		args.model,
		num_classes=len(class_to_idx),
		pretrained=not args.no_pretrained,
		freeze_backbone=args.freeze_backbone,
	).to(device)
	if args.dropout < 0 or args.dropout >= 1:
		parser.error("dropout must be in [0, 1)")
	if hasattr(model, "classifier") and isinstance(model.classifier[-1], nn.Linear):
		for module in model.classifier:
			if isinstance(module, nn.Dropout):
				module.p = args.dropout
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(
		filter(lambda parameter: parameter.requires_grad, model.parameters()),
		lr=args.learning_rate,
		weight_decay=args.weight_decay,
	)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
	best_val_accuracy = -1.0
	epochs_without_improvement = 0
	best_val_accuracy = -1.0
	args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

	print(f"Device: {device}; class mapping: {class_to_idx}")
	for epoch in range(1, args.epochs + 1):
		train_loss, train_accuracy = run_epoch(
			model, loaders["train"], criterion, device, optimizer
		)
		val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
		print(
			f"Epoch {epoch:02d}/{args.epochs} "
			f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
			f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
		)
		scheduler.step(val_accuracy)
		if val_accuracy > best_val_accuracy:
			best_val_accuracy = val_accuracy
			epochs_without_improvement = 0
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"class_to_idx": class_to_idx,
					"model_name": args.model,
					"image_size": image_size,
					"imagenet_normalization": is_pretrained_model,
					"epoch": epoch,
					"val_accuracy": val_accuracy,
				},
				args.checkpoint,
			)
		else:
			epochs_without_improvement += 1
			if epochs_without_improvement >= args.patience:
				print(f"Early stopping after epoch {epoch}")
				break

	print(f"Best validation accuracy: {best_val_accuracy:.4f}")
	print(f"Checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
	main()