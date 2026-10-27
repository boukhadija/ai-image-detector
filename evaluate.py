"""Evaluate a trained CIFAKE baseline on the held-out test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

from models.detector import create_model
from src.dataset import create_dataloaders
from train import choose_device


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate a CIFAKE CNN on the test split")
	parser.add_argument("--data-dir", type=Path, default=Path("data"))
	parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/baseline_cnn.pt"))
	parser.add_argument("--batch-size", type=int, default=128)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
	args = parser.parse_args()
	if not args.checkpoint.is_file():
		parser.error(f"Checkpoint does not exist: {args.checkpoint}")

	device = choose_device(args.device)
	checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
	model_name = checkpoint.get("model_name", "baseline")
	image_size = checkpoint.get("image_size", 32)
	imagenet_normalization = checkpoint.get("imagenet_normalization", model_name == "resnet18")
	loaders, class_to_idx = create_dataloaders(
		args.data_dir,
		args.batch_size,
		args.num_workers,
		image_size=image_size,
		imagenet_normalization=imagenet_normalization,
	)
	if checkpoint["class_to_idx"] != class_to_idx:
		raise ValueError("Checkpoint class mapping does not match the dataset class mapping")

	model = create_model(model_name, num_classes=len(class_to_idx), pretrained=False).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.eval()
	all_labels = []
	all_predictions = []
	all_fake_scores = []
	with torch.inference_mode():
		for images, labels in loaders["test"]:
			logits = model(images.to(device))
			all_labels.extend(labels.tolist())
			all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
			all_fake_scores.extend(torch.softmax(logits, dim=1)[:, class_to_idx["fake"]].cpu().tolist())

	if not all_labels:
		raise ValueError("The test split is empty")
	fake_labels = [
    1 if label == class_to_idx["fake"] else 0
    for label in all_labels
	]
	
	confusion_matrix_cm=confusion_matrix(all_labels, all_predictions, labels=[class_to_idx["fake"], class_to_idx["real"]])
	tp, fn = confusion_matrix_cm[0]
	fp, tn = confusion_matrix_cm[1]
	metrics = {
			"accuracy": accuracy_score(all_labels, all_predictions),
			"f1": f1_score(all_labels, all_predictions, pos_label=class_to_idx["fake"]),
			"auc_roc": roc_auc_score(fake_labels, all_fake_scores),
			"precision": precision_score( all_labels, all_predictions, pos_label=class_to_idx["fake"]),
			"recall": recall_score(all_labels, all_predictions, pos_label=class_to_idx["fake"]),
		
			}
	print(f"Test images: {len(all_labels)}; checkpoint epoch: {checkpoint['epoch']}")
	for metric_name, metric_value in metrics.items():
		print(f"{metric_name}: {metric_value:.4f}")
	print(f"TP: {tp}")
	print(f"FN: {fn}")
	print(f"FP: {fp}")
	print(f"TN: {tn}")


if __name__ == "__main__":
	main()