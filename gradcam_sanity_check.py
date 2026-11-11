import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageDraw

from gradcam import load_model, prepare_image, generate_gradcam
from train import choose_device


def pearson(a, b):
    a, b = a.ravel().astype(float), b.ravel().astype(float)
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def colored_overlay(original, heatmap):
    w, h = original.size
    m = Image.fromarray(np.uint8(np.clip(heatmap, 0, 1) * 255), 'L').resize((w, h))
    a = np.asarray(m, dtype=np.float32) / 255
    rgb = np.stack([255*a, 80*(1-a), 255*(1-a)], axis=-1).astype('uint8')
    return Image.blend(original, Image.fromarray(rgb, 'RGB'), .45)


def main():
    p = argparse.ArgumentParser(description='Grad-CAM last-block randomization sanity check')
    p.add_argument('--sample-csv', type=Path, default=Path('results/gradcam_review/sample_200.csv'))
    p.add_argument('--checkpoint', type=Path, default=Path('checkpoints/resnet18_cifake.pt'))
    p.add_argument('--output-dir', type=Path, default=Path('results/gradcam_sanity'))
    p.add_argument('--per-group', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda','mps'], default='auto')
    args = p.parse_args()
    if args.per_group < 1: p.error('--per-group must be >= 1')
    if not args.sample_csv.exists(): p.error(f'Missing {args.sample_csv}')
    device = choose_device(args.device)
    model, checkpoint, classes = load_model(args.checkpoint, device)
    random_model = copy.deepcopy(model)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    changed = []
    for name, module in random_model.backbone.layer4[-1].named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            module.reset_parameters()
            changed.append(name or '<root>')
    random_model.eval()
    print('Randomized modules in final residual block:', changed, flush=True)
    with args.sample_csv.open(newline='') as f:
        samples = list(csv.DictReader(f))
    selected = []
    for group in ('TP', 'TN', 'FP', 'FN'):
        subset = [r for r in samples if r['group'] == group]
        if len(subset) < args.per_group: p.error(f'Not enough {group} rows')
        selected.extend(subset[:args.per_group])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = []
    for i, row in enumerate(selected, 1):
        path = Path(row['path'])
        if not path.exists(): p.error(f'Missing image {path}')
        original, tensor = prepare_image(path, checkpoint)
        trained_pred, _, trained_cam = generate_gradcam(model, tensor, device)
        random_pred, _, random_cam = generate_gradcam(random_model, tensor, device)
        corr = pearson(trained_cam, random_cam)
        mae = float(np.abs(trained_cam - random_cam).mean())
        result = dict(row)
        result.update(trained_pred_index=trained_pred, randomized_pred_index=random_pred,
                      same_predicted_class=(trained_pred == random_pred),
                      heatmap_pearson=corr, heatmap_mean_abs_difference=mae)
        output.append(result)
        w, h = original.size
        panel = Image.new('RGB', (w*3, h+30), 'white')
        for j, (img, title) in enumerate(((original, 'Original'),
                                           (colored_overlay(original, trained_cam), 'Trained'),
                                           (colored_overlay(original, random_cam), 'Randomized'))):
            panel.paste(img, (w*j, 30))
            ImageDraw.Draw(panel).text((w*j+5, 7), title, fill='black')
        panel.save(args.output_dir / f'{row["group"]}_{i:02d}_comparison.png')
        print(f'{i}/{len(selected)} {row["group"]} corr={corr:.3f} MAE={mae:.3f} same_class={trained_pred == random_pred}', flush=True)
    fields = list(output[0])
    with (args.output_dir / 'sanity_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    valid = [r['heatmap_pearson'] for r in output if np.isfinite(r['heatmap_pearson'])]
    print('Mean heatmap correlation:', float(np.mean(valid)) if valid else 'undefined')
    print('Mean heatmap absolute difference:', float(np.mean([r['heatmap_mean_abs_difference'] for r in output])))
    print('IMPORTANT: This is a preliminary last-block weight sensitivity test; it uses each model\'s predicted class, so class changes can also alter maps.')
    print('Saved to', args.output_dir)


if __name__ == '__main__':
    main()