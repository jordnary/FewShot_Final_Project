# -*- coding: utf-8 -*-
"""Episodic linear-probe evaluation on cached frozen features.

This evaluates the improvement route for Stage 6:

1. no-FroFA: train a closed-form L2 linear classifier on support features.
2. FroFA: augment support features in frozen feature space, then train the same
   closed-form linear classifier.

The implementation uses global CLIP features. Therefore the augmentation is a
vector-feature FroFA variant: min-max map each feature vector to [0, 1], apply
brightness or contrast perturbations, and invert the map.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch


def load_features(path):
    data = np.load(path, allow_pickle=True)
    features = data["features"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    class_names = data["class_names"].astype(str)
    return features, labels, class_names


def group_indices(labels):
    groups = {}
    for index, label in enumerate(labels.tolist()):
        groups.setdefault(label, []).append(index)
    return {label: np.asarray(indices, dtype=np.int64) for label, indices in groups.items()}


def sample_episode(groups, way, shot, query, rng):
    classes = np.asarray(sorted(groups.keys()))
    chosen = rng.choice(classes, size=way, replace=False)
    support_indices = []
    query_indices = []
    support_targets = []
    query_targets = []

    total_needed = shot + query
    for target, class_id in enumerate(chosen.tolist()):
        class_indices = groups[class_id]
        replace = len(class_indices) < total_needed
        picked = rng.choice(class_indices, size=total_needed, replace=replace)
        support_indices.extend(picked[:shot].tolist())
        query_indices.extend(picked[shot:].tolist())
        support_targets.extend([target] * shot)
        query_targets.extend([target] * query)

    return (
        np.asarray(support_indices, dtype=np.int64),
        np.asarray(query_indices, dtype=np.int64),
        np.asarray(support_targets, dtype=np.int64),
        np.asarray(query_targets, dtype=np.int64),
    )


def normalize_rows(x):
    return torch.nn.functional.normalize(x, dim=-1)


def frofa_augment_vectors(x, num_aug, alpha, augmentations, generator):
    if num_aug <= 0:
        return x

    x_min = x.amin(dim=1, keepdim=True)
    x_max = x.amax(dim=1, keepdim=True)
    scale = (x_max - x_min).clamp_min(1e-6)
    x01 = (x - x_min) / scale
    augmented = [x]

    for aug_name in augmentations:
        for _ in range(num_aug):
            if aug_name == "brightness":
                delta = (
                    torch.rand(
                        x01.shape,
                        device=x.device,
                        dtype=x.dtype,
                        generator=generator,
                    )
                    * 2.0
                    - 1.0
                ) * alpha
                aug01 = (x01 + delta).clamp(0.0, 1.0)
            elif aug_name == "contrast":
                center = x01.mean(dim=1, keepdim=True)
                factor = 1.0 + (
                    torch.rand(
                        (x01.size(0), 1),
                        device=x.device,
                        dtype=x.dtype,
                        generator=generator,
                    )
                    * 2.0
                    - 1.0
                ) * alpha
                aug01 = ((x01 - center) * factor.clamp_min(0.05) + center).clamp(
                    0.0, 1.0
                )
            else:
                raise ValueError(f"Unsupported augmentation: {aug_name}")
            augmented.append(aug01 * scale + x_min)

    return torch.cat(augmented, dim=0)


def fit_ridge_classifier(x_support, y_support, way, ridge_lambda):
    one_hot = torch.nn.functional.one_hot(y_support, num_classes=way).float()
    ones = torch.ones((x_support.size(0), 1), device=x_support.device)
    design = torch.cat([x_support, ones], dim=1)
    gram = design.T @ design
    eye = torch.eye(gram.size(0), device=x_support.device, dtype=gram.dtype)
    eye[-1, -1] = 0.0
    rhs = design.T @ one_hot
    return torch.linalg.solve(gram + ridge_lambda * eye, rhs)


def predict_linear(x_query, weights):
    ones = torch.ones((x_query.size(0), 1), device=x_query.device)
    design = torch.cat([x_query, ones], dim=1)
    return design @ weights


def evaluate_method(
    features,
    groups,
    args,
    shot,
    method,
    rng,
    torch_generator,
):
    device = torch.device(args.device)
    accuracies = []

    feature_tensor = torch.from_numpy(features).to(device)
    if args.normalize:
        feature_tensor = normalize_rows(feature_tensor)

    for _ in range(args.episodes):
        support_idx, query_idx, support_y, query_y = sample_episode(
            groups, args.way, shot, args.query, rng
        )
        x_support = feature_tensor[torch.from_numpy(support_idx).to(device)]
        x_query = feature_tensor[torch.from_numpy(query_idx).to(device)]
        y_support = torch.from_numpy(support_y).long().to(device)
        y_query = torch.from_numpy(query_y).long().to(device)

        if method == "frofa":
            x_support = frofa_augment_vectors(
                x_support,
                args.num_aug,
                args.alpha,
                args.augmentations,
                torch_generator,
            )
            y_support = y_support.repeat(1 + args.num_aug * len(args.augmentations))

        weights = fit_ridge_classifier(
            x_support, y_support, args.way, args.ridge_lambda
        )
        logits = predict_linear(x_query, weights)
        acc = (logits.argmax(dim=1) == y_query).float().mean().item() * 100.0
        accuracies.append(acc)

    values = np.asarray(accuracies, dtype=np.float64)
    mean = values.mean()
    ci95 = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    return mean, ci95


def write_results(rows, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "backbone", "classifier", "setting", "accuracy", "ci95"])
        writer.writerows(rows)

    return csv_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-file",
        default="06_improvement/results/features/cub_test_clip_vit_b16.npz",
    )
    parser.add_argument("--output-prefix", default="06_improvement/results/clip_frofa_linear")
    parser.add_argument("--backbone-name", default="CLIP ViT-B/16")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--num-aug", type=int, default=8)
    parser.add_argument(
        "--augmentations", nargs="+", default=["brightness"], choices=["brightness", "contrast"]
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    features, labels, _ = load_features(args.feature_file)
    groups = group_indices(labels)
    if len(groups) < args.way:
        raise RuntimeError(f"Need at least {args.way} classes, found {len(groups)}")

    rows = []
    for shot in args.shots:
        for method, method_name in [
            ("baseline", "no-FroFA"),
            ("frofa", "FroFA"),
        ]:
            rng = np.random.default_rng(args.seed + shot * 100 + len(rows))
            torch_generator = torch.Generator(device=args.device)
            torch_generator.manual_seed(args.seed + shot * 1000 + len(rows))
            mean, ci95 = evaluate_method(
                features, groups, args, shot, method, rng, torch_generator
            )
            rows.append(
                [
                    method_name,
                    args.backbone_name,
                    "closed-form L2 linear probe",
                    f"{args.way}-way {shot}-shot",
                    mean,
                    ci95,
                ]
            )
            print(
                f"{method_name} {args.way}-way {shot}-shot: {mean:.3f}% +/- {ci95:.3f}%"
            )

    csv_path = write_results(rows, args.output_prefix)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
