# -*- coding: utf-8 -*-
"""Episode-level MAP-head evaluation on cached CLIP patch tokens.

For each sampled 5-way episode, this script trains a small Multi-head Attention
Pooling (MAP) classifier on support patch tokens, then evaluates query tokens.
The FroFA condition augments support patch tokens with brightness c2FroFA before
the MAP head is trained.
"""
import argparse
import csv
import math
import os
from pathlib import Path

for env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    if not os.environ.get(env_name, "").isdigit() or int(os.environ.get(env_name, "0")) <= 0:
        os.environ[env_name] = "8"

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def load_tokens(path):
    data = np.load(path, allow_pickle=True)
    tokens = data["tokens"].astype(np.float32)
    labels = data["labels"].astype(np.int64)
    class_names = data["class_names"].astype(str)
    return tokens, labels, class_names


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


def build_episodes(groups, way, shot, query, episode_count, seed):
    rng = np.random.default_rng(seed)
    return [
        sample_episode(groups, way, shot, query, rng)
        for _ in range(episode_count)
    ]


def frofa_augment_patch_tokens(tokens, num_aug, alpha, augmentations, generator):
    """Apply c2-style FroFA on N x P x C patch tokens."""
    if num_aug <= 0:
        return tokens

    feat_min = tokens.amin(dim=1, keepdim=True)
    feat_max = tokens.amax(dim=1, keepdim=True)
    scale = (feat_max - feat_min).clamp_min(1e-6)
    tokens01 = (tokens - feat_min) / scale
    augmented = [tokens]

    for aug_name in augmentations:
        for _ in range(num_aug):
            if aug_name == "brightness":
                delta = (
                    torch.rand(
                        (tokens01.size(0), 1, tokens01.size(2)),
                        device=tokens.device,
                        dtype=tokens.dtype,
                        generator=generator,
                    )
                    * 2.0
                    - 1.0
                ) * alpha
                aug01 = (tokens01 + delta).clamp(0.0, 1.0)
            elif aug_name == "contrast":
                center = tokens01.mean(dim=1, keepdim=True)
                factor = 1.0 + (
                    torch.rand(
                        (tokens01.size(0), 1, tokens01.size(2)),
                        device=tokens.device,
                        dtype=tokens.dtype,
                        generator=generator,
                    )
                    * 2.0
                    - 1.0
                ) * alpha
                aug01 = ((tokens01 - center) * factor.clamp_min(0.05) + center).clamp(
                    0.0, 1.0
                )
            else:
                raise ValueError(f"Unsupported FroFA augmentation: {aug_name}")
            augmented.append(aug01 * scale + feat_min)

    return torch.cat(augmented, dim=0)


class MAPClassifier(nn.Module):
    def __init__(self, feature_dim, way, num_heads=8, num_queries=1, dropout=0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_queries, feature_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, way)

    def pool(self, tokens):
        query = self.query.unsqueeze(0).expand(tokens.size(0), -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        pooled = pooled.mean(dim=1)
        return self.norm(pooled)

    def forward(self, tokens):
        return self.classifier(self.pool(tokens))


def train_episode_model(x_support, y_support, args, method, generator):
    if method == "frofa" and not args.resample_aug_each_step:
        x_train = frofa_augment_patch_tokens(
            x_support, args.num_aug, args.alpha, args.augmentations, generator
        )
        repeat_times = 1 + args.num_aug * len(args.augmentations)
        y_train = y_support.repeat(repeat_times)
    else:
        x_train = x_support
        y_train = y_support

    model = MAPClassifier(
        feature_dim=x_train.size(-1),
        way=args.way,
        num_heads=args.map_heads,
        num_queries=args.map_queries,
        dropout=args.dropout,
    ).to(x_train.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    model.train()
    for _ in range(args.train_steps):
        if method == "frofa" and args.resample_aug_each_step:
            x_train = frofa_augment_patch_tokens(
                x_support, args.num_aug, args.alpha, args.augmentations, generator
            )
            repeat_times = 1 + args.num_aug * len(args.augmentations)
            y_train = y_support.repeat(repeat_times)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()
    return model


def evaluate_method(tokens, groups, args, shot, method, rng, generator):
    device = torch.device(args.device)
    token_tensor = torch.from_numpy(tokens).to(device=device, dtype=torch.float32)
    if args.normalize_tokens:
        token_tensor = F.normalize(token_tensor, dim=-1)

    accuracies = []
    if args.paired_episodes:
        episodes = build_episodes(
            groups,
            args.way,
            shot,
            args.query,
            args.episodes,
            args.seed + shot * 100,
        )
    else:
        episodes = None

    for episode_id in range(args.episodes):
        if episodes is None:
            support_idx, query_idx, support_y, query_y = sample_episode(
                groups, args.way, shot, args.query, rng
            )
        else:
            support_idx, query_idx, support_y, query_y = episodes[episode_id]
        x_support = token_tensor[torch.from_numpy(support_idx).to(device)]
        x_query = token_tensor[torch.from_numpy(query_idx).to(device)]
        y_support = torch.from_numpy(support_y).long().to(device)
        y_query = torch.from_numpy(query_y).long().to(device)

        episode_seed = args.seed + shot * 100000 + episode_id
        torch.manual_seed(episode_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(episode_seed)

        model = train_episode_model(x_support, y_support, args, method, generator)
        model.eval()
        with torch.no_grad():
            logits = model(x_query)
            acc = (logits.argmax(dim=1) == y_query).float().mean().item() * 100.0
        accuracies.append(acc)

        if args.log_interval > 0 and (episode_id + 1) % args.log_interval == 0:
            print(
                f"{method} {args.way}-way {shot}-shot episode "
                f"{episode_id + 1}/{args.episodes}: running mean {np.mean(accuracies):.3f}%"
            )

    values = np.asarray(accuracies, dtype=np.float64)
    mean = values.mean()
    ci95 = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    return mean, ci95


def write_results(rows, output_prefix):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "backbone", "head", "setting", "accuracy", "ci95"])
        writer.writerows(rows)

    lines = [
        "# CLIP patch-token FroFA + MAP-head results",
        "",
        "| Method | Backbone | Head | Setting | Accuracy | 95% CI |",
        "|---|---|---|---|---:|---:|",
    ]
    for method, backbone, head, setting, accuracy, ci95 in rows:
        lines.append(
            f"| {method} | {backbone} | {head} | {setting} | {accuracy:.3f}% | +/- {ci95:.3f}% |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token-file",
        default="06_improvement/results/features/cub_test_clip_vit_b16_patch_tokens.npz",
    )
    parser.add_argument(
        "--output-prefix",
        default="06_improvement/results/clip_vit_b16_patch_frofa_map_cub",
    )
    parser.add_argument("--backbone-name", default="CLIP ViT-B/16 patch tokens")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--map-heads", type=int, default=8)
    parser.add_argument("--map-queries", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--num-aug", type=int, default=8)
    parser.add_argument(
        "--resample-aug-each-step",
        action="store_true",
        default=True,
        help="Resample FroFA support tokens every MAP optimization step.",
    )
    parser.add_argument(
        "--fixed-aug-per-episode",
        action="store_false",
        dest="resample_aug_each_step",
        help="Sample FroFA support tokens once per episode instead.",
    )
    parser.add_argument(
        "--augmentations", nargs="+", default=["brightness"], choices=["brightness", "contrast"]
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--normalize-tokens", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--paired-episodes",
        action="store_true",
        default=True,
        help="Evaluate all methods on the same episode samples for each shot.",
    )
    parser.add_argument(
        "--unpaired-episodes",
        action="store_false",
        dest="paired_episodes",
        help="Use independent episode samples for each method.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tokens, labels, _ = load_tokens(args.token_file)
    groups = group_indices(labels)
    if len(groups) < args.way:
        raise RuntimeError(f"Need at least {args.way} classes, found {len(groups)}")

    rows = []
    for shot in args.shots:
        for method, method_name in [("baseline", "MAP"), ("frofa", "FroFA + MAP")]:
            rng = np.random.default_rng(args.seed + shot * 100 + len(rows))
            generator = torch.Generator(device=args.device)
            generator.manual_seed(args.seed + shot * 1000 + len(rows))
            mean, ci95 = evaluate_method(tokens, groups, args, shot, method, rng, generator)
            rows.append(
                [
                    method_name,
                    args.backbone_name,
                    "episode-trained MAP head",
                    f"{args.way}-way {shot}-shot",
                    mean,
                    ci95,
                ]
            )
            print(
                f"{method_name} {args.way}-way {shot}-shot: {mean:.3f}% +/- {ci95:.3f}%"
            )

    csv_path, md_path = write_results(rows, args.output_prefix)
    print(f"saved {csv_path}")
    print(f"saved {md_path}")


if __name__ == "__main__":
    main()
