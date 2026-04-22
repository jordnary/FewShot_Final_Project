# -*- coding: utf-8 -*-
"""Validation sweep for patch-token FroFA + MAP.

This script mimics the paper workflow more closely than direct test tuning:
select FroFA/MAP hyperparameters on a validation token cache, then optionally
evaluate the selected configuration on a held-out test token cache.
"""
import argparse
import copy
import csv
from pathlib import Path

import numpy as np
import torch

from run_frofa_map_eval import evaluate_method, group_indices, load_tokens


def parse_aug_set(value):
    return [item for item in value.split("+") if item]


def build_configs(args):
    configs = []
    for alpha in args.alphas:
        for num_aug in args.num_augs:
            for aug_set in args.augmentation_sets:
                for train_steps in args.train_steps_grid:
                    for weight_decay in args.weight_decays:
                        configs.append(
                            {
                                "alpha": alpha,
                                "num_aug": num_aug,
                                "augmentations": parse_aug_set(aug_set),
                                "train_steps": train_steps,
                                "weight_decay": weight_decay,
                            }
                        )
    return configs


def eval_config(tokens, groups, args, shot, method, index):
    rng = np.random.default_rng(args.seed + shot * 100 + index)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed + shot * 1000 + index)
    return evaluate_method(tokens, groups, args, shot, method, rng, generator)


def run_rows(token_file, args, stage, configs):
    tokens, labels, _ = load_tokens(token_file)
    groups = group_indices(labels)
    rows = []
    baseline_cache = {}

    for index, config in enumerate(configs):
        print(
            "[{}] config {}/{}: alpha={}, num_aug={}, aug={}, steps={}, wd={}".format(
                stage,
                index + 1,
                len(configs),
                config["alpha"],
                config["num_aug"],
                "+".join(config["augmentations"]),
                config["train_steps"],
                config["weight_decay"],
            ),
            flush=True,
        )
        cfg_args = copy.copy(args)
        cfg_args.alpha = config["alpha"]
        cfg_args.num_aug = config["num_aug"]
        cfg_args.augmentations = config["augmentations"]
        cfg_args.train_steps = config["train_steps"]
        cfg_args.weight_decay = config["weight_decay"]

        for shot in args.shots:
            baseline_key = (shot, cfg_args.train_steps, cfg_args.weight_decay)
            if baseline_key not in baseline_cache:
                baseline_cache[baseline_key] = eval_config(
                    tokens, groups, cfg_args, shot, "baseline", index
                )
            map_mean, map_ci = baseline_cache[baseline_key]
            frofa_mean, frofa_ci = eval_config(
                tokens, groups, cfg_args, shot, "frofa", index
            )
            rows.append(
                {
                    "stage": stage,
                    "shot": shot,
                    "alpha": cfg_args.alpha,
                    "num_aug": cfg_args.num_aug,
                    "augmentations": "+".join(cfg_args.augmentations),
                    "train_steps": cfg_args.train_steps,
                    "weight_decay": cfg_args.weight_decay,
                    "map_acc": map_mean,
                    "map_ci95": map_ci,
                    "frofa_acc": frofa_mean,
                    "frofa_ci95": frofa_ci,
                    "gain": frofa_mean - map_mean,
                }
            )
            print(
                "[{}] shot {} done: MAP {:.3f}, FroFA {:.3f}, gain {:+.3f}".format(
                    stage, shot, map_mean, frofa_mean, frofa_mean - map_mean
                ),
                flush=True,
            )
    return rows


def select_best(rows, select_by):
    grouped = {}
    for row in rows:
        key = (
            row["alpha"],
            row["num_aug"],
            row["augmentations"],
            row["train_steps"],
            row["weight_decay"],
        )
        grouped.setdefault(key, []).append(row)

    scored = []
    for key, key_rows in grouped.items():
        mean_gain = float(np.mean([row["gain"] for row in key_rows]))
        mean_frofa = float(np.mean([row["frofa_acc"] for row in key_rows]))
        five_shot = [row["gain"] for row in key_rows if row["shot"] == 5]
        five_shot_gain = float(np.mean(five_shot)) if five_shot else mean_gain
        score = {
            "mean_gain": mean_gain,
            "mean_frofa_accuracy": mean_frofa,
            "5shot_gain": five_shot_gain,
        }[select_by]
        scored.append((score, key, mean_gain, mean_frofa, five_shot_gain))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0]


def write_rows(rows, output_prefix, write_md=True):
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = output_prefix.with_suffix(".md")
    columns = [
        "stage",
        "shot",
        "alpha",
        "num_aug",
        "augmentations",
        "train_steps",
        "weight_decay",
        "map_acc",
        "map_ci95",
        "frofa_acc",
        "frofa_ci95",
        "gain",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    if write_md:
        lines = [
            "# Patch-token FroFA + MAP validation sweep",
            "",
            "| Stage | Shot | Aug | Alpha | Num aug | Steps | WD | MAP | FroFA | Gain |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {stage} | {shot} | {augmentations} | {alpha:.3f} | {num_aug} | "
                "{train_steps} | {weight_decay:.4f} | {map_acc:.3f} | "
                "{frofa_acc:.3f} | {gain:+.3f} |".format(**row)
            )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-token-file", required=True)
    parser.add_argument("--test-token-file", default="")
    parser.add_argument(
        "--output-prefix",
        default="06_improvement/results/clip_vit_b16_postln_patch_frofa_map_sweep",
    )
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--test-episodes", type=int, default=600)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--num-augs", nargs="+", type=int, default=[4, 8])
    parser.add_argument(
        "--augmentation-sets",
        nargs="+",
        default=["brightness", "brightness+posterize", "brightness+contrast"],
    )
    parser.add_argument("--train-steps-grid", nargs="+", type=int, default=[40])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.01])
    parser.add_argument(
        "--select-by",
        choices=["mean_gain", "mean_frofa_accuracy", "5shot_gain"],
        default="mean_gain",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--map-heads", type=int, default=8)
    parser.add_argument("--map-queries", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--normalize-tokens", action="store_true")
    parser.add_argument("--resample-aug-each-step", action="store_true", default=True)
    parser.add_argument("--paired-episodes", action="store_true", default=True)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--no-md", action="store_true", help="Only write CSV output.")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = build_configs(args)
    val_rows = run_rows(args.val_token_file, args, "val", configs)
    _, best_key, _, _, _ = select_best(val_rows, args.select_by)
    best_config = {
        "alpha": best_key[0],
        "num_aug": best_key[1],
        "augmentations": parse_aug_set(best_key[2]),
        "train_steps": best_key[3],
        "weight_decay": best_key[4],
    }
    print(f"Best validation config by {args.select_by}: {best_config}", flush=True)

    rows = list(val_rows)
    if args.test_token_file:
        test_args = copy.copy(args)
        test_args.episodes = args.test_episodes
        rows.extend(run_rows(args.test_token_file, test_args, "test", [best_config]))

    csv_path, md_path = write_rows(rows, args.output_prefix, write_md=not args.no_md)
    print(f"saved {csv_path}", flush=True)
    if not args.no_md:
        print(f"saved {md_path}", flush=True)


if __name__ == "__main__":
    main()
