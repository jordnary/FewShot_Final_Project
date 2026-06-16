# -*- coding: utf-8 -*-
"""Generate report figures and derived tables from experiment CSV files."""

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"
TABLE_DIR = ROOT / "report" / "tables"


COLORS = {
    "one": "#2F6FBB",
    "five": "#D9793D",
    "gain": "#2E8B57",
    "loss": "#B84A62",
    "neutral": "#6B7280",
    "accent": "#7C3AED",
}


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with (ROOT / path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pct(value):
    return float(str(value).strip().rstrip("%"))


def fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def write_text(path, text):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def label_bars(ax, bars, values, digits=1, suffix="", offset=0.6):
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.{digits}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def generate_baseline(rows):
    labels = ["Quick\nConv64F", "Formal\nConv64F", "Boosted\nResNet12"]
    one = [pct(row["5-way 1-shot"]) for row in rows]
    five = [pct(row["5-way 5-shot"]) for row in rows]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    bars1 = ax.bar(x - width / 2, one, width, label="5-way 1-shot", color=COLORS["one"])
    bars2 = ax.bar(x + width / 2, five, width, label="5-way 5-shot", color=COLORS["five"])
    ax.set_title("ProtoNet Baseline Accuracy on CUB", fontsize=13, weight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="upper left")
    label_bars(ax, bars1, one, suffix="%")
    label_bars(ax, bars2, five, suffix="%")
    savefig(FIG_DIR / "baseline_accuracy_comparison.png")

    lines = [
        "# Baseline Results",
        "",
        "| Setting | Method | Backbone | Training | Test episodes | 5-way 1-shot | 5-way 5-shot |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {category} | {method} | {backbone} | {train_setting} | {test_episodes} | {one} | {five} |".format(
                category=row["category"],
                method=row["method"],
                backbone=row["backbone"],
                train_setting=row["train_setting"],
                test_episodes=row["test_episodes"],
                one=row["5-way 1-shot"],
                five=row["5-way 5-shot"],
            )
        )
    write_text(TABLE_DIR / "baseline_results.md", "\n".join(lines))


def generate_frofa(rows):
    labels = ["ProtoNet\nbaseline", "Joint\nFroFA", "Frozen\nFroFA"]
    one = [pct(row["5-way 1-shot"]) for row in rows]
    five = [pct(row["5-way 5-shot"]) for row in rows]
    delta_one = [value - one[0] for value in one]
    delta_five = [value - five[0] for value in five]
    x = np.arange(len(labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, one, width, label="5-way 1-shot", color=COLORS["one"])
    bars2 = ax.bar(x + width / 2, five, width, label="5-way 5-shot", color=COLORS["five"])
    ax.set_title("FroFAProtoNet Accuracy", fontsize=12, weight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(72, 88)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    label_bars(ax, bars1, one, digits=2, suffix="%", offset=0.12)
    label_bars(ax, bars2, five, digits=2, suffix="%", offset=0.12)

    ax = axes[1]
    delta_x = np.arange(2)
    joint = [delta_one[1], delta_five[1]]
    frozen = [delta_one[2], delta_five[2]]
    bars3 = ax.bar(delta_x - width / 2, joint, width, label="Joint FroFA", color=COLORS["gain"])
    bars4 = ax.bar(delta_x + width / 2, frozen, width, label="Frozen FroFA", color=COLORS["loss"])
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Delta vs ProtoNet", fontsize=12, weight="bold")
    ax.set_ylabel("Accuracy delta (pp)")
    ax.set_xticks(delta_x)
    ax.set_xticklabels(["1-shot", "5-shot"])
    ax.set_ylim(-0.4, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    for bars, values in [(bars3, joint), (bars4, frozen)]:
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.035 if value >= 0 else -0.055),
                f"{value:+.3f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8.5,
            )

    savefig(FIG_DIR / "frofa_reproduction_comparison.png")

    lines = [
        "# FroFA Reproduction Results",
        "",
        "| Method | Dataset | 5-way 1-shot | Delta 1-shot | 5-way 5-shot | Delta 5-shot |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(rows):
        lines.append(
            "| {method} | {dataset} | {one:.3f}% | {d_one:+.3f} | {five:.3f}% | {d_five:+.3f} |".format(
                method=row["method"],
                dataset=row["dataset"],
                one=one[index],
                d_one=delta_one[index],
                five=five[index],
                d_five=delta_five[index],
            )
        )
    write_text(TABLE_DIR / "frofa_reproduction_results.md", "\n".join(lines))


def generate_clip_summary(rows):
    labels = [f"{row['experiment']}\n{row['method']}" for row in rows]
    one = [float(row["5-way_1-shot"]) for row in rows]
    one_ci = [float(row["ci95_1-shot"]) for row in rows]
    five = [float(row["5-way_5-shot"]) for row in rows]
    five_ci = [float(row["ci95_5-shot"]) for row in rows]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.8), sharey=True)
    for ax, values, ci, title, color in [
        (axes[0], one, one_ci, "5-way 1-shot", COLORS["one"]),
        (axes[1], five, five_ci, "5-way 5-shot", COLORS["five"]),
    ]:
        ax.barh(y, values, xerr=ci, color=color, alpha=0.92, ecolor="#111827", capsize=3)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Accuracy (%)")
        ax.grid(axis="x", alpha=0.25)
        min_x = max(0, math.floor((min(values) - 5) / 10) * 10)
        ax.set_xlim(min_x, 100)
        for yi, value in enumerate(values):
            ax.text(value + 0.75, yi, f"{value:.1f}", va="center", fontsize=8.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.2)
    axes[0].invert_yaxis()
    fig.suptitle("CLIP-FroFA Experiment Summary with 95% CI", y=1.02, fontsize=13, weight="bold")
    savefig(FIG_DIR / "clip_frofa_summary_accuracy.png")

    lines = [
        "# CLIP-FroFA Final Summary",
        "",
        "| Experiment | Method | 5-way 1-shot | 95% CI | 5-way 5-shot | 95% CI | Main conclusion |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {method} | {one} | {one_ci} | {five} | {five_ci} | {conclusion} |".format(
                experiment=row["experiment"],
                method=row["method"],
                one=fmt(row["5-way_1-shot"]),
                one_ci=fmt(row["ci95_1-shot"]),
                five=fmt(row["5-way_5-shot"]),
                five_ci=fmt(row["ci95_5-shot"]),
                conclusion=row["main_conclusion"],
            )
        )
    write_text(TABLE_DIR / "clip_frofa_final_summary.md", "\n".join(lines))


def generate_sweep(rows):
    val_rows = [row for row in rows if row["stage"] == "val"]
    grouped = {}
    for row in val_rows:
        key = (
            row["alpha"],
            row["num_aug"],
            row["augmentations"],
            row["train_steps"],
            row["weight_decay"],
        )
        grouped.setdefault(key, {})[int(row["shot"])] = float(row["gain"])

    records = []
    for key, gains in grouped.items():
        mean_gain = float(np.mean([gains.get(1, 0.0), gains.get(5, 0.0)]))
        records.append((mean_gain, key, gains.get(1, 0.0), gains.get(5, 0.0)))
    records.sort(key=lambda item: item[0])

    labels = [f"a={key[0]}, n={key[1]}, {key[2]}" for _, key, _, _ in records]
    gain_one = [item[2] for item in records]
    gain_five = [item[3] for item in records]
    y = np.arange(len(records))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10.8, 8.6))
    ax.barh(y - height / 2, gain_one, height, label="Val 1-shot gain", color=COLORS["one"])
    ax.barh(y + height / 2, gain_five, height, label="Val 5-shot gain", color=COLORS["five"])
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_title("Post-LN Patch-Token FroFA Validation Sweep", fontsize=13, weight="bold")
    ax.set_xlabel("Accuracy gain vs MAP (pp)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(-1.05, max(max(gain_one), max(gain_five)) + 0.45)

    best_key = ("0.3", "8", "brightness", "40", "0.01")
    for idx, (_, key, _, _) in enumerate(records):
        if key == best_key:
            ax.text(
                -0.98,
                idx,
                "selected",
                va="center",
                ha="left",
                fontsize=9,
                color="white",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": COLORS["gain"],
                    "edgecolor": COLORS["gain"],
                    "linewidth": 0.8,
                },
            )
    savefig(FIG_DIR / "postln_validation_sweep_gain.png")

    lines = [
        "# Post-LN Validation-Selected FroFA Result",
        "",
        "| Stage | Shot | Alpha | Num aug | Augmentations | Train steps | Weight decay | MAP | FroFA + MAP | Gain |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    selected = [
        row
        for row in rows
        if row["alpha"] == "0.3"
        and row["num_aug"] == "8"
        and row["augmentations"] == "brightness"
    ]
    for row in selected:
        lines.append(
            "| {stage} | {shot} | {alpha} | {num_aug} | {augmentations} | {train_steps} | {weight_decay} | {map_acc} | {frofa_acc} | {gain} |".format(
                stage=row["stage"],
                shot=row["shot"],
                alpha=fmt(row["alpha"], 2),
                num_aug=row["num_aug"],
                augmentations=row["augmentations"],
                train_steps=row["train_steps"],
                weight_decay=fmt(row["weight_decay"], 2),
                map_acc=fmt(row["map_acc"]),
                frofa_acc=fmt(row["frofa_acc"]),
                gain=f"{float(row['gain']):+.3f}",
            )
        )
    write_text(TABLE_DIR / "postln_selected_config_results.md", "\n".join(lines))


def main():
    ensure_dirs()
    baseline_rows = read_csv("experiments/baseline/results/summary_all.csv")
    frofa_rows = read_csv("experiments/frofa_reproduction/results/frofa_vs_baseline_summary.csv")
    clip_rows = read_csv("experiments/clip_frofa_improvement/results/final_summary.csv")
    sweep_rows = read_csv("experiments/clip_frofa_improvement/results/clip_vit_b16_postln_patch_frofa_map_sweep.csv")

    generate_baseline(baseline_rows)
    generate_frofa(frofa_rows)
    generate_clip_summary(clip_rows)
    generate_sweep(sweep_rows)

    print("Generated figures in report/figures")
    print("Generated tables in report/tables")


if __name__ == "__main__":
    main()
