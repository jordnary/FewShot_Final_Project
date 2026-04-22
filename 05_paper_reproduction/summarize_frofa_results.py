import csv
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "05_paper_reproduction" / "logs"
BASELINE_LOG_DIR = PROJECT_ROOT / "04_baseline_experiments" / "logs"
RESULT_DIR = PROJECT_ROOT / "05_paper_reproduction" / "results"


@dataclass(frozen=True)
class Experiment:
    name: str
    method: str
    dataset: str
    backbone: str
    mode: str
    shot: str
    train_setting: str
    test_episodes: int
    log_file: str


EXPERIMENTS = [
    Experiment(
        name="frofa_proto_cub_resnet12_5way_1shot_cloud",
        method="FroFAProtoNet",
        dataset="CUB",
        backbone="ResNet12",
        mode="joint",
        shot="5-way 1-shot",
        train_setting="120 epochs; 2000 train episodes/epoch",
        test_episodes=1000,
        log_file="frofa_proto_cub_resnet12_5way_1shot_cloud_console.log",
    ),
    Experiment(
        name="frofa_proto_cub_resnet12_5way_5shot_cloud",
        method="FroFAProtoNet",
        dataset="CUB",
        backbone="ResNet12",
        mode="joint",
        shot="5-way 5-shot",
        train_setting="120 epochs; 2000 train episodes/epoch",
        test_episodes=1000,
        log_file="frofa_proto_cub_resnet12_5way_5shot_cloud_console.log",
    ),
    Experiment(
        name="frofa_proto_cub_resnet12_frozen_5way_1shot_cloud",
        method="FroFAProtoNet",
        dataset="CUB",
        backbone="ResNet12",
        mode="frozen",
        shot="5-way 1-shot",
        train_setting="30 epochs; 2000 train episodes/epoch",
        test_episodes=1000,
        log_file="frofa_proto_cub_resnet12_frozen_5way_1shot_cloud_console.log",
    ),
    Experiment(
        name="frofa_proto_cub_resnet12_frozen_5way_5shot_cloud",
        method="FroFAProtoNet",
        dataset="CUB",
        backbone="ResNet12",
        mode="frozen",
        shot="5-way 5-shot",
        train_setting="30 epochs; 2000 train episodes/epoch",
        test_episodes=1000,
        log_file="frofa_proto_cub_resnet12_frozen_5way_5shot_cloud_console.log",
    ),
]


def extract_final_test_acc(log_path: Path) -> str:
    if not log_path.exists():
        return "missing"

    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "End of experiment" not in text:
        return "unfinished"

    testing_pos = text.rfind("Testing on the test")
    search_text = text[testing_pos:] if testing_pos >= 0 else text
    matches = re.findall(r"\*\s+Acc@1\s+([0-9]+(?:\.[0-9]+)?)", search_text)
    if not matches:
        return "not found"
    return f"{float(matches[-1]):.3f}%"


def extract_runtime(log_path: Path) -> str:
    if not log_path.exists():
        return "missing"

    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"End of experiment, took\s+([0-9:]+)", text)
    if not matches:
        return "not found"
    return matches[-1]


def build_row(experiment: Experiment) -> dict[str, str]:
    log_path = LOG_DIR / experiment.log_file
    status = "finished" if log_path.exists() and "End of experiment" in log_path.read_text(
        encoding="utf-8", errors="replace"
    ) else "unfinished" if log_path.exists() else "missing"

    return {
        "experiment": experiment.name,
        "method": experiment.method,
        "dataset": experiment.dataset,
        "backbone": experiment.backbone,
        "mode": experiment.mode,
        "shot": experiment.shot,
        "train_setting": experiment.train_setting,
        "test_episodes": str(experiment.test_episodes),
        "final_test_acc": extract_final_test_acc(log_path),
        "runtime": extract_runtime(log_path),
        "status": status,
        "log_file": experiment.log_file,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "experiment",
        "method",
        "dataset",
        "backbone",
        "mode",
        "shot",
        "train_setting",
        "test_episodes",
        "final_test_acc",
        "runtime",
        "status",
        "log_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["method", "dataset", "5-way 1-shot", "5-way 5-shot"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def value_by_mode_and_shot(rows: list[dict[str, str]], mode: str, shot: str) -> str:
    for row in rows:
        if row["mode"] == mode and row["shot"] == shot:
            return row["final_test_acc"]
    return "missing"


def build_comparison_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    baseline_one = extract_final_test_acc(
        BASELINE_LOG_DIR / "proto_cub_resnet12_boost_5way_1shot_cloud_console.log"
    )
    baseline_five = extract_final_test_acc(
        BASELINE_LOG_DIR / "proto_cub_resnet12_boost_5way_5shot_cloud_console.log"
    )
    return [
        {
            "method": "ProtoNet-ResNet12 baseline",
            "dataset": "CUB",
            "5-way 1-shot": baseline_one,
            "5-way 5-shot": baseline_five,
        },
        {
            "method": "FroFAProtoNet-ResNet12",
            "dataset": "CUB",
            "5-way 1-shot": value_by_mode_and_shot(rows, "joint", "5-way 1-shot"),
            "5-way 5-shot": value_by_mode_and_shot(rows, "joint", "5-way 5-shot"),
        },
        {
            "method": "FroFAProtoNet-ResNet12 frozen",
            "dataset": "CUB",
            "5-way 1-shot": value_by_mode_and_shot(rows, "frozen", "5-way 1-shot"),
            "5-way 5-shot": value_by_mode_and_shot(rows, "frozen", "5-way 5-shot"),
        },
    ]


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [build_row(experiment) for experiment in EXPERIMENTS]

    for row in rows:
        write_csv(RESULT_DIR / f"{row['experiment']}.csv", [row])

    comparison_rows = build_comparison_rows(rows)
    write_comparison_csv(RESULT_DIR / "frofa_vs_baseline_summary.csv", comparison_rows)

    print("Wrote per-experiment CSV results and FroFA baseline summary to", RESULT_DIR)


if __name__ == "__main__":
    main()
