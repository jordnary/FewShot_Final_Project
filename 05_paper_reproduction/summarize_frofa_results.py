import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROFA_LOG_DIR = PROJECT_ROOT / "05_paper_reproduction" / "logs"
BASELINE_LOG_DIR = PROJECT_ROOT / "04_baseline_experiments" / "logs"
RESULT_DIR = PROJECT_ROOT / "05_paper_reproduction" / "results"

RUNS = [
    (
        "ProtoNet-ResNet12 baseline",
        BASELINE_LOG_DIR / "proto_cub_resnet12_boost_5way_1shot_cloud_console.log",
        BASELINE_LOG_DIR / "proto_cub_resnet12_boost_5way_5shot_cloud_console.log",
    ),
    (
        "FroFAProtoNet-ResNet12",
        FROFA_LOG_DIR / "frofa_proto_cub_resnet12_5way_1shot_cloud_console.log",
        FROFA_LOG_DIR / "frofa_proto_cub_resnet12_5way_5shot_cloud_console.log",
    ),
    (
        "FroFAProtoNet-ResNet12 frozen",
        FROFA_LOG_DIR / "frofa_proto_cub_resnet12_frozen_5way_1shot_cloud_console.log",
        FROFA_LOG_DIR / "frofa_proto_cub_resnet12_frozen_5way_5shot_cloud_console.log",
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
    return f"{float(matches[-1]):.2f}%"


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, one_shot_log, five_shot_log in RUNS:
        rows.append(
            (
                method,
                "CUB",
                extract_final_test_acc(one_shot_log),
                extract_final_test_acc(five_shot_log),
            )
        )

    csv_lines = ["method,dataset,5-way 1-shot,5-way 5-shot"]
    csv_lines.extend(",".join(row) for row in rows)
    (RESULT_DIR / "frofa_vs_baseline_summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    md_lines = [
        "# FroFA Reproduction Results",
        "",
        "| 方法 | 数据集 | 5-way 1-shot | 5-way 5-shot |",
        "| :--- | :--- | :--- | :--- |",
    ]
    md_lines.extend(f"| {m} | {d} | {a1} | {a5} |" for m, d, a1, a5 in rows)
    (RESULT_DIR / "frofa_vs_baseline_summary.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
