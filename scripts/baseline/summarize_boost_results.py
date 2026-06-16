import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "artifacts" / "logs" / "baseline"
RESULT_DIR = PROJECT_ROOT / "experiments" / "baseline" / "results"

RUNS = [
    (
        "ProtoNet-ResNet12",
        "CUB",
        "proto_cub_resnet12_boost_5way_1shot_cloud_console.log",
        "proto_cub_resnet12_boost_5way_5shot_cloud_console.log",
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


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, dataset, one_shot_log, five_shot_log in RUNS:
        one_shot = extract_final_test_acc(LOG_DIR / one_shot_log)
        five_shot = extract_final_test_acc(LOG_DIR / five_shot_log)
        if one_shot == "missing" and five_shot == "missing":
            continue
        rows.append((method, dataset, one_shot, five_shot))

    if not rows:
        print("No boosted cloud logs found yet.")
        return

    csv_lines = ["method,dataset,5-way 1-shot,5-way 5-shot"]
    csv_lines.extend(",".join(row) for row in rows)
    (RESULT_DIR / "baseline_boost_summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    print("Wrote", RESULT_DIR / "baseline_boost_summary.csv")


if __name__ == "__main__":
    main()
