import os
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
RUNNER = WORKSPACE / "04_baseline_experiments" / "run_libfewshot_config.py"
LOG_DIR = WORKSPACE / "04_baseline_experiments" / "logs"

JOBS = [
    (
        "proto_cub_conv64f_formal_5way_1shot",
        WORKSPACE
        / "04_baseline_experiments"
        / "configs"
        / "proto_cub_conv64f_formal_5way_1shot.yaml",
    ),
    (
        "proto_cub_conv64f_formal_5way_5shot",
        WORKSPACE
        / "04_baseline_experiments"
        / "configs"
        / "proto_cub_conv64f_formal_5way_5shot.yaml",
    ),
]


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    for name, config in JOBS:
        log_path = LOG_DIR / f"{name}_console.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"Starting {name}\n")
            log.flush()
            completed = subprocess.run(
                [sys.executable, str(RUNNER), str(config)],
                cwd=str(WORKSPACE),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(f"\nFinished {name} with exit code {completed.returncode}\n")
            log.flush()
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
