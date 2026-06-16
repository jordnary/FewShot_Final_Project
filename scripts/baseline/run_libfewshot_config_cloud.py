import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBFEWSHOT_ROOT = PROJECT_ROOT / "third_party" / "LibFewShot"


def render_config(config_path: Path) -> Path:
    text = config_path.read_text(encoding="utf-8")
    project_root = PROJECT_ROOT.as_posix()
    text = text.replace("${PROJECT_ROOT}", project_root)

    temp_dir = Path(tempfile.mkdtemp(prefix="libfewshot_config_"))
    rendered = temp_dir / config_path.name
    rendered.write_text(text, encoding="utf-8")
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Absolute or project-relative YAML config path")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = render_config(config_path)

    os.chdir(LIBFEWSHOT_ROOT)
    sys.path.insert(0, str(LIBFEWSHOT_ROOT))

    from core.config import Config
    from core import Trainer

    sys.argv = [sys.argv[0]]
    config = Config(str(config_path)).get_config_dict()

    if config["n_gpu"] > 1:
        import torch

        os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_ids"])
        torch.multiprocessing.spawn(
            lambda rank, cfg: Trainer(rank, cfg).train_loop(rank),
            nprocs=config["n_gpu"],
            args=(config,),
        )
    else:
        Trainer(0, config).train_loop(0)


if __name__ == "__main__":
    main()
