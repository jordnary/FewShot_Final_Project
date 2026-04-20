import argparse
import os
import sys

import torch


WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBFEWSHOT_ROOT = os.path.join(WORKSPACE, "02_libfewshot", "LibFewShot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Absolute or workspace-relative config path")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(WORKSPACE, config_path)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.chdir(LIBFEWSHOT_ROOT)
    sys.path.insert(0, LIBFEWSHOT_ROOT)

    from core.config import Config
    from core import Trainer

    sys.argv = [sys.argv[0]]
    config = Config(config_path).get_config_dict()

    if config["n_gpu"] > 1:
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
