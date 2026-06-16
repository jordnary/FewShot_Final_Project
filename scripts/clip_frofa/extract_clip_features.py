# -*- coding: utf-8 -*-
"""Extract frozen CLIP image features for the CUB few-shot splits.

The script expects the LibFewShot-style CUB directory used in this project:

    CUB_200_2011/
      images/
      train.csv
      val.csv
      test.csv

It writes one compressed NPZ per split. Each NPZ contains global frozen CLIP
features, integer labels local to that split, the original class names, and the
image paths. The encoder is never updated.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CubCsvDataset(Dataset):
    def __init__(self, data_root, split, transform):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.samples, self.class_to_idx = self._read_csv()

    def _read_csv(self):
        csv_path = self.data_root / f"{self.split}.csv"
        samples = []
        class_to_idx = {}
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    continue
                if len(row) < 2:
                    continue
                image_name, class_name = row[0], row[1]
                if class_name not in class_to_idx:
                    class_to_idx[class_name] = len(class_to_idx)
                samples.append((image_name, class_to_idx[class_name], class_name))
        if not samples:
            raise RuntimeError(f"No samples found in {csv_path}")
        return samples, class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_name, label, class_name = self.samples[index]
        image_path = self.data_root / "images" / image_name
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, label, class_name, image_name


def _load_openai_clip(model_name, device):
    try:
        import clip

        openai_name = model_name.replace("ViT-B-16", "ViT-B/16").replace(
            "ViT-B-32", "ViT-B/32"
        )
        model, preprocess = clip.load(openai_name, device=device)
        model = model.eval()
        return model, preprocess, f"openai_clip:{openai_name}"
    except ImportError as exc:
        raise ImportError(
            "OpenAI CLIP is not installed. Run: "
            "pip install git+https://github.com/openai/CLIP.git"
        ) from exc


def _load_open_clip(model_name, pretrained, device):
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        return model, preprocess, f"open_clip:{model_name}:{pretrained}"
    except ImportError as exc:
        raise ImportError("open_clip_torch is not installed.") from exc


def _load_torchvision_vit(model_name, device):
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError as exc:
        raise ImportError("torchvision is not installed.") from exc

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    model.heads = torch.nn.Identity()
    model = model.to(device).eval()
    preprocess = weights.transforms()
    return model, preprocess, f"torchvision:{model_name}:IMAGENET1K_V1"


def load_clip_model(model_name, pretrained, device, backend):
    """Load CLIP with a selectable backend.

    The cloud script defaults to open_clip. In classroom cloud environments,
    set HF_ENDPOINT to a usable HuggingFace mirror if direct downloads fail.
    """
    if backend == "openai":
        return _load_openai_clip(model_name, device)
    if backend == "open_clip":
        return _load_open_clip(model_name, pretrained, device)
    if backend == "torchvision_vit":
        return _load_torchvision_vit(model_name, device)

    errors = []
    for loader in (_load_openai_clip,):
        try:
            return loader(model_name, device)
        except Exception as exc:
            errors.append(str(exc))
    try:
        return _load_open_clip(model_name, pretrained, device)
    except Exception as exc:
        errors.append(str(exc))
    try:
        return _load_torchvision_vit(model_name, device)
    except Exception as exc:
        errors.append(str(exc))
    raise RuntimeError("Unable to load CLIP. Errors:\n- " + "\n- ".join(errors))


@torch.no_grad()
def extract_split(args, model, preprocess, model_tag, split):
    dataset = CubCsvDataset(args.data_root, split, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )

    all_features = []
    all_labels = []
    all_class_names = []
    all_paths = []
    for images, labels, class_names, image_names in loader:
        images = images.to(args.device, non_blocking=True)
        if hasattr(model, "encode_image"):
            features = model.encode_image(images)
        else:
            features = model(images)
        if args.normalize:
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        all_features.append(features.detach().cpu().float().numpy())
        all_labels.append(labels.numpy())
        all_class_names.extend(list(class_names))
        all_paths.extend(list(image_names))

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    class_names = np.asarray(all_class_names)
    paths = np.asarray(all_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cub_{split}_{args.output_prefix}.npz"
    np.savez_compressed(
        output_path,
        features=features,
        labels=labels,
        class_names=class_names,
        paths=paths,
        model_tag=np.asarray(model_tag),
        split=np.asarray(split),
        normalized=np.asarray(args.normalize),
    )
    return output_path, features.shape, len(dataset.class_to_idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="data/CUB_200_2011",
        help="Path to CUB_200_2011 containing images and split CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/features/clip_frofa",
        help="Directory for compressed feature files.",
    )
    parser.add_argument("--output-prefix", default="clip_vit_b16")
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument(
        "--backend",
        default="open_clip",
        choices=["openai", "open_clip", "torchvision_vit", "auto"],
        help="Use open_clip for CLIP ViT-B/16; torchvision_vit is a non-CLIP fallback.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--normalize", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model, preprocess, model_tag = load_clip_model(
        args.model_name, args.pretrained, args.device, args.backend
    )
    records = []
    for split in args.splits:
        output_path, shape, class_count = extract_split(
            args, model, preprocess, model_tag, split
        )
        records.append(
            {
                "split": split,
                "path": str(output_path),
                "feature_shape": list(shape),
                "class_count": class_count,
            }
        )
        print(f"[{split}] saved {shape} features to {output_path}")

    metadata_path = Path(args.output_dir) / f"metadata_{args.output_prefix}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "data_root": os.path.abspath(args.data_root),
                "model": model_tag,
                "normalized": args.normalize,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
