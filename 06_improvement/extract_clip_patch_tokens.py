# -*- coding: utf-8 -*-
"""Extract frozen CLIP ViT patch tokens for the CUB few-shot splits.

This script is the patch-token counterpart of extract_clip_features.py. It uses
the frozen CLIP image tower, keeps the final patch-token grid, and stores an
N x P x C token tensor per split. For CLIP ViT-B/16 at 224px, P is 14 x 14.
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
                if row_idx == 0 or len(row) < 2:
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


def load_open_clip(model_name, pretrained, device):
    try:
        import open_clip
    except ImportError as exc:
        raise ImportError("Install open_clip_torch before extracting patch tokens.") from exc

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    return model, preprocess, f"open_clip:{model_name}:{pretrained}"


def encode_open_clip_patch_tokens(model, images, token_stage="post_proj", normalize_tokens=False):
    """Forward CLIP ViT and return final patch tokens.

    open_clip exposes encode_image for pooled features, but FroFA needs the
    token grid. ``post_ln`` keeps final tokens before CLIP projection and is
    closer to the frozen patch feature setting used by FroFA.
    """
    visual = model.visual
    if not hasattr(visual, "conv1"):
        raise RuntimeError("Patch-token extraction currently expects open_clip ViT models.")

    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    class_embedding = visual.class_embedding.to(x.dtype)
    class_token = class_embedding + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([class_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)

    if hasattr(visual, "patch_dropout"):
        x = visual.patch_dropout(x)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    if token_stage == "pre_ln":
        patch_tokens = x[:, 1:, :]
    else:
        x = visual.ln_post(x)
        patch_tokens = x[:, 1:, :]

    if token_stage == "post_proj" and getattr(visual, "proj", None) is not None:
        x = x @ visual.proj
        patch_tokens = x[:, 1:, :]
    elif token_stage not in ("pre_ln", "post_ln", "post_proj"):
        raise ValueError(f"Unsupported token_stage: {token_stage}")

    if normalize_tokens:
        patch_tokens = torch.nn.functional.normalize(patch_tokens.float(), dim=-1)
    return patch_tokens


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

    all_tokens = []
    all_labels = []
    all_class_names = []
    all_paths = []
    for images, labels, class_names, image_names in loader:
        images = images.to(args.device, non_blocking=True)
        tokens = encode_open_clip_patch_tokens(
            model,
            images,
            token_stage=args.token_stage,
            normalize_tokens=args.normalize_tokens,
        )
        tokens = tokens.detach().cpu()
        if args.feature_dtype == "float16":
            tokens = tokens.half()
        all_tokens.append(tokens.numpy())
        all_labels.append(labels.numpy())
        all_class_names.extend(list(class_names))
        all_paths.extend(list(image_names))

    tokens = np.concatenate(all_tokens, axis=0)
    labels = np.concatenate(all_labels, axis=0).astype(np.int64)
    class_names = np.asarray(all_class_names)
    paths = np.asarray(all_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"cub_{split}_{args.output_prefix}.npz"
    np.savez_compressed(
        output_path,
        tokens=tokens,
        labels=labels,
        class_names=class_names,
        paths=paths,
        model_tag=np.asarray(model_tag),
        split=np.asarray(split),
        normalized=np.asarray(args.normalize_tokens),
        token_stage=np.asarray(args.token_stage),
    )
    return output_path, tokens.shape, len(dataset.class_to_idx)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="03_datasets/CUB_200_2011")
    parser.add_argument("--output-dir", default="06_improvement/results/features")
    parser.add_argument("--output-prefix", default="clip_vit_b16_patch_tokens")
    parser.add_argument("--model-name", default="ViT-B-16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--feature-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--token-stage",
        choices=["pre_ln", "post_ln", "post_proj"],
        default="post_proj",
        help="post_ln keeps final CLIP ViT tokens before projection and is closest to paper-style patch features.",
    )
    parser.add_argument("--normalize-tokens", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model, preprocess, model_tag = load_open_clip(
        args.model_name, args.pretrained, args.device
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
                "token_shape": list(shape),
                "class_count": class_count,
            }
        )
        print(f"[{split}] saved {shape} patch tokens to {output_path}")

    metadata_path = Path(args.output_dir) / f"metadata_{args.output_prefix}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "data_root": os.path.abspath(args.data_root),
                "model": model_tag,
                "feature_dtype": args.feature_dtype,
                "normalized": args.normalize_tokens,
                "token_stage": args.token_stage,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
