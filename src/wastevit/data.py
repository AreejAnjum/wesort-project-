from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    images = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"No supported images found below {root}")
    return images


def image_to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = ImageOps.fit(image.convert("RGB"), (size, size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (tensor - mean) / std


def ssl_view(image: Image.Image, size: int) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    scale = random.uniform(0.55, 1.0)
    crop_w, crop_h = max(1, int(width * scale)), max(1, int(height * scale))
    left = random.randint(0, max(0, width - crop_w))
    top = random.randint(0, max(0, height - crop_h))
    image = image.crop((left, top, left + crop_w, top + crop_h))
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.7, 1.3))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.6, 1.4))
    if random.random() < 0.2:
        image = ImageOps.grayscale(image).convert("RGB")
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
    return image_to_tensor(image, size)


class UnlabeledPairDataset(Dataset):
    """Returns two augmented views and never exposes labels to SSL."""

    def __init__(self, root: str | Path, image_size: int):
        self.paths = discover_images(root)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return ssl_view(image, self.image_size), ssl_view(image, self.image_size)


class ManifestDataset(Dataset):
    def __init__(self, manifest: str | Path, image_size: int, class_to_idx=None):
        manifest = Path(manifest)
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or not {"path", "label"}.issubset(rows[0]):
            raise ValueError(f"Manifest must contain path,label columns: {manifest}")
        labels = sorted({row["label"] for row in rows})
        self.class_to_idx = class_to_idx or {label: i for i, label in enumerate(labels)}
        unknown = sorted(set(labels) - set(self.class_to_idx))
        if unknown:
            raise ValueError(f"Unknown labels in {manifest}: {unknown}")
        self.samples = [(Path(row["path"]), self.class_to_idx[row["label"]]) for row in rows]
        missing = [str(path) for path, _ in self.samples if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest references missing images (first 3): {missing[:3]}")
        self.image_size = image_size

    @property
    def paths(self):
        return [path for path, _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image_to_tensor(image, self.image_size), label
