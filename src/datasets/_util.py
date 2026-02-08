from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path

import numpy as np


class _SegmentationDataset(ABC):
    paths: list[tuple[Path, Path]]

    def __init__(self, root_dir: Path) -> None:
        image_paths = sorted(root_dir.rglob("*.jpg"))
        mask_paths = [path.with_suffix(".png") for path in image_paths]
        self.paths = list(zip(image_paths, mask_paths, strict=True))

    @classmethod
    @abstractmethod
    def download(cls) -> _SegmentationDataset:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.paths.__len__()


class _SegmentationClasses(IntEnum):
    @classmethod
    @abstractmethod
    def from_color(cls, color: tuple[int, int, int]) -> _SegmentationClasses:
        raise NotImplementedError

    @abstractmethod
    def to_color(self) -> tuple[int, int, int]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def convert_image_to_mask(cls, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for class_ in cls:
            color = class_.to_color()
            value = class_.value

            matches = np.all(image == color, axis=-1)
            mask[matches] = value

        return mask
