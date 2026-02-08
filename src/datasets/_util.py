from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path

import numpy as np


class _SegmentationDataset(ABC):
    paths: list[tuple[Path, Path]]
    classes: _SegmentationClasses

    def __init__(self, root_dir: Path) -> None:
        image_paths = sorted(root_dir.rglob("*.jpg"))
        mask_paths = [path.with_suffix(".png") for path in image_paths]
        self.paths = list(zip(image_paths, mask_paths, strict=True))

    def __len__(self) -> int:
        return self.paths.__len__()

    @classmethod
    @abstractmethod
    def download(cls) -> _SegmentationDataset:
        raise NotImplementedError


class _SegmentationClasses(IntEnum):
    @classmethod
    @abstractmethod
    def from_color(cls, color: tuple[int, int, int]) -> _SegmentationClasses:
        raise NotImplementedError

    @abstractmethod
    def to_color(self) -> tuple[int, int, int]:
        raise NotImplementedError

    @abstractmethod
    def to_merged(self):  # noqa: ANN202  can not import MergedClasses due to circular import
        raise NotImplementedError

    @classmethod
    def convert_image_to_mask(cls, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for class_ in cls:
            color = class_.to_color()
            value = class_.value

            matches = np.all(image == color, axis=-1)
            mask[matches] = value

        return mask

    @classmethod
    def convert_mask_to_merged(cls, mask: np.ndarray) -> np.ndarray:
        converted_mask = np.zeros(mask.shape)
        for class_ in cls:
            original_value = class_.value
            new_value = class_.to_merged().value

            matches = mask == original_value
            converted_mask[matches] = new_value

        return converted_mask
