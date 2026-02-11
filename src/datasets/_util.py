from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_SIZE = (572, 572)


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
        mask = np.ones(image.shape[:2], dtype=np.int8) * -1
        for class_ in cls:
            color = class_.to_color()
            value = class_.value

            matches = np.all(image == color, axis=-1)
            mask[matches] = value

        if (mask < 0).any():
            unexpected_colors = np.unique(image[mask < 0], axis=0)
            raise ValueError(f"Unexpected color found: {unexpected_colors.tolist()}")

        return mask

    @classmethod
    def convert_mask_to_image(cls, mask: np.ndarray) -> np.ndarray:
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_ in cls:
            color = class_.to_color()
            matches = mask == class_.value
            image[matches] = color

        return image

    @classmethod
    def convert_mask_to_merged(cls, mask: np.ndarray) -> np.ndarray:
        converted_mask = np.zeros(mask.shape)
        for class_ in cls:
            original_value = class_.value
            new_value = class_.to_merged().value

            matches = mask == original_value
            converted_mask[matches] = new_value

        return converted_mask

    @classmethod
    def convert_mask_to_output(cls, mask: np.ndarray) -> np.ndarray:
        output = np.zeros((*mask.shape, len(cls)), dtype=np.uint8)
        for class_ in cls:
            output_array = np.zeros(len(cls), dtype=np.uint8)
            output_array[class_.value] = 1

            matches = mask == class_.value
            output[matches] = output_array

        return output


class _SegmentationDataset(ABC):
    paths: list[tuple[Path, Path]]
    classes: type[_SegmentationClasses]

    def __init__(self, root_dir: Path) -> None:
        image_paths = sorted(root_dir.rglob("*.jpg"))
        mask_paths = [path.with_suffix(".png") for path in image_paths]
        self.paths = list(zip(image_paths, mask_paths, strict=True))

    def __len__(self) -> int:
        return self.paths.__len__()

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.paths[idx]

        image = Image.open(image_path)
        background = self.classes(0)
        image = _resize(image, background.to_color()).convert("L")

        mask = Image.open(mask_path).convert("RGB")
        mask = _resize(mask, background.to_color())
        mask = self.classes.convert_image_to_mask(np.array(mask))
        mask = self.classes.convert_mask_to_merged(mask)

        return np.array(image), mask

    @classmethod
    @abstractmethod
    def download(cls) -> _SegmentationDataset:
        raise NotImplementedError


def _resize(image: Image.Image, background_color: tuple[int, int, int]) -> Image.Image:
    image.thumbnail(TARGET_SIZE, Image.Resampling.NEAREST)

    background = Image.new("RGB", TARGET_SIZE, background_color)
    offset = ((TARGET_SIZE[0] - image.width) // 2, (TARGET_SIZE[1] - image.height) // 2)
    background.paste(image, offset)
    return background
