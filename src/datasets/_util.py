from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum

import numpy as np


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
