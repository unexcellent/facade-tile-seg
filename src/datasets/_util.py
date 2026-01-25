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
        mask = np.zeros(image.shape[:2], dtype=int)
        for x, y in np.ndindex(image.shape[:2]):
            mask[x, y] = cls.from_color(tuple(image[x, y]))

        return mask
