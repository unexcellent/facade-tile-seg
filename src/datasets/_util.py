from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum


class _SegmentationClasses(IntEnum):
    @classmethod
    @abstractmethod
    def from_color(cls, color: tuple[int, int, int]) -> _SegmentationClasses:
        raise NotImplementedError
