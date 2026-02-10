# the MergedClasses needed to be separated from MergedDataset to avoid circular import
from __future__ import annotations

from src.datasets._util import _SegmentationClasses


class MergedClasses(_SegmentationClasses):
    """Segmentation classes used in the final dataset."""

    BACKGROUND = 0
    USABLE = 1
    NOT_USABLE = 2

    @classmethod
    def from_color(cls, color: tuple[int, int, int]) -> MergedClasses:
        """Construct this class from the RGB value in the segmentation mask png."""
        try:
            return _COLOR_TO_CLASS_MAPPING[color]
        except KeyError:
            raise ValueError(f"Unsupported color {color}") from None

    def to_color(self) -> tuple[int, int, int]:
        """Convert this class to the RGB value in the segmentation mask png."""
        return _CLASS_TO_COLOR_MAPPING[self]


_COLOR_TO_CLASS_MAPPING = {
    (0, 0, 0): MergedClasses.BACKGROUND,
    (0, 255, 0): MergedClasses.USABLE,
    (255, 0, 0): MergedClasses.NOT_USABLE,
}

_CLASS_TO_COLOR_MAPPING = {v: k for k, v in _COLOR_TO_CLASS_MAPPING.items()}
