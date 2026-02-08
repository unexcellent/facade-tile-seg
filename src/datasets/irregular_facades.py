from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from ._util import _SegmentationClasses, _SegmentationDataset


class IrregularFacadesClasses(_SegmentationClasses):
    """Classes from the irregular facades dataset."""

    BACKGROUND = 0
    PLANT = 1
    WALL = 2
    WINDOW = 3
    DOOR = 4
    FENCE = 5

    @classmethod
    def from_color(cls, color: tuple[int, int, int]) -> IrregularFacadesClasses:
        """Construct this class from the RGB value in the segmentation mask png."""
        try:
            return _COLOR_TO_CLASS_MAPPING[color]
        except KeyError:
            raise ValueError(f"Unsupported color {color}") from None

    def to_color(self) -> tuple[int, int, int]:
        """Convert this class to the RGB value in the segmentation mask png."""
        return _CLASS_TO_COLOR_MAPPING[self]


_COLOR_TO_CLASS_MAPPING = {
    (0, 0, 0): IrregularFacadesClasses.BACKGROUND,
    (128, 0, 128): IrregularFacadesClasses.PLANT,
    (128, 0, 0): IrregularFacadesClasses.WALL,
    (0, 128, 0): IrregularFacadesClasses.WINDOW,
    (128, 128, 0): IrregularFacadesClasses.DOOR,
    (0, 0, 128): IrregularFacadesClasses.FENCE,
}

_CLASS_TO_COLOR_MAPPING = {v: k for k, v in _COLOR_TO_CLASS_MAPPING.items()}


class IrregularFacades(_SegmentationDataset):
    """The Irregular Facades dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    @classmethod
    def download(cls) -> IrregularFacades:
        """Download and construct the dataset."""
        download_path = Path(kagglehub.dataset_download("liushuyuu/irregular-facades-irfs"))
        target_path = Path(__file__).parent / ".data" / "irregular_facades"

        if not target_path.parent.is_dir():
            target_path.parent.mkdir(parents=True)

        if not target_path.is_dir():
            shutil.copytree(str(download_path), str(target_path))

        return cls(target_path)
