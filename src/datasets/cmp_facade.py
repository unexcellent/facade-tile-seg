from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from src.datasets._util import _SegmentationClasses, _SegmentationDataset
from src.datasets.merged import MergedClasses


class CMPFacadeClasses(_SegmentationClasses):
    """Classes from the CMP facades dataset."""

    BACKGROUND = 0
    FACADE = 1
    WINDOW = 2
    DOOR = 3
    CORNICE = 4
    SILL = 5
    BALCONY = 6
    BLIND = 7
    DECO = 8
    MOLDING = 9
    PILLAR = 10
    SHOP = 11

    @classmethod
    def from_color(cls, color: tuple[int, int, int]) -> CMPFacadeClasses:
        """Construct this class from the RGB value in the segmentation mask png."""
        try:
            return _COLOR_TO_CLASS_MAPPING[color]
        except KeyError:
            raise ValueError(f"Unsupported color {color}") from None

    def to_color(self) -> tuple[int, int, int]:
        """Convert this class to the RGB value in the segmentation mask png."""
        return _CLASS_TO_COLOR_MAPPING[self]

    def to_merged(self) -> MergedClasses:
        """Map this class to the class in the merged dataset."""
        match self:
            case self.BACKGROUND:
                return MergedClasses.BACKGROUND
            case self.FACADE:
                return MergedClasses.USABLE
            case _:
                return MergedClasses.NOT_USABLE


_COLOR_TO_CLASS_MAPPING = {
    (0, 0, 163): CMPFacadeClasses.BACKGROUND,
    (0, 0, 245): CMPFacadeClasses.FACADE,
    (34, 84, 245): CMPFacadeClasses.WINDOW,
    (76, 167, 248): CMPFacadeClasses.DOOR,
    (236, 98, 42): CMPFacadeClasses.CORNICE,
    (0, 0, 0): CMPFacadeClasses.SILL,
    (189, 253, 112): CMPFacadeClasses.BALCONY,
    (255, 255, 84): CMPFacadeClasses.BLIND,
    (243, 174, 61): CMPFacadeClasses.DECO,
    (117, 251, 253): CMPFacadeClasses.MOLDING,
    (234, 51, 35): CMPFacadeClasses.PILLAR,
    (156, 31, 20): CMPFacadeClasses.SHOP,
}

_CLASS_TO_COLOR_MAPPING = {v: k for k, v in _COLOR_TO_CLASS_MAPPING.items()}


class CMPFacade(_SegmentationDataset):
    """The CMP Facades dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    classes = CMPFacadeClasses

    @classmethod
    def download(cls) -> CMPFacade:
        """Download and construct the dataset."""
        download_path = Path(kagglehub.dataset_download("adlteam/facade-dataset"))
        target_path = Path(__file__).parent / ".data" / "cmp_facade"

        if not target_path.parent.is_dir():
            target_path.parent.mkdir(parents=True)

        if not target_path.is_dir():
            shutil.copytree(str(download_path), str(target_path))

        return cls(target_path)
