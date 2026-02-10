from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub

from src.datasets._util import _SegmentationClasses, _SegmentationDataset
from src.datasets.merged_classes import MergedClasses


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
    (0, 0, 170): CMPFacadeClasses.BACKGROUND,
    (0, 0, 255): CMPFacadeClasses.FACADE,
    (255, 255, 0): CMPFacadeClasses.BLIND,
    (170, 255, 85): CMPFacadeClasses.BALCONY,
    (0, 85, 255): CMPFacadeClasses.WINDOW,
    (85, 255, 170): CMPFacadeClasses.SILL,
    (255, 170, 0): CMPFacadeClasses.DECO,
    (255, 85, 0): CMPFacadeClasses.CORNICE,
    (255, 0, 0): CMPFacadeClasses.PILLAR,
    (0, 170, 255): CMPFacadeClasses.DOOR,
    (170, 0, 0): CMPFacadeClasses.SHOP,
    (0, 255, 255): CMPFacadeClasses.MOLDING,
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
