from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ._util import _SegmentationClasses


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


class CMPFacade(Dataset):
    """The CMP Facades dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    paths: list[tuple[Path, Path]]

    def __init__(self, root_dir: Path) -> None:
        """Construct the dataset using the root path of the data directory."""
        samples_directory = root_dir / "base"
        image_paths = sorted(samples_directory.glob("*.jpg"))
        mask_paths = sorted(samples_directory.glob("*.png"))

        self.paths = list(zip(image_paths, mask_paths, strict=True))

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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.paths[index]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        return (
            image,
            CMPFacadeClasses.convert_image_to_mask(mask),
        )
