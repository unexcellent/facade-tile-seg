from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
from torch.utils.data import Dataset

from ._util import _SegmentationClasses


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
            return _CLASS_COLOR_MAPPING[color]
        except KeyError:
            raise ValueError(f"Unsupported color {color}") from None


_CLASS_COLOR_MAPPING = {
    (0, 0, 0): IrregularFacadesClasses.BACKGROUND,
    (128, 0, 128): IrregularFacadesClasses.PLANT,
    (128, 0, 0): IrregularFacadesClasses.WALL,
    (0, 128, 0): IrregularFacadesClasses.WINDOW,
    (128, 128, 0): IrregularFacadesClasses.DOOR,
    (0, 0, 128): IrregularFacadesClasses.FENCE,
}


class IrregularFacades(Dataset):
    """The Irregular Facades dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    paths: list[tuple[Path, Path]]

    def __init__(self, root_dir: Path) -> None:
        """Construct the dataset using the root path of the data directory."""
        samples_directory = root_dir / "samples" / "samples"
        image_paths = list(samples_directory.glob("*.jpg"))
        mask_paths = list(samples_directory.glob("*.png"))

        self.paths = list(zip(image_paths, mask_paths, strict=True))

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

    def __len__(self) -> int:
        return len(self.paths)
