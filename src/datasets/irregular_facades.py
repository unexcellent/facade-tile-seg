from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
from torch.utils.data import Dataset


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

        if len(image_paths) != len(mask_paths):
            raise IndexError

        self.paths = list(zip(image_paths, mask_paths))

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
