from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
from torch.utils.data import Dataset


class CMPFacade(Dataset):
    """The CMP Facade dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    paths: list[tuple[Path, Path]]

    def __init__(self, root_dir: Path) -> None:
        """Construct the dataset using the root path of the data directory."""
        samples_directory = root_dir / "base"
        image_paths = list(samples_directory.glob("*.jpg"))
        mask_paths = list(samples_directory.glob("*.png"))

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
