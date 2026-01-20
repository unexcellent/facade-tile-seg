from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests
from torch.utils.data import Dataset
from tqdm import tqdm

DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/k387xkyc5f/files/e5c4ddb5-2a79-480a-ad67-a688f1087c52/file_downloaded"


class Hznu(Dataset):
    """The Hznu Facade dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    paths: list[tuple[Path, Path]]

    def __init__(self, root_dir: Path) -> None:
        """Construct the dataset using the root path of the data directory."""
        samples_directory = root_dir / "all-json_adjust_zhengmian"
        image_paths = list(samples_directory.glob("*.jpg"))
        annotation_paths = list(samples_directory.glob("*.json"))

        self.paths = list(zip(image_paths, annotation_paths, strict=True))

    @classmethod
    def download(cls) -> Hznu:
        """Download and construct the dataset."""
        target_path = Path(__file__).parent / ".data" / "hznu"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(DOWNLOAD_URL, stream=True, timeout=60)
        response.raise_for_status()

        if not target_path.is_dir():
            buffer = _download_zip(response)
            _unzip_to_folder(buffer, target_path)

        return cls(target_path)

    def __len__(self) -> int:
        return len(self.paths)


def _download_zip(response: requests.Response) -> io.BytesIO:
    buffer = io.BytesIO()

    total_size = int(response.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading Hznu") as pbar:
        for data in response.iter_content(1024):
            pbar.update(len(data))
            buffer.write(data)

    return buffer


def _unzip_to_folder(buffer: io.BytesIO, target_path: Path) -> None:
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as zip_buffer:
        zip_buffer.extractall(target_path)


if __name__ == "__main__":
    Hznu.download()
