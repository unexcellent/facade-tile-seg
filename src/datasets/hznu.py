from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from tqdm import tqdm

from src.datasets._util import _SegmentationClasses, _SegmentationDataset
from src.datasets.merged import MergedClasses

DOWNLOAD_URL = "https://data.mendeley.com/public-files/datasets/k387xkyc5f/files/e5c4ddb5-2a79-480a-ad67-a688f1087c52/file_downloaded"


class HznuClasses(_SegmentationClasses):
    """Classes from the HZNU dataset."""

    BACKGROUND = 0
    BUILDING = 1
    CAR = 2
    TREE = 3
    WINDOW = 4
    DOOR = 5

    @classmethod
    def from_color(cls, color: tuple[int, int, int]) -> HznuClasses:
        """Construct this class from the RGB value in the segmentation mask png."""
        try:
            return _COLOR_TO_CLASS_MAPPING[color]
        except KeyError:
            raise ValueError(f"Unsupported color {color}") from None

    @classmethod
    def from_str(cls, name: str) -> HznuClasses:
        """Construct this class from its name in the annotation files."""
        try:
            return _NAME_TO_CLASS_MAPPING[name]
        except KeyError:
            raise ValueError(f"Unsupported name '{name}'") from None

    def to_color(self) -> tuple[int, int, int]:
        """Convert this class to the RGB value in the segmentation mask png."""
        return _CLASS_TO_COLOR_MAPPING[self]

    def to_merged(self) -> MergedClasses:
        """Map this class to the class in the merged dataset."""
        match self:
            case self.BACKGROUND | self.CAR | self.TREE:
                return MergedClasses.BACKGROUND
            case self.BUILDING:
                return MergedClasses.USABLE
            case _:
                return MergedClasses.NOT_USABLE


_COLOR_TO_CLASS_MAPPING = {
    (0, 0, 0): HznuClasses.BACKGROUND,
    (0, 0, 255): HznuClasses.BUILDING,
    (255, 255, 255): HznuClasses.CAR,
    (0, 255, 0): HznuClasses.TREE,
    (255, 0, 0): HznuClasses.WINDOW,
    (255, 255, 0): HznuClasses.DOOR,
}

_CLASS_TO_COLOR_MAPPING = {v: k for k, v in _COLOR_TO_CLASS_MAPPING.items()}

_NAME_TO_CLASS_MAPPING = {
    "sky": HznuClasses.BACKGROUND,
    "building": HznuClasses.BUILDING,
    "car": HznuClasses.CAR,
    "tree": HznuClasses.TREE,
    "window": HznuClasses.WINDOW,
    "door": HznuClasses.DOOR,
}


class Hznu(_SegmentationDataset):
    """The Hznu Facade dataset.

    This class should be preferably be constructed using the `.download()` constructor.
    """

    classes = HznuClasses

    @classmethod
    def download(cls) -> Hznu:
        """Download and construct the dataset."""
        target_path = Path(__file__).parent / ".data" / "hznu"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if target_path.is_dir():
            return cls(target_path)

        response = requests.get(DOWNLOAD_URL, stream=True, timeout=60)
        response.raise_for_status()

        buffer = _download_zip(response)
        _unzip_to_folder(buffer, target_path)
        _rasterize_annotations(target_path)

        return cls(target_path)


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


def _rasterize_annotations(root_dir: Path) -> None:
    samples_directory = root_dir / "all-json_adjust_zhengmian"
    image_paths = sorted(samples_directory.glob("*.jpg"))
    pbar = tqdm(image_paths, desc="Rasterizing Hznu")
    for image_path in pbar:
        annotation_path = image_path.with_suffix(".json")
        with annotation_path.open() as annotation_file:
            annotations = json.load(annotation_file)

        width, height = Image.open(image_path).size
        mask_path = image_path.with_suffix(".png")
        _rasterize_mask(annotations, width, height).save(mask_path)


def _rasterize_mask(annotations: dict, width: int, height: int) -> Image.Image:
    min_viable_point_amount = 2
    mask = Image.new("RGB", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for shape in annotations["shapes"]:
        class_ = HznuClasses.from_str(shape["label"])

        points = [tuple(point) for point in shape["points"]]
        if len(points) < min_viable_point_amount:
            continue

        draw.polygon(points, outline=class_.to_color(), fill=class_.to_color())

    return mask


if __name__ == "__main__":
    Hznu.download()
