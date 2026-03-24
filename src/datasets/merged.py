from __future__ import annotations

import shutil
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.io import ImageReadMode, read_image, write_jpeg, write_png
from torchvision.transforms.v2 import ColorJitter, Compose
from tqdm import tqdm

from src.datasets._util import _SegmentationDataset
from src.datasets.merged_classes import MergedClasses


class Subset(Dataset):
    """A simple subset of the merged dataset."""

    def __init__(
        self,
        all_paths: list[tuple[Path, Path]],
        indices: list[int],
        augment: bool = False,
    ) -> None:
        self.paths = [all_paths[i] for i in indices]
        self.augment = augment
        self.image_transforms = Compose([ColorJitter()])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.paths[index]

        image = read_image(str(image_path), mode=ImageReadMode.GRAY).float() / 255.0
        if self.augment:
            image = self.image_transforms(image)

        mask = torch.from_numpy(np.load(mask_path)).long()

        return image, mask


class MergedDataset(LightningDataModule):
    """The final dataset merged from the sub-datasets."""

    def __init__(
        self,
        train_datasets: list[_SegmentationDataset],
        test_datasets: list[_SegmentationDataset],
        augment: bool,
        root_dir: Path | None = None,
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.augment = augment
        self.root_dir = root_dir or Path(__file__).parent / ".data" / "merged"
        self.batch_size = batch_size
        self.num_workers = 7

    def prepare_data(self) -> None:
        """Return the final dataset merged from the sub-datasets."""
        _process_datasets(self.train_datasets, self.root_dir / "train")
        _process_datasets(self.test_datasets, self.root_dir / "test")

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load the paths and do the dataset split."""
        train_dir = self.root_dir / "train"
        test_dir = self.root_dir / "test"

        train_image_paths = sorted(train_dir.rglob("*.jpg"))
        train_mask_paths = [path.with_suffix(".npy") for path in train_image_paths]
        train_paths = list(zip(train_image_paths, train_mask_paths, strict=True))

        test_image_paths = sorted(test_dir.rglob("*.jpg"))
        test_mask_paths = [path.with_suffix(".npy") for path in test_image_paths]
        test_paths = list(zip(test_image_paths, test_mask_paths, strict=True))

        train_len = int(0.8 * len(train_paths))
        val_len = len(train_paths) - train_len

        train_indices, val_indices = random_split(
            range(len(train_paths)),
            [train_len, val_len],
            generator=torch.Generator().manual_seed(0),
        )

        self.train_dataset = Subset(train_paths, train_indices, augment=self.augment)
        self.val_dataset = Subset(train_paths, val_indices)
        self.test_dataset = Subset(test_paths, list(range(len(test_paths))))

    def train_dataloader(self) -> DataLoader:
        """Load the paths and do the dataset split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return the testing dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


def _process_datasets(datasets: list[_SegmentationDataset], target_dir: Path) -> None:
    if len(datasets) == 0:
        return

    total_images = sum([len(dataset) for dataset in datasets])
    if len(list(target_dir.glob("*.jpg"))) == total_images - 1:
        return

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    progress_bar = tqdm(desc=f"Processing to {target_dir.name}", total=total_images)
    for dataset in datasets:
        for i in range(len(dataset)):
            image, mask = dataset[i]

            image_hash = sha256(image.tobytes()).hexdigest()[:32]
            image_path = target_dir / f"{image_hash}.jpg"

            image_tensor = torch.from_numpy(image).unsqueeze(0)
            mask_rgb = MergedClasses.convert_mask_to_image(mask)
            mask_tensor = torch.from_numpy(mask_rgb).permute(2, 0, 1)

            mask_png_path = image_path.with_suffix(".png")
            mask_npy_path = image_path.with_suffix(".npy")

            write_jpeg(image_tensor, str(image_path))
            write_png(mask_tensor, str(mask_png_path))
            np.save(mask_npy_path, mask)

            progress_bar.update(1)

    progress_bar.close()
