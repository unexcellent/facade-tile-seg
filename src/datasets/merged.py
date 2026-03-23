from __future__ import annotations

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
        datasets: list[_SegmentationDataset],
        augment: bool,
        root_dir: Path = Path(__file__).parent / ".data" / "merged",
        batch_size: int = 16,
        num_workers: int = 7,
    ) -> None:
        super().__init__()
        self.datasets = datasets
        self.augment = augment
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Download and process the data."""
        self.root_dir.mkdir(parents=True)
        pbar = tqdm(self.datasets, desc="Processing Datasets")
        for dataset in pbar:
            for i in range(len(dataset)):
                image, mask = dataset[i]

                image_tensor = torch.from_numpy(image).unsqueeze(0)
                mask_rgb = MergedClasses.convert_mask_to_image(mask)
                mask_tensor = torch.from_numpy(mask_rgb).permute(2, 0, 1)

                image_hash = sha256(image.tobytes()).hexdigest()[:8]
                image_path = self.root_dir / f"{image_hash}.jpg"
                mask_png_path = image_path.with_suffix(".png")
                mask_npy_path = image_path.with_suffix(".npy")

                write_jpeg(image_tensor, str(image_path))
                write_png(mask_tensor, str(mask_png_path))
                np.save(mask_npy_path, mask)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Load the paths and do the dataset split."""
        image_paths = sorted(self.root_dir.rglob("*.jpg"))
        mask_paths = [path.with_suffix(".npy") for path in image_paths]
        paths = list(zip(image_paths, mask_paths, strict=True))

        train_len = int(0.6 * len(paths))
        val_len = int(0.2 * len(paths))
        test_len = len(paths) - train_len - val_len

        train_indices, val_indices, test_indices = random_split(
            range(len(paths)),
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(0),
        )

        self.train_dataset = Subset(paths, train_indices, augment=self.augment)
        self.val_dataset = Subset(paths, val_indices)
        self.test_dataset = Subset(paths, test_indices)

        self.data = Subset(paths, list(range(len(paths))))

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        """Return the testing dataloader."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def __len__(self) -> int:
        return len(self.data)


if __name__ == "__main__":
    dataset = MergedDataset(datasets=[], augment=False)
    dataset.prepare_data()
    dataset.setup()
