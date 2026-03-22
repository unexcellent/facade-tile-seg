from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms.v2 import ColorJitter, Compose, RandomVerticalFlip
from tqdm import tqdm

from src.datasets.cmp_facade import CMPFacade
from src.datasets.hznu import Hznu
from src.datasets.irregular_facades import IrregularFacades
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
        self.mask_transforms = Compose([RandomVerticalFlip()])
        self.image_transforms = Compose([RandomVerticalFlip(), ColorJitter()])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.paths[index]

        image = np.array(Image.open(image_path), dtype=np.float32) / 255.0
        image = image[np.newaxis, ...]
        if self.augment:
            image = self.image_transforms(image)

        mask = np.array(Image.open(mask_path))
        mask = MergedClasses.convert_image_to_mask(mask)
        if self.augment:
            mask = self.image_transforms(mask)

        return image, mask.astype(np.int64)


class MergedDataset(LightningDataModule):
    """The final dataset merged from the sub-datasets."""

    def __init__(
        self,
        root_dir: Path = Path(__file__).parent / ".data" / "merged",
        batch_size: int = 32,
        num_workers: int = 7,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Download and process the data."""
        if self.root_dir.is_dir():
            return

        self.root_dir.mkdir(parents=True)
        datasets = [
            CMPFacade.download(),
            Hznu.download(),
            IrregularFacades.download(),
        ]
        pbar = tqdm(datasets, desc="Processing Datasets")
        for dataset in pbar:
            for i in range(len(dataset)):
                image, mask = dataset[i]

                image_pil = Image.fromarray(image)
                mask_pil = Image.fromarray(MergedClasses.convert_mask_to_image(mask))

                image_hash = sha256(image_pil.tobytes()).hexdigest()[:8]
                image_path = self.root_dir / f"{image_hash}.jpg"
                mask_path = image_path.with_suffix(".png")

                image_pil.save(image_path)
                mask_pil.save(mask_path)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002 stage is a kwarg in the parent method so it can not be renamed
        """Load the paths and do the dataset split."""
        image_paths = sorted(self.root_dir.rglob("*.jpg"))
        mask_paths = [path.with_suffix(".png") for path in image_paths]
        paths = list(zip(image_paths, mask_paths, strict=True))

        train_len = int(0.6 * len(paths))
        val_len = int(0.2 * len(paths))
        test_len = len(paths) - train_len - val_len

        train_indices, val_indices, test_indices = random_split(
            range(len(paths)),
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(0),
        )

        self.train_dataset = Subset(paths, train_indices, augment=True)
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
    dataset = MergedDataset()
    dataset.prepare_data()
    dataset.setup()
