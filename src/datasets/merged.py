from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from src.datasets.cmp_facade import CMPFacade
from src.datasets.hznu import Hznu
from src.datasets.irregular_facades import IrregularFacades
from src.datasets.merged_classes import MergedClasses


class Subset(Dataset):
    """A simple subset of the merged dataset."""

    def __init__(self, all_paths: list[tuple[Path, Path]], indices: list[int]) -> None:
        self.paths = [all_paths[i] for i in indices]

    def __len__(self) -> int:
        return self.paths.__len__()

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.paths[index]

        image = np.array(Image.open(image_path))

        mask = np.array(Image.open(mask_path))
        mask = MergedClasses.convert_image_to_mask(mask)
        mask = MergedClasses.convert_mask_to_output(mask)

        return image, mask


class MergedDataset(LightningDataModule):
    """The final dataset merged from the sub-datasets."""

    def __init__(
        self, root_dir: Path = Path(__file__).parent / ".data" / "merged", batch_size: int = 32
    ) -> None:
        self.root_dir = root_dir
        self.batch_size = batch_size

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

                image = Image.fromarray(image)
                mask = Image.fromarray(MergedClasses.convert_mask_to_image(mask))

                image_path = self.root_dir / f"{sha256(image.tobytes()).hexdigest()[:8]}.jpg"
                mask_path = image_path.with_suffix(".png")
                image.save(image_path)
                mask.save(mask_path)

    def setup(self, _: str = "") -> None:
        """Load the paths and do the dataset split."""
        image_paths = sorted(self.root_dir.rglob("*.jpg"))
        mask_paths = [path.with_suffix(".png") for path in image_paths]
        paths = list(zip(image_paths, mask_paths, strict=True))
        self.data = Subset(paths, list(range(len(paths))))

        train_len = int(0.6 * len(paths))
        val_len = int(0.2 * len(paths))
        test_len = len(paths) - train_len - val_len

        train_indices, val_indices, test_indices = random_split(
            range(len(paths)),
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(0),
        )

        self.train_dataset = Subset(paths, train_indices)
        self.val_dataset = Subset(paths, val_indices)
        self.test_dataset = Subset(paths, test_indices)

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return the testing dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def __len__(self) -> int:
        return self.data.__len__()


if __name__ == "__main__":
    dataset = MergedDataset()
    dataset.prepare_data()
    dataset.setup()
