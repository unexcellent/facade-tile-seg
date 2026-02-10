from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from lightning import LightningDataModule
from PIL import Image
from tqdm import tqdm

from src.datasets.cmp_facade import CMPFacade
from src.datasets.hznu import Hznu
from src.datasets.irregular_facades import IrregularFacades
from src.datasets.merged_classes import MergedClasses


class MergedDataset(LightningDataModule):
    """The final dataset merged from the sub-datasets."""

    def __init__(self, root_dir: Path = Path(__file__).parent / ".data" / "merged") -> None:
        self.root_dir = root_dir

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


if __name__ == "__main__":
    MergedDataset().prepare_data()
