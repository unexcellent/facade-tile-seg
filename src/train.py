from dataclasses import dataclass, field

from lightning import LightningModule, Trainer

from src.datasets._util import _SegmentationDataset
from src.datasets.cmp_facade import CMPFacade
from src.datasets.hznu import Hznu
from src.datasets.irregular_facades import IrregularFacades
from src.datasets.merged import MergedDataset
from src.model import FacadeSegmenter


def _all_datasets() -> list[_SegmentationDataset]:
    return [
        CMPFacade.download(),
        Hznu.download(),
        IrregularFacades.download(),
    ]


@dataclass
class TrainingConfig:
    """Configuration settings for model training."""

    max_epochs: int = 4
    augment: bool = True
    datasets: list[_SegmentationDataset] = field(default_factory=_all_datasets)


def train(config: TrainingConfig) -> tuple[LightningModule, float]:
    """Train a model based on a TrainingConfig and return its mIoU."""
    trainer = Trainer(max_epochs=config.max_epochs, precision="16-mixed")
    model = FacadeSegmenter()
    dataset = MergedDataset(config.datasets, config.augment)

    trainer.fit(model, dataset)
    performance = trainer.test(model, dataset)[-1]
    return model, performance["test_miou"]


if __name__ == "__main__":
    train(TrainingConfig())
