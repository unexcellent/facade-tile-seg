from dataclasses import dataclass, field

from lightning import LightningModule, Trainer
from torch.nn import Module

from src.datasets._util import _SegmentationDataset
from src.datasets.cmp_facade import CMPFacade
from src.datasets.hznu import Hznu
from src.datasets.irregular_facades import IrregularFacades
from src.datasets.merged import MergedDataset
from src.model import FacadeSegmenter


def _default_train_datasets() -> list[_SegmentationDataset]:
    return [
        CMPFacade.download(),
        IrregularFacades.download(),
    ]


def _default_test_datasets() -> list[_SegmentationDataset]:
    return [
        Hznu.download(),
    ]


@dataclass
class TrainingConfig:
    """Configuration settings for model training."""

    name: str = ""
    max_epochs: int = 8
    augment: bool = True
    train_datasets: list[_SegmentationDataset] = field(default_factory=_default_train_datasets)
    test_datasets: list[_SegmentationDataset] = field(default_factory=_default_test_datasets)
    model: Module | None = None


def train(config: TrainingConfig) -> tuple[LightningModule, float]:
    """Train a model based on a TrainingConfig and return its mIoU."""
    trainer = Trainer(max_epochs=config.max_epochs, precision="16-mixed")
    model = FacadeSegmenter(config.model)
    dataset = MergedDataset(
        train_datasets=config.train_datasets,
        test_datasets=config.test_datasets,
        augment=config.augment,
    )

    trainer.fit(model, dataset)
    performance = trainer.test(model, dataset)[-1]
    return model, performance["test_miou"]


if __name__ == "__main__":
    train(TrainingConfig())
