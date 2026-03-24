import torch
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

from src.attention import AttentionUnet
from src.datasets.merged_classes import MergedClasses


class FacadeSegmenter(LightningModule):
    """A model used to segment building facades."""

    model: nn.Module
    loss_fn: nn.Module
    learning_rate: float
    test_iou_metric: MulticlassJaccardIndex
    test_acc_metric: MulticlassAccuracy
    test_prec_metric: MulticlassPrecision
    test_rec_metric: MulticlassRecall
    test_f1_metric: MulticlassF1Score

    def __init__(
        self,
        model: nn.Module | None,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = model if model is not None else AttentionUnet()
        self.loss_fn = nn.CrossEntropyLoss()

        num_classes = len(MergedClasses)
        self.test_iou_metric = MulticlassJaccardIndex(num_classes=num_classes)
        self.test_acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_prec_metric = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_rec_metric = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro")

        self.validation_step_outputs: list[dict[str, Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward propagation on the model."""
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], _: int = 0) -> Tensor:
        """Compute and return the training loss and some additional metrics."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], _: int = 0) -> dict[str, Tensor]:
        """Operates on a single batch of data from the validation set."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        output = {"val_loss": loss.detach()}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        """Log the validation loss."""
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()

        self.log("val_loss", avg_loss, prog_bar=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[Tensor, Tensor], _: int = 0) -> None:  # noqa: PT019
        """Operates on a single batch of data from the test set."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.test_iou_metric.update(preds, y)
        self.test_acc_metric.update(preds, y)
        self.test_prec_metric.update(preds, y)
        self.test_rec_metric.update(preds, y)
        self.test_f1_metric.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Log the test metrics."""
        self.log("test_miou", self.test_iou_metric.compute())
        self.log("test_acc", self.test_acc_metric.compute())
        self.log("test_precision", self.test_prec_metric.compute())
        self.log("test_recall", self.test_rec_metric.compute())
        self.log("test_f1", self.test_f1_metric.compute())

        self.test_iou_metric.reset()
        self.test_acc_metric.reset()
        self.test_prec_metric.reset()
        self.test_rec_metric.reset()
        self.test_f1_metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Return the optimizer used."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
