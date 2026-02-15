import segmentation_models_pytorch as smp
import torch
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics.classification import MulticlassJaccardIndex

from src.datasets.merged_classes import MergedClasses


class FacadeSegmenter(LightningModule):
    """A model for segmenting building facades."""

    model: nn.Module
    loss_fn: nn.Module
    learning_rate: float
    iou_metric: MulticlassJaccardIndex

    def __init__(
        self,
        encoder_name: str = "resnet34",
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=1,
            classes=len(MergedClasses),
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.iou_metric = MulticlassJaccardIndex(num_classes=len(MergedClasses))
        self.validation_step_outputs: list[dict[str, Tensor]] = []

    def forward(self, x: Tensor) -> Tensor:
        """Predict using the model."""
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor], _: int = 0) -> Tensor:
        """Train the model with a single batch."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], _: int = 0) -> dict[str, Tensor]:
        """Validate the model with a single batch."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.iou_metric.update(preds, y)

        output = {"val_loss": loss}
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        """Log the metrics."""
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        miou = self.iou_metric.compute()

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_miou", miou, prog_bar=True)

        self.iou_metric.reset()
        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[Tensor, Tensor], _: int = 0) -> None:  # noqa: PT019
        """Test the model with a single batch."""
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.iou_metric.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Log the metrics."""
        miou = self.iou_metric.compute()
        self.log("test_miou", miou)
        self.iou_metric.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
