from lightning import Trainer

from src.datasets.merged import MergedDataset
from src.model import FacadeSegmenter

trainer = Trainer(max_epochs=4, deterministic=True)
model = FacadeSegmenter()
dataset = MergedDataset()

trainer.fit(model, dataset)
trainer.test(model, dataset)
