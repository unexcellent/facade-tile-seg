from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.datasets.hznu import Hznu
from src.datasets.merged import MergedDataset
from src.datasets.merged_classes import MergedClasses
from src.model import FacadeSegmenter


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.25) -> Image.Image:
    """Overlay the segmentation mask visually over an image."""
    image_uint8 = (image[0] * 255).astype(np.uint8)
    image_rgb = np.stack((image_uint8,) * 3, axis=-1)
    base = Image.fromarray(image_rgb).convert("RGBA")

    mask_rgb = MergedClasses.convert_mask_to_image(mask)
    overlay = Image.fromarray(mask_rgb).convert("RGBA")

    return Image.blend(base, overlay, alpha).convert("RGB")


def save_predictions(model: FacadeSegmenter, dataset: MergedDataset, output_dir: Path) -> None:
    """Store the model predictions of a dataset to a directory."""
    dataset.prepare_data()
    dataset.setup()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    dataloader = dataset.test_dataloader()
    paths = dataset.test_dataset.paths

    idx = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Saving images", total=len(dataloader)):
            logits = model(x.to(device))
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            images = x.cpu().numpy()

            for i in range(len(images)):
                image_with_mask = overlay_mask(images[i], predictions[i])

                original_filename = paths[idx][0].name
                image_with_mask.save(output_dir / original_filename)
                idx += 1


if __name__ == "__main__":
    model_path = Path(__file__).parent / "datasets" / ".data" / "model.pt"
    datamodule = MergedDataset(train_datasets=[], test_datasets=[Hznu.download()], augment=False)
    model = FacadeSegmenter.load_from_checkpoint(model_path)
    prediction_output = Path(__file__).parent / "datasets" / ".data" / "predictions"
    save_predictions(model, datamodule, prediction_output)
