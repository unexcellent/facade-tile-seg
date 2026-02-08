import numpy as np
import pytest

from src.datasets import CMPFacade, CMPFacadeClasses
from src.datasets.merged import MergedClasses


def test_dataset_length():
    dataset = CMPFacade.download()
    assert len(dataset) == 378


def test_paths_exist():
    dataset = CMPFacade.download()

    for image_path, mask_path in dataset.paths:
        assert image_path.is_file()
        assert mask_path.is_file()


def test_paths_are_correct_format():
    dataset = CMPFacade.download()

    for image_path, mask_path in dataset.paths:
        assert image_path.suffix == ".jpg"
        assert mask_path.suffix == ".png"
        assert str(image_path).split(".jpg")[0] == str(mask_path).split(".png")[0]


def test_classes_from_color_black():
    dark_blue = (0, 0, 163)
    assert CMPFacadeClasses.from_color(dark_blue) == CMPFacadeClasses.BACKGROUND


def test_classes_from_color_invalid():
    white = (256, 256, 256)
    with pytest.raises(ValueError):
        CMPFacadeClasses.from_color(white)


def test_classes_to_color_background():
    dark_blue = (0, 0, 163)
    assert CMPFacadeClasses.BACKGROUND.to_color() == dark_blue


def test_convert_image_to_mask():
    dark_blue = (0, 0, 163)
    blue = (0, 0, 245)
    red = (234, 51, 35)

    image = np.array(
        [
            [dark_blue, blue, dark_blue],
            [blue, red, blue],
            [dark_blue, blue, dark_blue],
        ]
    )

    background = CMPFacadeClasses.BACKGROUND
    wall = CMPFacadeClasses.FACADE
    pillar = CMPFacadeClasses.PILLAR

    mask = np.array(
        [
            [background, wall, background],
            [wall, pillar, wall],
            [background, wall, background],
        ],
        dtype=int,
    )

    assert (CMPFacadeClasses.convert_image_to_mask(image) == mask).all()


def test_classes_to_merged():
    assert CMPFacadeClasses.FACADE.to_merged() == MergedClasses.USABLE
