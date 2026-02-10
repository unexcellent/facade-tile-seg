import pytest

from src.datasets import CMPFacade, CMPFacadeClasses
from src.datasets.merged_classes import MergedClasses


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
    dark_blue = (0, 0, 170)
    assert CMPFacadeClasses.from_color(dark_blue) == CMPFacadeClasses.BACKGROUND


def test_classes_from_color_invalid():
    white = (256, 256, 256)
    with pytest.raises(ValueError):
        CMPFacadeClasses.from_color(white)


def test_classes_to_color_background():
    dark_blue = (0, 0, 170)
    assert CMPFacadeClasses.BACKGROUND.to_color() == dark_blue


def test_classes_to_merged():
    assert CMPFacadeClasses.FACADE.to_merged() == MergedClasses.USABLE
