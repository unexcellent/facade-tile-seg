import pytest
import numpy as np


from src.datasets import IrregularFacades, IrregularFacadesClasses


def test_dataset_length():
    dataset = IrregularFacades.download()
    assert len(dataset) == 1057


def test_paths_exist():
    dataset = IrregularFacades.download()

    for image_path, mask_path in dataset.paths:
        assert image_path.is_file()
        assert mask_path.is_file()


def test_paths_are_correct_format():
    dataset = IrregularFacades.download()

    for image_path, mask_path in dataset.paths:
        assert image_path.suffix == ".jpg"
        assert mask_path.suffix == ".png"
        assert str(image_path).split(".jpg")[0] == str(mask_path).split(".png")[0]


def test_classes_from_color_black():
    black = (0, 0, 0)
    assert IrregularFacadesClasses.from_color(black) == IrregularFacadesClasses.BACKGROUND


def test_classes_from_color_invalid():
    white = (256, 256, 256)
    with pytest.raises(ValueError):
        IrregularFacadesClasses.from_color(white)


def test_classes_to_color_background():
    black = (0, 0, 0)
    assert IrregularFacadesClasses.BACKGROUND.to_color() == black


def test_convert_image_to_mask():
    black = (0, 0, 0)
    red = (128, 0, 0)
    green = (0, 128, 0)

    image = np.array(
        [
            [black, red, black],
            [red, green, red],
            [black, red, black],
        ]
    )

    background = IrregularFacadesClasses.BACKGROUND
    wall = IrregularFacadesClasses.WALL
    window = IrregularFacadesClasses.WINDOW

    mask = np.array(
        [
            [background, wall, background],
            [wall, window, wall],
            [background, wall, background],
        ],
        dtype=int,
    )

    assert (IrregularFacadesClasses.convert_image_to_mask(image) == mask).all()
