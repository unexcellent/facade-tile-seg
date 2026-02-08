import numpy as np

from src.datasets import Hznu
from src.datasets.hznu import HznuClasses, _rasterize_mask


def test_dataset_length():
    dataset = Hznu.download()
    assert len(dataset) == 624


def test_paths_exist():
    dataset = Hznu.download()

    for image_path, mask_path in dataset.paths:
        assert image_path.is_file()
        assert mask_path.is_file()


def test_paths_are_correct_format():
    dataset = Hznu.download()

    for image_path, mask_path in dataset.paths:
        assert image_path.suffix == ".jpg"
        assert mask_path.suffix == ".png"
        assert str(image_path).split(".jpg")[0] == str(mask_path).split(".png")[0]


def test_rasterize_mask_single_color():
    annotations = {
        "shapes": [
            {
                "label": "building",
                "points": [
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [2.0, 2.0],
                    [0.0, 2.0],
                ],
            },
        ]
    }
    building = HznuClasses.BUILDING.to_color()

    mask = np.array(
        [
            [building, building],
            [building, building],
        ],
        dtype=np.uint8,
    )

    actual = np.array(_rasterize_mask(annotations, 2, 2))
    assert (actual == mask).all()


def test_rasterize_mask_two_colors():
    annotations = {
        "shapes": [
            {
                "label": "sky",
                "points": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            },
            {
                "label": "building",
                "points": [
                    [1.0, 0.0],
                    [2.0, 2.0],
                    [2.0, 2.0],
                    [1.0, 2.0],
                ],
            },
        ]
    }
    background = HznuClasses.BACKGROUND.to_color()
    building = HznuClasses.BUILDING.to_color()

    mask = np.array(
        [
            [background, building],
            [background, building],
        ],
        dtype=np.uint8,
    )
    actual = np.array(_rasterize_mask(annotations, 2, 2))
    assert (actual == mask).all()


def test_rasterize_mask_three_overlapping_colors():
    annotations = {
        "shapes": [
            {
                "label": "sky",
                "points": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            },
            {
                "label": "building",
                "points": [
                    [1.0, 0.0],
                    [3.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                ],
            },
            {
                "label": "window",
                "points": [
                    [2.0, 0.0],
                    [3.0, 2.0],
                    [3.0, 2.0],
                    [1.0, 2.0],
                ],
            },
        ]
    }
    background = HznuClasses.BACKGROUND.to_color()
    building = HznuClasses.BUILDING.to_color()
    window = HznuClasses.WINDOW.to_color()

    mask = np.array(
        [
            [background, building, window],
            [background, building, window],
        ],
        dtype=np.uint8,
    )
    actual = np.array(_rasterize_mask(annotations, 3, 2))
    assert (actual == mask).all()
