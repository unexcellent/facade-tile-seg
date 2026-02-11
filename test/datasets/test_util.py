import numpy as np

from src.datasets.cmp_facade import CMPFacadeClasses
from src.datasets.hznu import HznuClasses
from src.datasets.merged_classes import MergedClasses


def test_convert_mask_to_merged():
    original_mask = np.array(
        [
            [HznuClasses.BACKGROUND, HznuClasses.BUILDING],
            [HznuClasses.WINDOW, HznuClasses.CAR],
        ]
    )
    right = np.array(
        [
            [MergedClasses.BACKGROUND, MergedClasses.USABLE],
            [MergedClasses.NOT_USABLE, MergedClasses.BACKGROUND],
        ]
    )
    actual = HznuClasses.convert_mask_to_merged(original_mask)
    assert (actual == right).all()


def test_convert_image_to_mask():
    dark_blue = (0, 0, 170)
    blue = (0, 0, 255)
    red = (255, 0, 0)

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


def test_convert_mask_to_image():
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

    dark_blue = (0, 0, 170)
    blue = (0, 0, 255)
    red = (255, 0, 0)

    image = np.array(
        [
            [dark_blue, blue, dark_blue],
            [blue, red, blue],
            [dark_blue, blue, dark_blue],
        ]
    )

    assert (CMPFacadeClasses.convert_mask_to_image(mask) == image).all()


def test_convert_mask_to_output():
    background = MergedClasses.BACKGROUND
    usable = MergedClasses.USABLE
    not_usable = MergedClasses.NOT_USABLE

    mask = np.array(
        [
            [background, usable, background],
            [usable, not_usable, usable],
            [background, usable, background],
        ],
        dtype=int,
    )

    background_seglayer = (1, 0, 0)
    usable_seglayer = (0, 1, 0)
    not_usable_seglayer = (0, 0, 1)

    output = np.array(
        [
            [background_seglayer, usable_seglayer, background_seglayer],
            [usable_seglayer, not_usable_seglayer, usable_seglayer],
            [background_seglayer, usable_seglayer, background_seglayer],
        ]
    )

    assert (MergedClasses.convert_mask_to_output(mask) == output).all()
