import numpy as np

from src.datasets.hznu import HznuClasses
from src.datasets.merged import MergedClasses


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
