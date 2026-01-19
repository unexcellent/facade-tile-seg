from src.datasets import IrregularFacades


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
