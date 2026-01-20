from src.datasets import Hznu


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
        assert mask_path.suffix == ".json"
