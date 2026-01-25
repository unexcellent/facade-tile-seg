from src.datasets import CMPFacade


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
