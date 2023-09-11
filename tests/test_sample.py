import numpy as np

from src.sample import Sample


def test_sample_default_name():
    sample = Sample()

    assert sample.name == "test"


def test_sample_custom_name():
    sample = Sample(name="train")

    assert sample.name == "train"


def test_sample_load_success():
    sample = Sample()
    sample.load()

    assert sample.normal_images[0].shape == (128, 128)
    assert sample.pneumo_images[0].shape == (128, 128)


def test_get_concatenated_data():
    sample = Sample()
    sample.load()

    data = sample.get_concatenated_images()
    _, cols, rows, extra = data.shape

    assert (cols, rows, extra) == (128, 128, 1)
