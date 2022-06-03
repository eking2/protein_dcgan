from src.datasets import Distograms
import pytest

@pytest.fixture
def dataset():
    path = 'tests/data/demo_disto.hdf5'
    return Distograms(path, 100)

def test_dataset_shape(dataset):
    for data in dataset:
        assert data.shape == (1, 64, 64)

def test_dataset_min_max(dataset):
    for data in dataset:
        assert data.max() <= 1
        assert data.min() >= 0
