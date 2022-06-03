from src import utils
import torch

POS_LABEL = 0.9
PROB_FLIP = 0.5
SIZE = (10,)

def test_flip_labels_neg():

    y = torch.full(SIZE, POS_LABEL)
    y_flip = utils.flip_labels(y, PROB_FLIP, POS_LABEL)
    
    assert y_flip.sum() < y.sum()

def test_flip_labels_pos():

    y = torch.full(SIZE, 0.)
    y_flip = utils.flip_labels(y, PROB_FLIP, POS_LABEL)

    assert y_flip.sum() > y.sum()
