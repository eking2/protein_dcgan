import torch
from torch import Tensor

def flip_labels(y: Tensor, 
                prob_flip: float,
                pos_label: float = 0.9
                ) -> Tensor:

    """randomly flip true labels to introduce noise"""

    y_flip = y.clone()

    n_flip = int(prob_flip * len(y))
    flip_idx = torch.randperm(len(y))[:n_flip]
    y_flip[flip_idx] = pos_label - y_flip[flip_idx]

    return y_flip


