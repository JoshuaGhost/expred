from typing import Callable

import torch


def resampling_rebalanced_crossentropy(seq_reduction : str ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns the loss function with given seq_reduction strategy.

    The individual token loss of token $i$ in sequence $S$ s $\vertS_{t^i}\vert \cdot BCE(p^i, t^i)$ where
    * $p^i$ and $t^i$ are the predicted and target labels of token $i$ respectively
    *  \vertS_{t^i}\vert is the number of tokens that have the same target a label as token $i$
    :param seq_reduction: either 'none' or 'mean'.
    :return:
    """
    def loss(y_pred, y_true):
        prior_pos = torch.mean(y_true, dim=-1, keepdims=True)  # percentage of positive tokens (rational)
        prior_neg = torch.mean(1 - y_true, dim=-1, keepdim=True)  # vice versa
        eps = 1e-10
        weight = y_true / (prior_pos + eps) + (1 - y_true) / (prior_neg + eps)
        ret = -weight * (y_true * (torch.log(y_pred + eps)) + (1 - y_true) * (torch.log(1 - y_pred + eps)))
        if seq_reduction == 'mean':
            return torch.mean(ret, dim=-1)
        elif seq_reduction == 'none':
            return ret

    return loss
