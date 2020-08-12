import torch
def resampling_rebalanced_crossentropy(seq_reduction):
    def loss(y_pred, y_true):
        prior_pos = torch.mean(y_true, dim=-1, keepdims=True)
        prior_neg = torch.mean(1-y_true, dim=-1, keepdim=True)
        eps=1e-10
        weight = y_true / (prior_pos + eps) + (1 - y_true) / (prior_neg + eps)
        ret =  -weight * (y_true * (torch.log(y_pred + eps)) + (1 - y_true) * (torch.log(1 - y_pred + eps)))
        if seq_reduction == 'mean':
            return torch.mean(ret, dim=-1)
        elif seq_reduction == 'none':
            return ret
    return loss

