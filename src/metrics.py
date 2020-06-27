import torch

def mse_loss(out, targets):
    """
    return Mean-Squared-Error between `out` and `targets`.

    Parameters
    ----------
    out : torch.tensor
        of shape `(batch_size, channel, h, w)`.
    target : torch.tensor
        of shape `(batch_size, channel, h, w)`.

    Returns
    -------
    torch.tensor
        shape `()`
    """

    return torch.mean(torch.sum((out - targets)**2, axis=(1,2,3)), axis=0)