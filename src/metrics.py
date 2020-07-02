import torch

def psnr(out, targets):
    """
    return peak signal-to-noise ratio between `out` and `targets`.

    Parameters
    ----------
    out : torch.tensor
        of shape `(batch_size, channel, h, w)`.
    target : torch.tensor
        of shape `(batch_size, channel, h, w)`.

    Returns
    -------
    torch.tensor
        shape `(batch_size)`
    """

    MSEs = torch.mean((out - targets)**2, axis=(1,2,3))
    maxs = targets
    while len(maxs.shape) > 1:
        maxs = torch.max(maxs, axis=1).values

    return 10*(2*torch.log(maxs) - torch.log(MSEs))