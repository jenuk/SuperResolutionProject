import torch

def inner_pad(x, factor, padding = 0):
    """
    Add padding between numbers in x.

    Parameters
    ----------
    x : torch.tensor
        of shape (..., h, w)
    factor : int
        factor by with to increase last two dimensions
    padding : float, optional
        Number to insert between elements of x (default `0`)

    Returns
    -------
    torch.tensor
        of shape (..., factor*h, factor*w)
    """

    res_shape = x.shape[:-2] + (factor*x.shape[-2], factor*x.shape[-1])
    res = torch.zeros(res_shape, requires_grad = True) + padding
    res[..., ::factor, ::factor] = x

    return res

def clip(x):
    """
    Clips values of x between 0 and 1.

    Parameters
    ----------
    x : torch.tensor
        any shape

    Returns
    -------
    torch.tensor
        same shape as x
    """

    out = torch.max(x, torch.tensor(0.0))
    out = torch.min(out, torch.tensor(1.0))
    return out