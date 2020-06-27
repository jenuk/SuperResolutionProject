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
