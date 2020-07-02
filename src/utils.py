import torch

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

    out = torch.max(x, torch.tensor(0.0, device=x.device))
    out = torch.min(out, torch.tensor(1.0, device=x.device))
    return out