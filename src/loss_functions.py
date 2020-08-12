import torch
from torch import nn
import torchvision

class PerceptionLoss(nn.Module):
    """
    calculates perceptual loss.
    """

    def __init__(self, model, *ind_layers):
        """
        Parameters
        ----------
        model : torch.module
            should have the method `get_features` (e.g. the discriminator and
            VGG below)
        ind_layers : sequence of ints
            indices to layers of which the features will be used to calculate
            the loss.
        """
        super(PerceptionLoss, self).__init__()

        self.model = model
        self.mse = nn.MSELoss()
        self.ind_layers = ind_layers

    def forward(self, x, targets):
        x_features = self.model.get_features(x, *self.ind_layers)
        with torch.no_grad():
            targets_features = self.model.get_features(targets, *self.ind_layers)

        loss = 0
        for xf, tf in zip(x_features, targets_features):
            loss = loss + self.mse(xf, tf)

        return loss

class VGG(nn.Module):
    """
    wrapper for torchvision.models.vgg16 that adds the get_features method
    """

    def __init__(self):
        super(VGG, self).__init__()

        self.model = torchvision.models.vgg16(True)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x, *ind_layers):
        """
        Parameters
        ----------
        x : torch.tensor
            input to network
        ind_layers : sequence of ints
            indices to layers from which results are to be returned

        Returns
        -------
        list of torch.tensor
            results from layers specified by ind_layers
        """
        res = []
        out = x
        for k, layer in enumerate(self.model.features):
            out = layer(out)
            if k == ind_layers[0]:
                res.append(out)
                if len(ind_layers) == 1:
                    break
                else:
                    ind_layers = ind_layers[1:]

        return res

class VGGPerceptionLoss(PerceptionLoss):
    """
    Default values for Perception loss, such that it calcualtes the orginal
    proposed feature loss.
    """

    def __init__(self):
        super(VGGPerceptionLoss, self).__init__(VGG(), 8)





class DiscriminatorLoss(nn.Module):
    """
    Calculates discrimantor loss for classic GAN setup.

    Loss function adapted as in lecture 9
    """


    def __init__(self, disc):
        super(DiscriminatorLoss, self).__init__()

        self.disc = disc

    def forward(self, x):
        return -torch.mean(torch.log(self.disc(x)))

class TotalVariationLoss(nn.Module):
    """
    calculates toal variation loss of an image
    """

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.tensor
            of shape (N, C, H, W)

        Returns
        -------
        torch.tensor
            shape (), batchwise mean total variation
        """
        res = torch.sum((x[..., :, 1:] - x[..., :, :-1])**2, axis=(1,2,3))
        res = res + torch.sum((x[..., 1:, :] - x[..., :-1, :])**2, axis=(1,2,3))
        res = torch.mean(res)

        return res