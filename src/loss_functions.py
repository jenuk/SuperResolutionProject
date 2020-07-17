import torch
from torch import nn
import torchvision

class PerceptionLoss(nn.Module):
    def __init__(self, model, *ind_layers):
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
    def __init__(self):
        super(VGG, self).__init__()

        self.model = torchvision.models.vgg16(True)

    def forward(self, x):
        return self.model(x)

    def get_features(self, x, *ind_layers):
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
    def __init__(self):
        super(VGGPerceptionLoss, self).__init__(VGG(), 8)

class DiscriminatorLoss(nn.Module):
    def __init__(self, disc):
        super(DiscriminatorLoss, self).__init__()

        self.disc = disc
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.criterion(x, torch.zeros(batch_size, 1, device=x.device))

class MixLoss(nn.Module):
    def __init__(self, **losses):
        super(MixLoss, self).__init__()

        self.losses = losses

    def forward(self, **args):
        loss = 0
        for key in args:
            loss = loss + self.losses[key][1] * self.losses[key][0](*args[key])

        return loss