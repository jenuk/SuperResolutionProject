import torch
from torch import nn

class Discriminator(nn.Module):
    """
    torch Module that earns p(upscaled | img)

    Discriminator to distinguish upscaled from original high resolution images.
    Offers `get_features` to access feautres after different layers.
    See report for schematic.
    """

    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size

        self.convs = nn.ModuleList([
            nn.Sequential(                        # 0
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(                        # 1
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(                        # 2
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, 2),# -> input_size/2  # 3
            nn.Sequential(                        # 4
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(                        # 5
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, 2),# -> input_size/4  # 6
            nn.Sequential(                        # 7
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
            ),
            nn.Sequential(                        # 8
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, 2),# -> input_size/8  # 9
            *(nn.Sequential(
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            ) for _ in range(3)),
            nn.Conv2d(64, 16, 1),
        ]) # -> (N, 16, input_size/2^6, input_size/2^6)

        self.fc1 = nn.Linear(16*(input_size//(2**6))**2, 100)
        self.fc2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # -> 1 for real data
        # -> 0 for upscaled data

    def forward(self, x):
        out = x
        for k, layer in enumerate(self.convs):
            out = layer(out)
        out = out.reshape(-1, 16*(self.input_size//(2**6))**2)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out

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
        for k, layer in enumerate(self.convs):
            out = layer(out)
            if k == ind_layers[0]:
                res.append(out)
                if len(ind_layers) == 1:
                    break
                else:
                    ind_layers = ind_layers[1:]

        return res