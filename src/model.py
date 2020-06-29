from torch import nn
from utils import inner_pad

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 9, padding = 4)
        self.conv2 = nn.Conv2d(64, 32, 5, padding = 2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding = 2)
        self.conv4 = nn.Conv2d(32, 3, 5, padding = 2)


    def forward(self, x):
        out = inner_pad(x - 0.5, 2)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        return self.conv4(out)