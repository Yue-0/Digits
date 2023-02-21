from torch import nn

__all__ = ["LeNet"]


class LeNet(nn.Module):
    """
    LeNet model.
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.Tanh(),
            nn.Conv2d(16, 120, 5),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)
