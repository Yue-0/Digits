from torch import nn

__all__ = ["LeNet"]

LeNet = nn.Sequential(
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
