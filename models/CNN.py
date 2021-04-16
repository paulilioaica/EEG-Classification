from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 5, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(stride=1, padding=1, kernel_size=3))
        self.layer2 = nn.Sequential(
            nn.Conv2d(5, 1, padding=1, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(stride=1, padding=1, kernel_size=3))
        self.linear = nn.Linear(1024, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.linear(out.reshape(out.size(0), -1))
        return self.sigmoid(out)