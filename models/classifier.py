from torch import nn
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential
from torchsummary import summary


class Classifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.model = Sequential(
            Conv2d(
                in_channels=input_channels,
                out_channels=32,
                kernel_size=(5, 5),
            ),
            MaxPool2d((2, 2)),
            Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5)),
            MaxPool2d((2, 2)),
            Flatten(),
            Linear(1600, 512),
            ReLU(),
            Linear(512, num_classes),
        )

    def forward(self, X):
        return self.model(X)


class ClassifierBlind(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.model = Sequential(
            Conv2d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
            ),
            MaxPool2d((2, 2)),
            Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=(3, 3),
            ),
            MaxPool2d((2, 2)),
            Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
            ),
            MaxPool2d((2, 2)),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
            ),
            Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=(3, 3),
            ),
            MaxPool2d((2, 2)),
            Flatten(),
            Linear(32768, 4096),
            ReLU(),
            Linear(4096, num_classes),
        )

    def forward(self, X):
        return self.model(X)


if __name__ == "__main__":
    # summary(Classifier(1, 10).model, (1, 32, 32))

    summary(ClassifierBlind(1, 2).model, (1, 128, 128))
