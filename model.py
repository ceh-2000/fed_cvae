from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential


class MyModel:
    def __init__(self, input_channels, output_size):
        super().__init__()
        self.model = Sequential(
            Conv2d(
                in_channels=input_channels,
                out_channels=3,
                kernel_size=(3, 3),
                padding=1,
            ),
            ReLU(),
            MaxPool2d((2, 2)),
            Conv2d(in_channels=3, out_channels=2, kernel_size=(3, 3), padding=1),
            ReLU(),
            MaxPool2d((2, 2)),
            Flatten(),
            Linear(98, output_size),
        )

    def forward(self, X):
        return self.model(X)
