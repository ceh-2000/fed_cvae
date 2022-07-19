from torch import nn

class Decoder(nn.Module):
    def __init__(self, num_channels, image_size):
        super().__init__()
        self.model = None

    def forward(self, X):
        return self.model(X)

class ConditionalDecoder(nn.Module):
    def __init__(self, num_channels, image_size):
        super().__init__()
        self.model = None

    def forward(self, X):
        return self.model(X)
