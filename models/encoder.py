from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None

    def forward(self, X):
        return self.model(X)
