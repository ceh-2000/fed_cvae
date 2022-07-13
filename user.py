import torch

from model import MyModel


class User:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.model = MyModel().model
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

    def move_to_device(self, device):
        self.X.to(device)
        self.y.to(device)
        self.model.to(device)

    def train(self, local_epochs):
        for epoch in range(local_epochs):
            y1 = self.model(self.X)
            loss = self.loss_func(y1, self.y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
