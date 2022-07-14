from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from model import MyModel


class User:
    def __init__(self, user_id, dataloader, num_channels, num_classes):
        self.user_id = user_id
        self.dataloader = dataloader

        self.num_channels = num_channels
        self.num_classes = num_classes

        self.model = MyModel(self.num_channels, self.num_classes).model
        self.loss_func = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)

        print(f'Created user {user_id}')

    def train(self, local_epochs):
        self.model.train()

        for epoch in range(local_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):
                # Forward pass through model
                output = self.model(X_batch)

                # Compute loss with pre-defined loss function
                loss = self.loss_func(output, y_batch)

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
