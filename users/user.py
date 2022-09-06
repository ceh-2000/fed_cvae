from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from models.classifier import Classifier


class User:
    def __init__(self, params):
        self.device = params["device"]

        self.user_id = params["user_id"]
        self.dataloader = params["dataloader"]

        self.num_channels = params["num_channels"]
        self.num_classes = params["num_classes"]

        self.model = Classifier(self.num_channels, self.num_classes).to(self.device)
        self.loss_func = CrossEntropyLoss()

        if params["use_adam"]:
            self.optimizer = Adam(self.model.parameters(), lr=params["local_LR"])
        else:
            self.optimizer = SGD(self.model.parameters(), lr=params["local_LR"])

        print(f"Created user {self.user_id}")

    def train(self, local_epochs):
        self.model.train()

        for epoch in range(local_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                output = self.model(X_batch)

                # Compute loss with pre-defined loss function
                loss = self.loss_func(output, y_batch)

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
