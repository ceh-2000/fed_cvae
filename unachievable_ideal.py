import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.classifier import Classifier


class UnachievableIdeal:
    def __init__(self, params):
        self.device = params["device"]
        self.epochs = params["glob_epoch"]

        self.train_data = DataLoader(params["train_data"], shuffle=True, batch_size=32)
        self.test_data = DataLoader(params["test_data"], shuffle=True, batch_size=32)

        self.num_channels = params["num_channels"]
        self.num_classes = params["num_classes"]

        self.model = Classifier(self.num_channels, self.num_classes).to(self.device)
        self.loss_func = CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

        self.writer = params["writer"]

    def train(self):
        self.model.train()

        for epoch in range(self.epochs):
            self.evaluate(epoch)
            for batch_idx, (X_batch, y_batch) in enumerate(self.train_data):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                output = self.model(X_batch)

                # Compute loss with pre-defined loss function
                loss = self.loss_func(output, y_batch)

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self, e):
        with torch.no_grad():
            self.model.eval()

            total_correct = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.test_data):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                test_logits = self.model(X_batch).cpu()
                pred_probs = F.softmax(input=test_logits, dim=1)
                y_pred = torch.argmax(pred_probs, dim=1)
                y_batch = y_batch.cpu()
                total_correct += np.sum((y_pred == y_batch).numpy())

            accuracy = round(total_correct / len(self.test_data.dataset) * 100, 2)
            print(f"Model accuracy was: {accuracy}% on epoch {e}")

            if self.writer:
                self.writer.add_scalar("Global Accuracy/test", accuracy, e)

    def test(self):
        self.evaluate(self.epochs)
        print("Finished testing server.")
