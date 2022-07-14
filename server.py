import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import MyModel
from user import User


class Server:
    def __init__(
        self,
        devices,
        num_users,
        glob_epochs,
        local_epochs,
        data_subsets,
        data_server,
        num_channels,
        num_classes,
    ):
        self.devices = devices
        self.num_devices = len(self.devices)

        self.users = []
        self.num_users = num_users

        self.glob_epochs = glob_epochs
        self.local_epochs = local_epochs

        self.data_subsets = data_subsets
        self.dataloader = DataLoader(data_server, shuffle=True, batch_size=32)

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.server_model = MyModel(num_channels, num_classes).model

    def create_users(self):
        """
        Every user gets an id, dataloader corresponding to their unique, private data, and info about the data
        This is a stored in a list of users.
        """
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)
            new_user = User(
                user_id=u,
                dataloader=dl,
                num_channels=self.num_channels,
                num_classes=self.num_classes,
            )
            self.users.append(new_user)

    def train(self, writer):
        """
        Train the global server model and local user models

        :param writer: Logger
        """
        for e in range(self.glob_epochs):
            self.evaluate(writer, e)
            for u in self.users:
                u.train(self.local_epochs)

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")

    def evaluate(self, writer, e):
        """
        Evaluate the global server model by comparing the predicted global labels to the actual test labels

        :param writer: Logger
        :param e: Global epoch number
        """
        with torch.no_grad():
            self.server_model.eval()

            total_correct = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):
                # Forward pass through model
                test_logits = self.server_model(X_batch)
                pred_probs = F.softmax(input=test_logits, dim=1)
                y_pred = torch.argmax(pred_probs, dim=1)
                total_correct += np.sum((y_pred == y_batch).numpy())

            accuracy = round(total_correct / len(self.dataloader.dataset) * 100, 2)
            print(f"Server model accuracy was: {accuracy}% on epoch {e}")

            if writer:
                writer.add_scalar("Global Accuracy/test", accuracy, e)

    def test(self):
        self.evaluate(None, self.glob_epochs)
        print("Finished testing server.")
        pass
