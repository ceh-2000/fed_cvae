import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.model import MyModel
from users.user import User


class Server:
    def __init__(self, params):
        self.devices = params["devices"]
        self.num_devices = len(self.devices)

        self.users = []
        self.num_users = params["num_users"]

        self.user_fraction = params["user_fraction"]
        self.num_users_per_round = int(self.user_fraction * self.num_users)

        self.glob_epochs = params["glob_epochs"]
        self.local_epochs = params["local_epochs"]

        self.data_subsets = params["data_subsets"]
        self.dataloader = DataLoader(params["data_server"], shuffle=True, batch_size=32)

        self.num_channels = params["num_channels"]
        self.num_classes = params["num_classes"]
        self.server_model = MyModel(self.num_channels, self.num_classes)

        self.writer = params["writer"]

    def create_users(self):
        """
        Every user gets an id, dataloader corresponding to their unique, private data, and info about the data
        This is a stored in a list of users.
        """
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)
            new_user = User(
                {
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                }
            )
            self.users.append(new_user)

    def sample_users(self):
        """
        Sample a portion of users for training according to the indicated sample fraction

        :return: The chosen users
        """
        if self.user_fraction == 1:
            return self.users

        sampled_user_idxs = np.random.choice([i for i in range(self.num_users)], size = self.num_users_per_round, replace = False)
        selected_users = np.array(self.users)[sampled_user_idxs]

        return selected_users

    def train(self):
        """
        Train the global server model and local user models
        """
        for e in range(self.glob_epochs):
            self.evaluate(e)

            selected_users = self.sample_users()
            for u in selected_users:
                u.train(self.local_epochs)

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")

    def evaluate(self, e):
        """
        Evaluate the global server model by comparing the predicted global labels to the actual test labels

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

            if self.writer:
                self.writer.add_scalar("Global Accuracy/test", accuracy, e)

        self.server_model.train()

    def test(self):
        self.evaluate(self.glob_epochs)
        print("Finished testing server.")
