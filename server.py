import math

import torch

from model import MyModel
from user import User


class Server:
    def __init__(self, devices, num_users, glob_epochs, local_epochs, data_subsets):
        self.devices = devices
        self.num_devices = len(self.devices)

        self.users = []
        self.num_users = num_users

        self.glob_epochs = glob_epochs
        self.local_epochs = local_epochs

        self.server_model = MyModel().model

        self.data_subsets = data_subsets
        self.populate_model_data()

    def populate_model_data(self):
        self.X_test = torch.index_select(
            self.data_subsets[len(self.data_subsets) - 1].dataset.data,
            0,
            torch.IntTensor(self.data_subsets[len(self.data_subsets) - 1].indices),
        )
        self.y_test = torch.index_select(
            self.data_subsets[len(self.data_subsets) - 1].dataset.targets,
            0,
            torch.IntTensor(self.data_subsets[len(self.data_subsets) - 1].indices),
        )
        print(self.X_test.shape)
        print(self.y_test.shape)

    def create_users(self):
        for u in range(self.num_users):
            X_user = torch.index_select(
                self.data_subsets[u].dataset.data,
                0,
                torch.IntTensor(self.data_subsets[u].indices),
            )
            y_user = torch.index_select(
                self.data_subsets[u].dataset.targets,
                0,
                torch.IntTensor(self.data_subsets[u].indices),
            )

            new_user = User(X_user, y_user)
            self.users.append(new_user)

    def train(self, writer):
        for e in range(self.glob_epochs):
            for u in self.users:
                u.train(self.local_epochs)

            self.evaluate(writer, e)
            print(f"Finished training all users for epoch {e}")

    def evaluate(self, writer, e):
        if writer:
            writer.add_scalar("Global Accuracy/train", 50 + e, e)

    def test(self):
        print("Finished testing server.")
        pass
