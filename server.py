import math

from model import MyModel
from user import User


class Server:
    def __init__(self, devices, num_users, glob_epochs, local_epochs, X, y):
        self.devices = devices
        self.num_devices = len(self.devices)

        self.users = []
        self.num_users = num_users

        self.glob_epochs = glob_epochs
        self.local_epochs = local_epochs

        self.server_model = MyModel().model

        self.X = X
        self.y = y
        self.num_samples = self.X.shape[0]

    def start_and_end(self, a, b, i):
        factor = a / b
        start = math.floor(i * factor)
        end = math.floor((i + 1) * factor)
        if i == b - 1:
            return start, a + 1
        else:
            return start, end

    def create_users(self):
        for u in range(self.num_users):
            start, end = self.start_and_end(self.num_samples, self.num_users, u)
            X_data = self.X[start:end]
            y_data = self.y[start:end]

            new_user = User(X_data, y_data)
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
