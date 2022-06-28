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

    def num_users_per_device(self):
        if self.num_devices == 0:
            return 0
        return math.floor(self.num_users / self.num_devices)

    def start_and_end(self, a, b, i):
        factor = a / b
        start = math.floor(i * factor)
        end = math.floor((i + 1) * factor)
        if i == self.num_users - 1:
            return start, a + 1
        else:
            return start, end

    def create_users(self):
        for u in range(self.num_users):
            start, end = self.start_and_end(self.num_samples, self.num_users, u)
            X_data = self.X[start:end]
            y_data = self.y[start:end]

            self.users.append(User(X_data, y_data))

    def train(self):
        for e in range(self.glob_epochs):
            for d in self.devices:
                start, end = self.start_and_end(self.num_users, self.num_devices, d)
                cur_users = self.users[start:end]
                for c in cur_users:
                    c.move_to_device(self.devices[d])
                    print(f'Moved user to {self.devices[d]}')

            for u in self.users:
                u.train(self.local_epochs)

            print(f'Finished training all users for epoch {e}')









