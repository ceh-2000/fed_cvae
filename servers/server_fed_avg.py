from torch.utils.data import DataLoader

from servers.server import Server
from users.user_fed_avg import UserFedAvg
from utils import average_weights


class ServerFedAvg(Server):
    def __init__(self, base_params):
        super().__init__(base_params)

    def create_users(self):
        """
        Every user gets an id, dataloader corresponding to their unique, private data, and info about the data
        This is a stored in a list of users.
        """
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)
            new_user = UserFedAvg(
                {
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                }
            )
            self.users.append(new_user)

    def train(self):
        """
        Train the global server model and local user models.
        After each epoch average the weights of all users.
        """
        for e in range(self.glob_epochs):
            self.evaluate(e)
            for u in self.users:
                u.train(self.local_epochs)

            # models = []
            # for u in self.users:
            #     models.append(u.model)
            # self.server_model = average_weights(models)

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")
