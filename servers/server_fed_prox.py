import copy
from torch.utils.data import DataLoader

from servers.server import Server
from users.user_fed_prox import UserFedProx
from utils import average_weights


class ServerFedProx(Server):
    def __init__(self, base_params, mu):
        super().__init__(base_params)

        self.mu = mu

    def create_users(self):
        """
        Every user gets an id, dataloader corresponding to their unique, private data, info about the data, and mu
        This is a stored in a list of users.
        """
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)
            new_user = UserFedProx(
                {
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                },
                self.mu
            )
            self.users.append(new_user)

    def train(self):
        """
        Train the global server model and local user models.
        After each epoch average the weights of all users.
        """

        self.user_data_amts = [len(u.dataloader.dataset) for u in self.users]

        for e in range(self.glob_epochs):
            self.evaluate(e)

            # Sample users for training
            selected_users = self.sample_users()

            # Replace local user model weights with server model weights for SELECTED users and save a copy of the current global model
            server_model_weights = copy.deepcopy(self.server_model.state_dict())
            for u in selected_users:
                u.model.load_state_dict(server_model_weights)
                u.global_model = copy.deepcopy(self.server_model)

            # Train SELECTED user models and save model weights
            models = []
            for u in selected_users:
                u.train(self.local_epochs)
                models.append(u.model)

            # Average the weights of SELECTED user models and save in server
            state_dict = average_weights(models, data_amts=self.user_data_amts)
            self.server_model.load_state_dict(copy.deepcopy(state_dict))

            print(f"Finished training {len(selected_users)} users for epoch {e}")
            print("__________________________________________")
