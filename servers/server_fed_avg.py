import copy

from servers.server import Server
from utils import average_weights


class ServerFedAvg(Server):
    def __init__(self, base_params):
        super().__init__(base_params)

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

            # Replace local user model weights with server model weights for SELECTED users
            server_model_weights = copy.deepcopy(self.server_model.state_dict())
            for u in selected_users:
                u.model.load_state_dict(server_model_weights)

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

    def train_alt_one_comm(self):
        """
        Train the global server model and local user models, checking the effect of increased local computation.
        """

        self.user_data_amts = [len(u.dataloader.dataset) for u in self.users]

        # Ensure that all users start with the same weight initialization
        server_model_weights = copy.deepcopy(self.server_model.state_dict())
        for u in self.users:
            u.model.load_state_dict(server_model_weights)

        # Train for a certain number of local epochs (one global epoch) and record results for each level of local training
        for i in range(self.local_epochs):
            self.evaluate(i)

            # Sample users for training
            selected_users = self.sample_users()

            # Train SELECTED user models and save model weights
            models = []
            for u in selected_users:
                u.train(1)
                models.append(u.model)

            # Average the weights of SELECTED user models and save in server
            state_dict = average_weights(models, data_amts=self.user_data_amts)
            self.server_model.load_state_dict(copy.deepcopy(state_dict))

            print(f"Finished training {len(selected_users)} users for local epoch {i}")
            print("__________________________________________")
