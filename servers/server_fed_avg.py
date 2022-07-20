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
