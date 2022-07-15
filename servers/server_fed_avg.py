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

        for e in range(self.glob_epochs):
            self.evaluate(e)

            # Train and save user models to a list
            models = []
            for u in self.users:
                u.train(self.local_epochs)
                models.append(u.model)

            # Average the weights of the user models and send to server
            state_dict = average_weights(models)
            self.server_model.load_state_dict(copy.deepcopy(state_dict))

            # Replace local user model with server model
            for u in self.users:
                u.model.load_state_dict(copy.deepcopy(state_dict))

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")
