from torch.utils.data import DataLoader

from servers.server import Server
from users.user_fed_vae import UserFedVAE


class ServerFedVAE(Server):
    def __init__(self, base_params, z_dim, image_size):
        super().__init__(base_params)

        self.z_dim = z_dim
        self.image_size = image_size

    def create_users(self):
        for u in range(self.num_users):
            dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)

            new_user = UserFedVAE(
                {
                    "user_id": u,
                    "dataloader": dl,
                    "num_channels": self.num_channels,
                    "num_classes": self.num_classes,
                },
                self.z_dim,
                self.image_size
            )
            self.users.append(new_user)

    def train(self):
        for e in range(self.glob_epochs):
            self.evaluate(e)

            selected_users = self.sample_users()
            for u in selected_users:
                u.train(self.local_epochs)

            print(f"Finished training all users for epoch {e}")
            print("__________________________________________")

    def evaluate(self, e):
        print(e)

    def test(self):
        pass