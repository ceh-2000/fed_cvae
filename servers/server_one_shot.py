import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from servers.server import Server
from users.user import User
from users.user_one_shot import UserOneShot


class ServerOneShot(Server):
    def __init__(
        self,
        base_params,
        user_sampling_method,
        user_data_split,
        K,
        should_initialize_same,
    ):
        super().__init__(base_params)
        self.user_sampling_method = user_sampling_method
        self.user_data_split = user_data_split
        self.K = K
        self.dataloader = DataLoader(
            base_params["data_server"], shuffle=True, batch_size=1
        )
        self.should_initialize_same = should_initialize_same

    def create_users(self):
        """
        Every user gets an id, dataloader corresponding to their unique, private data, and info about the data
        This is a stored in a list of users.

        If the sampling method is "validation" then we want to split each users's data into train and validation
        using a `UserOneShot` object.
        """
        for u in range(self.num_users):
            if self.user_sampling_method == "validation":
                # Split the data into
                train_data_len = int(len(self.data_subsets[u]) * self.user_data_split)
                data = random_split(
                    self.data_subsets[u],
                    [train_data_len, len(self.data_subsets[u]) - train_data_len],
                )
                train_dl = DataLoader(data[0], shuffle=True, batch_size=32)
                valid_dl = DataLoader(data[1], shuffle=True, batch_size=32)
                new_user = UserOneShot(
                    {
                        "user_id": u,
                        "dataloader": train_dl,
                        "num_channels": self.num_channels,
                        "num_classes": self.num_classes,
                        "local_LR": self.local_LR,
                        "use_adam": self.use_adam,
                    },
                    valid_dl,
                )
            else:
                # Normal user generation (don't need to use special user class)
                dl = DataLoader(self.data_subsets[u], shuffle=True, batch_size=32)
                new_user = User(
                    {
                        "user_id": u,
                        "dataloader": dl,
                        "num_channels": self.num_channels,
                        "num_classes": self.num_classes,
                        "local_LR": self.local_LR,
                        "use_adam": self.use_adam,
                    },
                )
            self.users.append(new_user)

    def sample_users(self):
        if self.user_sampling_method == "random":
            # Sample uniformly without replacement from all users to get K elements
            return random.sample(self.users, self.K)
        elif self.user_sampling_method == "validation":
            accs = []
            for u in self.users:
                accs.append(u.evaluate())
            accs_sorted = np.argsort(accs)
            np_users = np.array(self.users)
            return list(np_users[accs_sorted[-self.K :]])
        elif self.user_sampling_method == "data":
            datas = []
            for u in self.users:
                datas.append(len(u.dataloader.dataset))
            datas_sorted = np.argsort(datas)
            np_users = np.array(self.users)
            return list(np_users[datas_sorted[-self.K :]])
        elif self.user_sampling_method == "all":
            return self.users
        else:
            raise NotImplementedError(
                "The specified method for sampling users for one shot FL has not been implemented."
            )

    def train(self):
        """
        Instead of training for multiple global iterations, allow all users to train and then ensemble select models
        """

        super().evaluate(0)

        # Ensure all models are initialized the same
        if self.should_initialize_same:
            weight_init_state_dict = self.users[0].model.state_dict()
            for u in self.users:
                u.model.load_state_dict(copy.deepcopy(weight_init_state_dict))

        # Train all local users once for as long as specified
        for u in self.users:
            u.train(self.local_epochs)

        print(f"Finished training all users for epoch 1.")
        print("__________________________________________")

    def evaluate(self, e):
        # Sample some or all users
        sampled_users = self.sample_users()

        # Ensemble results using a forward pass on the test set with majority vote
        num_correct = 0
        total = len(self.dataloader.dataset)
        with torch.no_grad():
            for X, y in self.dataloader:
                batch_pred = []
                for s in sampled_users:
                    test_logits = s.model(X)
                    pred_probs = F.softmax(input=test_logits, dim=1)
                    y_pred = torch.argmax(pred_probs, dim=1).tolist()
                    batch_pred.append(int(y_pred[0]))

                batch_pred = np.array(batch_pred).T
                y_pred_ensemble = np.bincount(batch_pred).argmax()

                if y_pred_ensemble == int(y[0]):
                    num_correct += 1

        accuracy = round(num_correct / total * 100, 2)
        if self.writer:
            self.writer.add_scalar("Global Accuracy/test", accuracy, e)

        print(f"Server model accuracy was: {accuracy}% on epoch {e}")

    def test(self):
        self.evaluate(1)
        print("Finished testing server.")
