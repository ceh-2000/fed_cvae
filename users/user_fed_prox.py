from users.user import User


class UserFedProx(User):
    def __init__(self, base_params, validation_data_loader):
        super().__init__(base_params)

    def train(self):
        pass
