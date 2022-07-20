from torch import sum as torch_sum

from users.user import User


class UserFedProx(User):
    def __init__(self, base_params, mu):
        super().__init__(base_params)

        self.mu = mu

        # This global model tracks the current aggregated model for the proximal term on the objective
        self.global_model = None

    def proximal_loss(self, local_model, global_model):
        """
        Squared loss on local model weights to ensure that the local model doesn't drift too far from the global model.
        """

        loss = 0
        for p1, p2 in zip(local_model.parameters(), global_model.parameters()):
            square_diff = (p1 - p2) ** 2
            loss += torch_sum(square_diff)

        return loss

    def train(self, local_epochs):
        self.model.train()

        for epoch in range(local_epochs):
            for batch_idx, (X_batch, y_batch) in enumerate(self.dataloader):
                # Forward pass through model
                output = self.model(X_batch)

                # Compute loss
                class_loss = self.loss_func(output, y_batch)
                reg_loss = self.proximal_loss(
                    self.model, self.global_model
                )  # proximal term on objective
                total_loss = class_loss + (self.mu / 2) * reg_loss

                # Gradient descent
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
