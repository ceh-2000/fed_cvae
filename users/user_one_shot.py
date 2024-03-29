import numpy as np
import torch
import torch.nn.functional as F

from users.user import User


class UserOneShot(User):
    def __init__(self, base_params, validation_data_loader):
        super().__init__(base_params)
        self.validation_data_loader = validation_data_loader

    def evaluate(self):
        with torch.no_grad():
            self.model.eval()

            total_correct = 0
            for batch_idx, (X_batch, y_batch) in enumerate(self.validation_data_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                # Forward pass through model
                test_logits = self.model(X_batch).cpu()
                pred_probs = F.softmax(input=test_logits, dim=1)
                y_pred = torch.argmax(pred_probs, dim=1)
                y_batch = y_batch.cpu()
                total_correct += np.sum((y_pred == y_batch).numpy())

        accuracy = round(
            total_correct / len(self.validation_data_loader.dataset) * 100, 2
        )

        return accuracy
