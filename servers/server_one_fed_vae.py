from servers.server_fed_vae import ServerFedVAE


class ServerOneFedVAE(ServerFedVAE):
    def __init__(
        self,
        base_params,
        z_dim,
        image_size,
        beta,
        classifier_num_train_samples,
        classifier_epochs,
    ):
        super().__init__(
            base_params,
            z_dim,
            image_size,
            beta,
            classifier_num_train_samples,
            classifier_epochs,
            None,
            None,
        )

    def train(self):
        self.evaluate(0)

        selected_users = self.sample_users()

        # Train selected users and collect their decoder weights
        decoders = []
        for u in selected_users:
            u.train(self.local_epochs)
            decoders.append(u.model.decoder)

        print(f"Finished training user models for epoch 0")

        # Qualitative image check - misc user!
        self.qualitative_check(
            1, self.users[0].model.decoder, "Novel images user 0 decoder"
        )

        # Generate a dataloader holding the generated images and labels
        self.classifier_dataloader = self.generate_dataset_from_user_decoders(
            selected_users, self.classifier_num_train_samples
        )
        print(
            f"Generated {len(self.classifier_dataloader.dataset)} samples to train server classifier for epoch 0"
        )

        # Train the server model's classifier
        self.train_classifier(reinitialize_weights=True)
        print(f"Trained server classifier for epoch 0")

        print("__________________________________________")
