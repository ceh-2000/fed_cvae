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
        print("HI")
