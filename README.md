# FedVAE

### Description
This project implements a variety of Federated Learning models as well as a novel model FedVAE.

### Prerequisites
1. Python 3.9.x+
2. `pip`

### Set up
Install Python 3.9 and `pip`. We recommend using the package pyenv, which is described in this article.
Create and enter a new virtual environment and run:
```
pip3 install -r requirements.txt
```
This will install the necessary dependencies.

### Algorithms
#### Unachievable Ideal
Run the following from command line.
```
python3 main.py --algorithm central --dataset mnist --sample_ratio 0.1 --glob_epochs 5 --should_log 1
```
Because we are not training this model in a distributed manner, global epochs just refers to the number of epochs for our centralized model.

#### FedAvg
Run the following from command line.
```
python3 main.py --algorithm fedavg --dataset mnist --num_users 10 --alpha 0.1 --sample_ratio 0.1 --glob_epochs 5 --local_epochs 1 --should_log 1
```

#### FedProx
Run the following from command line.
```
python3 main.py --algorithm fedprox --dataset mnist --num_users 10 --alpha 0.1 --sample_ratio 0.1 --glob_epochs 5 --local_epochs 1 --should_log 1 --mu 1
```
The `mu` argument controls the weight on the proximal term in the local objective for users.

#### One-shot ensembled FL
Run the following from command line.
```
python3 main.py --algorithm oneshot --num_users 5 --alpha 1.0 --sample_ratio 0.1 --local_epochs 5 --should_log 1 --one_shot_sampling random --user_data_split 0.9 --K 3
```
`--one_shot_sampling` can take on the following values:
- `random` (sample a random subset of K users to ensemble)
- `validation` (split each user's data into training and validation and choose the K best scoring user models on the validation set)
- `data` (choose the K users with the most data)
- `all` (ensemble all user models)

You can also adjust model specific parameters `--K` to adjust how many users are sampled for ensembling and `--user_data_split` to adjust the user train/validation split. Note that you need to choose a K <= number of users.

#### FedVAE
Run the following from command line.
```
python3 main.py --algorithm fedvae --num_users 5 --alpha 1.0 --sample_ratio 0.1 --glob_epochs 5 --local_epochs 5 --should_log 1 --z_dim 50 --beta 1.0 --num_train_samples 1000 --classifier_epochs 5
```
You can adjust model specific parameers `--z_dim` to change the latent vector dimension and `--beta` to change the weight of the KL divergence loss.
Modify `--num_train_samples` to change how many samples are generated and `--classifier_epochs` to adjust the server model train time.
 
### Logging
1. Enable logging by adding the command line argument `--should_log 1` to `python3 main.py`.
2. Run `tensorboard --logdir=runs` and navigate to [http://localhost:6006/](http://localhost:6006/).

### Format
1. Run `black .` from the repo root.
2. Run `isort .` also from the repo root.

### Multi-GPU Example
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
