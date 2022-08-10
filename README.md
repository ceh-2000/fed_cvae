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

The datasets available for benchmarking are [MNIST](http://yann.lecun.com/exdb/mnist/) and [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist). All below examples use MNIST.

Change to `--dataset fashionmnist` to use FashionMNIST. 

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
python3 main.py --algorithm oneshot --dataset mnist --num_users 5 --alpha 1.0 --sample_ratio 0.1 --local_epochs 5 --should_log 1 --one_shot_sampling random --user_data_split 0.9 --K 3
```
`--one_shot_sampling` can take on the following values:
- `random` (sample a random subset of K users to ensemble)
- `validation` (split each user's data into training and validation and choose the K best scoring user models on the validation set)
- `data` (choose the K users with the most data)
- `all` (ensemble all user models)

You can also adjust model specific parameters `--K` to adjust the number of sampled users for ensembling and `--user_data_split` to adjust the user train/validation split. Note that you need to choose a K <= number of users.

By default, one-shot ensembled FL only trains for 1 global epoch.

#### One-shot FedVAE
Run the following from command line. 
```
python3 main.py --algorithm onefedvae --dataset mnist --num_users 5 --alpha 1.0 --sample_ratio 0.1 --local_epochs 5 --should_log 1 --z_dim 50 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 5       
```
You can adjust model specific parameters `--z_dim` to change the latent vector dimension and `--beta` to change the weight of the KL divergence loss.
Modify `--classifier_num_train_samples` to change the number of generated samples to train the server classifier and `--classifier_epochs` to adjust the server classifier train time.

By default, one-shot FedVAE only trains for 1 global epoch.

#### FedVAE
Run the following from command line.
```
python3 main.py --algorithm fedvae --dataset mnist --num_users 5 --alpha 1.0 --sample_ratio 0.1 --glob_epochs 5 --local_epochs 5 --should_log 1 --z_dim 50 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 5 --decoder_num_train_samples 1000  --decoder_epochs 5         
```
You can adjust model specific parameters `--z_dim` to change the latent vector dimension and `--beta` to change the weight of the KL divergence loss.
Modify `--classifier_num_train_samples` to change the number of generated samples to train the server classifier and `--classifier_epochs` to adjust the server classifier train time.
Modify `--decoder_num_train_samples` to change the number of generated samples to train the server decoder and `--decoder_epochs` to adjust the server decoder train time.
 

### Logging
1. Enable logging by adding the command line argument `--should_log 1` to `python3 main.py`.
2. Run `tensorboard --logdir=runs` and navigate to [http://localhost:6006/](http://localhost:6006/).

### Format
1. Run `black .` from the repo root.
2. Run `isort .` also from the repo root.

### Distributed hyperparameter tuning
1. Modify `experiments/hyperparams_shell_gen.py` according to your preferences.
2. Run `python3 experiments/hyperparams_shell_gen.py`
3. Run `source hyperparam_runs_{HOST_NUMBER}.sh`.

Notes: This is designed to distribute hyperparameter tuning across the W&M lab computers. 

### Multi-GPU Example
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
