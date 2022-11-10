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

The datasets available for benchmarking are [MNIST](http://yann.lecun.com/exdb/mnist/), [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), and [SVHN](http://ufldl.stanford.edu/housenumbers/). All below examples use MNIST.

Change to `--dataset fashion` to use FashionMNIST. 
Chage to `--dataset svhn` to use SVHN.

#### Unachievable Ideal
Run the following from command line.
```
python3 main.py --algorithm central --dataset mnist --sample_ratio 0.1 --glob_epochs 5 --should_log 1 
```
Because we are not training this model in a distributed manner, global epochs just refers to the number of epochs for our centralized model.

#### FedAvg
Run the following from command line.
```
python3 main.py --algorithm fedavg --dataset mnist --num_users 10 --alpha 0.1 --sample_ratio 0.1 --glob_epochs 5 --local_epochs 1 --should_log 1 --use_adam 1
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
python3 main.py --algorithm oneshot --dataset mnist --num_users 5 --alpha 1.0 --sample_ratio 0.1 --local_epochs 5 --should_log 1 --one_shot_sampling random --user_data_split 0.9 --K 3 --use_adam 1
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
python3 main.py --algorithm onefedvae --dataset mnist --num_users 5 --alpha 1.0 --sample_ratio 0.1 --local_epochs 5 --should_log 1 --z_dim 50 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 5 --uniform_range "(-1.0, 1.0)" --use_adam 1       
```
You can adjust model specific parameters `--z_dim` to change the latent vector dimension and `--beta` to change the weight of the KL divergence loss.
Modify `--classifier_num_train_samples` to change the number of generated samples to train the server classifier and `--classifier_epochs` to adjust the server classifier train time.
Modify `--uniform_range` to change the uniform range that the decoder uses to draw samples.

By default, one-shot FedVAE only trains for 1 global epoch.

#### FedVAE
Run the following from command line.
```
python3 main.py --algorithm fedvae --dataset mnist --num_users 5 --alpha 1.0 --sample_ratio 0.1 --glob_epochs 5 --local_epochs 5 --should_log 1 --z_dim 50 --beta 1.0 --classifier_num_train_samples 1000 --classifier_epochs 5 --decoder_num_train_samples 1000 --decoder_epochs 5 --uniform_range "(-1.0, 1.0)" --use_adam 1  
```
You can adjust model specific parameters `--z_dim` to change the latent vector dimension and `--beta` to change the weight of the KL divergence loss.
Modify `--classifier_num_train_samples` to change the number of generated samples to train the server classifier and `--classifier_epochs` to adjust the server classifier train time.
Modify `--decoder_num_train_samples` to change the number of generated samples to train the server decoder and `--decoder_epochs` to adjust the server decoder train time.
Modify `--uniform_range` to change the uniform range that the decoder uses to draw samples.

### Experiments
1. `--should_weight_exp`: Turn on (`1`) or off (`0`) weighting when averaging models.
2. `--should_initialize_models_same`: Turn on (`1`) or off (`0`) initializing all user models with the same weights.
3. `--should_avg_exp`: Turn on (`1`) or off (`0`) averaging all user decoders for the server decoder (FedVAE-specific).
4. `--should_fine_tune_exp`: Turn on (`1`) or off (`0`) fine-tuning the server decoder (FedVAE-specific).
5. `--heterogeneous_models_exp`: Choose whether to use heterogeneous models or not. Pass in a string containing which versions of the CVAE to use. Passing in a string of length 1 yields homogeneous models. Version 0 is the standard CVAE, version 1 is a smaller alternate, and version 2 is ResNet-based. Ex. `"012"`
6. `--transform_exp`: Choose whether to apply transforms for FedVAE with SVHN (`1`) or not (`0`). 

### Logging
1. Enable logging by adding the command line argument `--should_log 1` to `python3 main.py`.
2. Run `tensorboard --logdir=runs` and navigate to [http://localhost:6006/](http://localhost:6006/).

### Format
1. Run `black .` from the repo root.
2. Run `isort .` also from the repo root.

### Using the scripts in `experiments`
1. Modify `experiments/[EXPERIMENT_NAME].py` according to your preferences.
2. `cd experiments`
3. Run `python3 [EXPERIMENT_NAME].py`
4. Run `nohup sh [EXPERIMENT_NAME]_{HOST_NUMBER}.sh &`
5. Copy the results to local machine: `scp -r ceheinbaugh@th121-1.cs.wm.edu:/home/ceheinbaugh/Desktop/fed_vae/experiments/runs/ /Users/clareheinbaugh/Desktop/fed_vae/`

Notes: This is designed to distribute hyperparameter tuning across the W&M lab computers. 

### Multi-GPU Example
Explanation of Multi-GPU setup.
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
