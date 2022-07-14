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

### Basic run
Run the following from command line.
```
python3 main.py --algorithm fedavg --dataset mnist --num_users 10 --alpha 0.1 --sample_ratio 0.5 --glob_epochs 5 --local_epochs 1 --should_log True
```

### Logging
1. Enable logging by adding the command line argument `--should_log True` to `python3 main.py`.
2. Run `tensorboard --logdir=runs` and navigate to [http://localhost:6006/](http://localhost:6006/).

### Format
1. Run `black .` from the repo root.
2. Run `isort .` also from the repo root.

### Multi-GPU Example
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
