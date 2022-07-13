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
python3 main.py --num_users 2

### Format
1. Run `black .` from the repo root.
2. Run `isort .` also from the repo root.

### Multi-GPU Example
https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
