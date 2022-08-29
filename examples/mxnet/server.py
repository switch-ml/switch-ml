import utility as utils
import sys
sys.path.insert(0, '/home/ubuntu/switchml')
from ml.server import start_server
from ml.strategy import FedAvg
from ml.parameter import weights_to_parameters
from mxnet import nd
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict


def fit_config(rnd: int):

    config = {
        "batch_size": 16,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):

    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


def get_eval_fn(model, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    if toy:
        # use only 10 samples as validation set
        trainset,testset = utils.load_partition(5)
    else:
        trainset,testset,_ = utils.load_data()

    # The `evaluate` function will be called after every round
    def evaluate(
        weights,
    ):
        # Update model with the latest parameters
        params = zip(model.collect_params(".*weight").keys(), weights)
        for key, value in params:
            model.collect_params().setattr(key, value)
        accuracy,loss = utils.test(model, testset)
        return loss, {"accuracy": accuracy}

    return evaluate


model = utils.get_model()
init = nd.random.uniform(shape=(2, 784))
model(init)

model_weights = utils.get_model_params(model)

strategy = FedAvg(
    fraction_fit=0.2,
    fraction_eval=0.2,
    min_fit_clients=2,
    min_eval_clients=2,
    min_available_clients=2,
    eval_fn=get_eval_fn(model, True),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
    initial_parameters=weights_to_parameters(model_weights),
)
start_server({"num_rounds": 20, "timeout": 10, "min_available_clients": 2}, strategy)
