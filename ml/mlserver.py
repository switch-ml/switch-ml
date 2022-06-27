from collections import OrderedDict
import torch
from torch.utils.data import DataLoader

from .utils import load_data, load_efficientnet_model, test

# import utils

# from .utils import load_efficientnet_model, load_data, test


def fit_config(rnd: int):
    """
    Conditions to train client model on client data
    """
    config = {
        "batch_size": 8,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """
    Conditions to test client model on client data
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


def get_eval_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for *******server-side evaluation*******."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 500 training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 500, n_train))
    valLoader = DataLoader(valset, batch_size=8)

    """The `evaluate` function will be called after every round."""

    def evaluate(weights):

        # Update model with the latest parameters from client
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # test the global model with client weights and test on server evaluation data.
        loss, accuracy = test(model, valLoader)
        return loss, {"accuracy": accuracy}

    return evaluate


def get_server_weights():
    # Server side weight intitialization
    model = load_efficientnet_model(classes=10)
    model_weights = [
        val.cpu().numpy().tolist() for _, val in model.state_dict().items()
    ]
    return model_weights
