from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader


import torch


import warnings

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ml.server.fedavg import FedAvg
from ml.parameter import weights_to_parameters
from ml.server.app import start_server
from client import wrk_utils

warnings.filterwarnings("ignore")


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if rnd < 2 else 2,
    }
    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


def get_eval_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = wrk_utils.load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 5k training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    valLoader = DataLoader(valset, batch_size=16)
    # The `evaluate` function will be called after every round
    def evaluate(
        weights,
    ):
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = wrk_utils.test(model, valLoader)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )

    args = parser.parse_args()

    model = wrk_utils.load_efficientnet(classes=10)

    model_weights = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = FedAvg(
        fraction_fit=0.2,
        fraction_eval=0.2,
        min_fit_clients=1,
        min_eval_clients=1,
        min_available_clients=1,
        eval_fn=get_eval_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=weights_to_parameters(model_weights),
    )

    # Start Flower server for four rounds of federated learning
    start_server("0.0.0.0:8080", config={"num_rounds": 4}, strategy=strategy)


if __name__ == "__main__":
    main()
