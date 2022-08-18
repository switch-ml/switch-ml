import utility as utils
import sys
sys.path.insert(0, '/home/ubuntu/switchml')
from ml.server import start_server
from ml.strategy import FedAvg
from ml.parameter import weights_to_parameters
import tensorflow as tf


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
    trainset,testset, _ = utils.load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valLoader, trainLoader = utils.load_partition(5)
        # valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        trainLoader,valLoader,_ = utils.load_data()
        # Use the last 5k training examples as a validation set
        # valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    
    # The `evaluate` function will be called after every round
    def evaluate(
        weights,
    ):
        # Update model with the latest parameters
        model.set_weights(weights)
        # params_dict = zip(model.state_dict().keys(), weights)
        # state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, valLoader)
        return loss, {"accuracy": accuracy}

    return evaluate


model = utils.get_model()

model_weights = model.get_weights()

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
