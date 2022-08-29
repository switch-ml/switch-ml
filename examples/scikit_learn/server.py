import utility as utils

from ml.server import start_server
from ml.strategy import FedAvg
from ml.parameter import weights_to_parameters
from sklearn.metrics import log_loss

def fit_config(rnd: int):

    config = {
    }
    return config


def evaluate_config(rnd: int):

    val_steps = 5 if rnd < 4 else 10
    return {"val_steps": val_steps}


def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset,testset, _ = utils.load_mnist()

    n_train = len(trainset[0])
    # The `evaluate` function will be called after every round
    def evaluate(
        weights,
    ):
        # Update model with the latest parameters
        model = utils.set_model_params(weights)
        x_test,y_test = utils.load_mnist()
        loss = log_loss(x_test,y_test)
        accuracy  =  model.score(x_test,y_test)
        return loss, {"accuracy": accuracy}
    return evaluate


model = utils.get_model()

model_weights = utils.get_model_parameters(model)

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
