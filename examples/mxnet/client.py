import sys
sys.path.insert(0, '/home/ubuntu/switchml')
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag
import mxnet.ndarray as F
from ml.client import start_client
import utility as utils
import random


class MnistClient:
    def __init__(
        self,
        trainset,
        testset,
        device: str,
        validation_split: int = 0.1,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = utils.get_model()
        params = zip(model.collect_params(".*weight").keys(), parameters)
        for key, value in params:
            model.collect_params().setattr(key, value)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = int(config["batch_size"])
        epochs: int = int(config["local_epochs"])
        train_data, val_data, num_examples = utils.load_data()
        results =   utils.train(model, train_data, val_data, epochs, "cpu")
        parameters_prime = utils.get_model_params(model)
        num_examples_train = num_examples["trainset"]
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        train_data, val_data, num_examples = utils.load_data()
        loss, accuracy = utils.test(model, val_data, None, 'cpu')
        return float(loss), num_examples['testset'], {"accuracy": float(accuracy)}


def main(toy):
    index = random.randint(1, 10)

    if toy:
        trainset,testset = utils.load_partition(index)
    else:
        trainset,testset,_ = utils.load_data()
    # Start Flower client
    client = MnistClient(trainset, testset,'cpu',0.1)

    start_client("localhost:4000", client)


main(True)


