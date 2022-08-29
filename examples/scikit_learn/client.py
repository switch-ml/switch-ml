import warnings
import sys
sys.path.insert(0, '/home/ubuntu/switchml')
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import utility as utils

import sys
sys.path.insert(0, '/home/ubuntu/switchml')
import random
from ml.client import start_client


class CifarClient:
    def __init__(
        self,
        trainset,
        testset,
        validation_split: int = 0.1,
    ):
        
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = utils.set_model_params(model, parameters)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)
        X_train = self.trainset[0]
        y_train = self.trainset[1]
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(X_train)
        loss = log_loss(y_train, model.predict_proba(X_train))
        accuracy  = model.score(X_train,y_train)
        results = {"loss":loss,"accuracy":accuracy}
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        x_test = self.testset[0]
        y_test = self.testset[1]
        accuracy  =  model.score(x_test,y_test)
        loss  = log_loss(x_test,y_test)
        return float(loss), len(self.testset[0]), {"accuracy": float(accuracy)}


def main(toy):
    trainset,testset = utils.load_mnist()
    # Start Flower client
    client = CifarClient(trainset, testset)
    start_client("localhost:4000", client)


main(True)
