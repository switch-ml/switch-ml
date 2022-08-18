import sys
sys.path.insert(0, '/home/ubuntu/switchml')
import utility as utils
import torch
import random
from ml.client import start_client

class CifarClient:
    def __init__(
        self,
        trainset,
        testset,
        device,
        validation_split = 0.1,
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
        model = utility.get_model()
        model.set_weights(parameters)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = int(config["batch_size"])
        epochs: int = int(config["local_epochs"])

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        results = utility.train(model, trainLoader, valLoader, epochs, self.device)

        parameters_prime = utility.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = int(config["val_steps"])

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        loss, accuracy = utility.test(model, testloader, steps, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def main(toy):
    index = random.randint(1, 10)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset, testset = utility.load_partition(index)

    if toy:
        trainset = torch.utils.data.Subset(trainset, range(10))
        testset = torch.utils.data.Subset(testset, range(10))

    # Start Flower client
    client = MnistClient(trainset, testset, device)

    start_client("localhost:4000", client)


main(True)
