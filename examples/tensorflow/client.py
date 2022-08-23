import sys
sys.path.insert(0, '/home/ubuntu/switchml')
import utility 
import torch
import random
import tensorflow as tf
from ml.client import start_client
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

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
        trainloader,testloader,_ = utility.load_data()
        results = utility.train(model, trainloader, testloader, epochs, self.device)
        parameters_prime = model.get_weights()
        # parameters_prime = utility.get_model_params(model)
        num_examples_train = len(trainloader[0])

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)
        trainloader,testloader,_ = utility.load_data()
        # Get config values
        steps: int = int(config["val_steps"])
        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = utility.test(model, testloader, steps, self.device)
        # print("Test Accuracy: {}".format(accuracy))
        return float(loss[0]), len(self.testset), {"accuracy": float(accuracy[0])}


def main(toy):
    index = random.randint(1, 10)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainset, testset,_= utility.load_data()
    if toy:
        testset,trainset = utility.load_partition(5)
    # Start Flower client
    client = CifarClient(trainset, testset, device)
    start_client("localhost:4000", client)
main(True)
