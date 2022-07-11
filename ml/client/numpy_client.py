from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np


import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ml.parameter import parameters_to_weights, weights_to_parameters, Parameters
from ml.typings import (
    GetParametersRes,
    GetPropertiesRes,
    Status,
    Code,
    FitRes,
    EvaluateRes,
    Metrics,
)


from .client import Client

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[List[np.ndarray], int, Dict[str, Scalar]]

Example
-------

    model.get_weights(), 10, {"accuracy": 0.95}

"""

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE = """
NumPyClient.evaluate did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[float, int, Dict[str, Scalar]]

Example
-------

    0.5, 10, {"accuracy": 0.95}

"""


class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    def get_properties(self, config):
        """Returns a client's set of properties.

        Parameters
        ----------
        config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """

    @abstractmethod
    def get_parameters(self, config) -> List[np.ndarray]:
        """Return the current local model parameters.

        Parameters
        ----------
        config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The local model parameters as a list of NumPy ndarrays.
        """

    @abstractmethod
    def fit(self, parameters: List[np.ndarray], config):
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    @abstractmethod
    def evaluate(self, parameters: List[np.ndarray], config):
        """Evaluate the provided weights using the locally held dataset.

        Parameters
        ----------
        parameters : List[np.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """


def has_get_properties(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != NumPyClient.get_properties


class NumPyClientWrapper(Client):
    """Wrapper which translates between Client and NumPyClient."""

    def __init__(self, numpy_client: NumPyClient) -> None:
        self.numpy_client = numpy_client

    def get_properties(self, ins):
        """Return the current client properties."""
        properties = self.numpy_client.get_properties(config=ins.config)
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties=properties,
        )

    def get_parameters(self, ins):
        """Return the current local model parameters."""
        parameters = self.numpy_client.get_parameters(config=ins.config)
        parameters_proto = weights_to_parameters(parameters)
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"), parameters=parameters_proto
        )

    def fit(self, ins):
        """Refine the provided weights using the locally held dataset."""
        # Deconstruct FitIns
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        # Train
        results: Tuple[List[np.ndarray], int, Metrics] = self.numpy_client.fit(
            parameters, ins.config
        )
        if not (
            len(results) == 3
            and isinstance(results[0], list)
            and isinstance(results[1], int)
            and isinstance(results[2], dict)
        ):
            raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

        # Return FitRes
        parameters_prime, num_examples, metrics = results
        parameters_prime_proto = weights_to_parameters(parameters_prime)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters_prime_proto,
            num_examples=num_examples,
            metrics=metrics,
        )

    def evaluate(self, ins):
        """Evaluate the provided parameters using the locally held dataset."""
        parameters: List[np.ndarray] = parameters_to_weights(ins.parameters)

        results: Tuple[float, int, Metrics] = self.numpy_client.evaluate(
            parameters, ins.config
        )
        if not (
            len(results) == 3
            and isinstance(results[0], float)
            and isinstance(results[1], int)
            and isinstance(results[2], dict)
        ):
            raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_EVALUATE)

        # Return EvaluateRes
        loss, num_examples, metrics = results
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=num_examples,
            metrics=metrics,
        )
