from abc import ABC, abstractmethod


class Client(ABC):
    """Abstract base class for Flower clients."""

    def get_properties(self, ins):
        """Return set of client's properties.

        Parameters
        ----------
        ins : GetPropertiesIns
            The get properties instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetPropertiesRes
            The current client properties.
        """

    @abstractmethod
    def get_parameters(self, ins):
        """Return the current local model parameters.

        Parameters
        ----------
        ins : GetParametersIns
            The get parameters instructions received from the server containing
            a dictionary of configuration values.

        Returns
        -------
        GetParametersRes
            The current local model parameters.
        """

    @abstractmethod
    def fit(self, ins):
        """Refine the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : FitIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.

        Returns
        -------
        FitRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """

    @abstractmethod
    def evaluate(self, ins):
        """Evaluate the provided weights using the locally held dataset.

        Parameters
        ----------
        ins : EvaluateIns
            The evaluation instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local evaluation process.

        Returns
        -------
        EvaluateRes
            The evaluation result containing the loss on the local dataset and
            other details such as the number of local data examples used for
            evaluation.
        """


def has_get_properties(client: Client) -> bool:
    """Check if Client implements get_properties."""
    return type(client).get_properties != Client.get_properties
