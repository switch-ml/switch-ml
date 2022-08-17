from io import BytesIO
from typing import cast

import numpy as np

from .typings import Parameters, Weights


def weights_to_parameters(weights: Weights) -> Parameters:
    """Convert NumPy weights to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def parameters_to_weights(parameters: Parameters) -> Weights:
    """Convert parameters object to NumPy weights."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)
