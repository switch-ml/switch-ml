import time
from logging import INFO
from typing import Optional, Union

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # == 512 * 1024 * 1024

from ml.server.logger import log
from ml.client.client import Client
from ml.client.connection import grpc_connection
from ml.client.message_handler import handle
from ml.client.numpy_client import NumPyClient, NumPyClientWrapper
from ml.client.numpy_client import has_get_properties as numpyclient_has_get_properties
from ml.service_pb2 import FetchWeightsRequest, Parameters, FitRes, SendWeightsRequest
from ml.parameter import parameters_to_weights, weights_to_parameters


ClientLike = Union[Client, NumPyClient]


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


def start_client(
    server_address: str,
    client: Client,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> None:

    while True:
        sleep_duration: int = 0
        with grpc_connection(
            server_address,
            max_message_length=grpc_max_message_length,
        ) as conn:
            stub = conn

            req = FetchWeightsRequest()
            res = stub.FetchWeights(req)

            weights = parameters_to_weights(res.parameters)

            weights, num_examples, results = client.fit(weights, fit_config())
            params = weights_to_parameters(weights)

            params = Parameters(tensors=params.tensors, tensor_type=params.tensor_type)

            fit_res = FitRes(
                parameters=params, num_examples=num_examples, metrics=results
            )

            req = SendWeightsRequest(fit_res=fit_res, round=1, client_id="123")

            res = stub.SendWeights(req)

            weights = parameters_to_weights(res.parameters)

            loss, count, results = client.evaluate(weights, evaluate_config())

            print("Eval Results: ", results)
            print("Eval Loss: ", loss)

            # while True:
            #     server_message = receive
            #     # print(server_message, "server_message")
            #     # client_message, sleep_duration, keep_going = handle(
            #     #     client, server_message
            #     # )
            #     keep_going = True

            #     if not keep_going:
            #         break
        if sleep_duration == 0:
            log(INFO, "Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        log(
            INFO,
            "Disconnect, then re-establish connection after %s second(s)",
            sleep_duration,
        )
        time.sleep(sleep_duration)


def start_numpy_client(
    server_address: str,
    client: NumPyClient,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
) -> None:

    # Wrap the NumPyClient
    flower_client = NumPyClientWrapper(client)

    # Delete get_properties method from NumPyClientWrapper if the user-provided
    # NumPyClient instance does not implement get_properties. This enables the
    # following call to start_client to handle NumPyClientWrapper instances like any
    # other Client instance (which might or might not implement get_properties).
    if not numpyclient_has_get_properties(client=client):
        del NumPyClientWrapper.get_properties

    # Start
    start_client(
        server_address=server_address,
        client=flower_client,
        grpc_max_message_length=grpc_max_message_length,
    )
