from contextlib import contextmanager
from logging import DEBUG, INFO
from queue import Queue
from typing import Callable, Iterator, Optional, Tuple

import grpc

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # == 512 * 1024 * 1024

from ml.server.logger import log
from ml.service_pb2_grpc import SwitchmlWeightsServiceStub
from ml.service_pb2 import FetchWeightsRequest


def on_channel_state_change(channel_connectivity: str) -> None:
    """Log channel connectivity."""
    log(DEBUG, channel_connectivity)


@contextmanager
def grpc_connection(
    server_address: str,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
):

    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    channel_options = [
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
    ]

    channel = grpc.insecure_channel(server_address, options=channel_options)
    log(INFO, "Opened insecure gRPC connection")

    channel.subscribe(on_channel_state_change)

    queue = Queue(maxsize=1)  # pylint: disable=unsubscriptable-object
    stub = SwitchmlWeightsServiceStub(channel)

    # print(server_message_iterator.parameters.tensors, "iter")

    # yield (server_message_iterator)
    return stub

    # iter(queue.get, None)

    # print(server_message_iterator, "iterator")

    # receive = lambda: next(server_message_iterator)
    # send = lambda msg: queue.put(msg, block=False)

    # try:
    #     yield (receive, send)
    # finally:
    #     # Make sure to have a final
    #     channel.close()
    #     log(DEBUG, "gRPC channel closed")
