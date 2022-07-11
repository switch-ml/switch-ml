from dataclasses import dataclass
import concurrent.futures
from logging import INFO
from typing import Dict, Optional, Tuple, Union

import grpc

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(SCRIPT_DIR))

from ml import service_pb2_grpc
from ml.server.servicer import SwitchmlServicer
from ml.server.client_manager import SimpleClientManager
from ml.server.server import Server
from ml.server.logger import log


DEFAULT_SERVER_ADDRESS = "[::]:8080"
GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # == 512 * 1024 * 1024


@dataclass
class Config:
    num_rounds: int = 1
    round_timeout: Optional[float] = None


def start_grpc_server(  # pylint: disable=too-many-arguments
    client_manager,
    server_address,
    max_message_length,
    max_concurrent_workers: int = 1000,
    keepalive_time_ms: int = 210000,
) -> grpc.Server:

    # Possible options:
    # https://github.com/grpc/grpc/blob/v1.43.x/include/grpc/impl/codegen/grpc_types.h
    options = [
        # Maximum number of concurrent incoming streams to allow on a http2
        # connection. Int valued.
        ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
        # Maximum message length that the channel can send.
        # Int valued, bytes. -1 means unlimited.
        ("grpc.max_send_message_length", max_message_length),
        # Maximum message length that the channel can receive.
        # Int valued, bytes. -1 means unlimited.
        ("grpc.max_receive_message_length", max_message_length),
        # The gRPC default for this setting is 7200000 (2 hours). Flower uses a
        # customized default of 210000 (3 minutes and 30 seconds) to improve
        # compatibility with popular cloud providers. Mobile Flower clients may
        # choose to increase this value if their server environment allows
        # long-running idle TCP connections.
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        # Setting this to zero will allow sending unlimited keepalive pings in between
        # sending actual data frames.
        ("grpc.http2.max_pings_without_data", 0),
    ]

    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_workers),
        # Set the maximum number of concurrent RPCs this server will service before
        # returning RESOURCE_EXHAUSTED status, or None to indicate no limit.
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=options,
    )

    servicer = SwitchmlServicer(client_manager)

    service_pb2_grpc.add_SwitchmlWeightsServiceServicer_to_server(servicer, server)

    if server_address is None:
        server_address = DEFAULT_SERVER_ADDRESS

    server.add_insecure_port(server_address)

    server.start()

    return server


def start_server(  # pylint: disable=too-many-arguments
    server_address,
    config,
    strategy=None,
    force_final_distributed_eval: bool = False,
):
    client_manager = SimpleClientManager()
    server = Server(client_manager=client_manager, strategy=strategy)

    initialized_config = Config(**config)

    # Start gRPC server
    grpc_server = start_grpc_server(
        client_manager=server.client_manager(),
        server_address=server_address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )

    num_rounds = initialized_config.num_rounds

    msg = f"Flower server running ({num_rounds} rounds)"
    log(INFO, msg)

    hist = server.fit(num_rounds=num_rounds, timeout=initialized_config.round_timeout)
    log(INFO, "app_fit: losses_distributed %s", str(hist.losses_distributed))
    log(INFO, "app_fit: metrics_distributed %s", str(hist.metrics_distributed))
    log(INFO, "app_fit: losses_centralized %s", str(hist.losses_centralized))
    log(INFO, "app_fit: metrics_centralized %s", str(hist.metrics_centralized))

    if force_final_distributed_eval:
        # Temporary workaround to force distributed evaluation
        server.strategy.eval_fn = None  # type: ignore

        # Evaluate the final trained model
        res = server.evaluate_round(rnd=-1, timeout=config.round_timeout)
        if res is not None:
            loss, _, (results, failures) = res
            log(INFO, "app_evaluate: federated loss: %s", str(loss))
            log(
                INFO,
                "app_evaluate: results %s",
                str([(res[0].cid, res[1]) for res in results]),
            )
            log(INFO, "app_evaluate: failures %s", str(failures))
        else:
            log(INFO, "app_evaluate: no evaluation result")

    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    # Stop the gRPC server
    grpc_server.stop(grace=1)

    return hist
