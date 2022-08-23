import grpc
import pickle

import concurrent.futures
from datetime import datetime, timedelta


from proto.service_pb2_grpc import (
    SwitchmlServiceServicer,
    add_SwitchmlServiceServicer_to_server,
)
from proto.service_pb2 import FetchWeightsResponse, SendWeightsResponse
from .parameter import parameters_to_weights, weights_to_parameters
from redis import Redis
from signal import signal, SIGINT


GRPC_MAX_MESSAGE_LENGTH = 536_870_912

channel_options = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_ping_strikes", 0),
]

redis = Redis(host="localhost", port=6379, db=0)
# redis.flushall()


class SwitchmlServer(SwitchmlServiceServicer):
    def __init__(self):
        self.strategy = None
        self.config = {}
        self.clients = []
        self.current_round = 1
        self.status = False
        self.server = None

    def initialize_self_values(self, strategy, config, server):
        self.config = config
        self.strategy = strategy
        self.server = server

    def add_client(self, ip):
        self.clients.append(ip)
        self.clients = list(set(self.clients))

    def parse_fit_eval(self, request):
        fit_res = request.fit_res
        eval_res = request.eval_res
        weights = parameters_to_weights(fit_res.parameters)
        fit_examples = fit_res.num_examples
        fit_metrics = {
            "train_accuracy": fit_res.metrics.get("train_accuracy", 0),
            "train_loss": fit_res.metrics.get("train_loss", 0),
            "val_loss": fit_res.metrics.get("val_loss", 0),
            "val_accuracy": fit_res.metrics.get("val_accuracy", 0),
        }

        eval_loss = eval_res.loss
        eval_examples = eval_res.num_examples
        eval_metrics = {"accuracy": eval_res.metrics.get("accuracy", 0)}

        return {
            "fit_res": {
                "num_examples": fit_examples,
                "metrics": fit_metrics,
            },
            "eval_res": {
                "loss": eval_loss,
                "num_examples": eval_examples,
                "metrics": eval_metrics,
            },
            "weights": weights,
        }

    def perform_strategy(self, round):

        key = f"{self.strategy.name}.{round}.weights"

        weights = redis.lrange(key, 0, -1)
        weights = [pickle.loads(val) for val in weights]

        parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(weights)

        loss_aggregated, metrics_aggregated = self.strategy.aggregate_evaluate(weights)

        print(f"\n{round} AGGREGATED EVAL LOSS:", loss_aggregated)

        loss, metrics = self.strategy.evaluate(parameters=parameters_aggregated)

        print("LOSS:", loss)

        print("METRICS:", metrics)

        redis.set(
            f"{self.strategy.name}.aggregated_weights",
            pickle.dumps(parameters_to_weights(parameters_aggregated)),
        )

        redis.set(
            f"{self.strategy.name}.{round}.aggregated_weights",
            pickle.dumps(parameters_to_weights(parameters_aggregated)),
        )

    def FetchWeights(self, request, context):

        self.add_client(context.peer())

        state = redis.get(f"{self.strategy.name}.aggregated_weights")
        state = pickle.loads(state) if state else None

        params = self.strategy.initial_parameters
        round = self.current_round

        if state:
            params = weights_to_parameters(state)
            # round = state.get("round", 1)

        config = {
            "round": round,
            **self.config,
            **self.strategy.on_fit_config_fn(1),
            **self.strategy.on_evaluate_config_fn(1),
        }

        return FetchWeightsResponse(
            parameters=params,
            config=config,
        )

    def SendWeights(self, request, context):

        round = request.round

        round_int = int(round.replace("round-", ""))

        data = self.parse_fit_eval(request)

        key = f"{self.strategy.name}.{round}.weights"

        agg_key = f"{self.strategy.name}.{round}.aggregated_weights"

        redis.set(agg_key, "")

        redis.rpush(key, pickle.dumps(data))

        round_status = True

        end_time = datetime.now() + timedelta(seconds=self.config.get("timeout", 60))

        while round_status:
            clients_len = len(self.clients)
            min_clients_available = self.config.get("min_available_clients", 2)

            if clients_len > min_clients_available:
                min_clients_available = clients_len

            if redis.llen(key) < min_clients_available:
                current_time = datetime.now()
                if current_time > end_time:
                    self.status = None
                    return
                continue

            state = redis.get(agg_key)

            if state == b"" and self.status is False:
                self.status = True
                self.perform_strategy(round)
                self.status = False
            elif state != b"":
                round_status = False

        self.current_round = round_int

        if round_int == self.config.get("num_rounds"):
            self.clients = []
            self.status = None

        config = {
            **self.config,
            **self.strategy.on_fit_config_fn(round_int),
            **self.strategy.on_evaluate_config_fn(round_int),
        }

        state = redis.get(f"{self.strategy.name}.aggregated_weights")

        parameters_aggregated = weights_to_parameters(pickle.loads(state))

        yield SendWeightsResponse(
            parameters=parameters_aggregated,
            config=config,
        )


def start_grpc_server():
    server_address = "[::]:8000"
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        # Set the maximum number of concurrent RPCs this server will service before
        # returning RESOURCE_EXHAUSTED status, or None to indicate no limit.
        maximum_concurrent_rpcs=10,
        options=channel_options,
    )

    servicer = SwitchmlServer()

    add_SwitchmlServiceServicer_to_server(servicer, server)

    server.add_insecure_port(server_address)

    return server, servicer


def start_server(config, strategy):

    grpc_server, servicer = start_grpc_server()

    # Evaluate with strategy initial parameters
    loss, metrics = strategy.evaluate(parameters=strategy.initial_parameters)

    print("LOSS:", loss)

    print("METRICS:", metrics)

    servicer.initialize_self_values(strategy, config, grpc_server)

    grpc_server.start()

    print("SWITCHML LISTENING FOR CLIENTS")

    end_time = datetime.now() + timedelta(seconds=config.get("timeout", 60))

    while True:
        if servicer.status is None:
            current_time = datetime.now()
            if current_time > end_time:
                break
    grpc_server.stop(grace=1)

    def handle_sigterm(*_):
        """Shutdown gracefully."""
        grpc_server.stop(grace=1)

    signal(SIGINT, handle_sigterm)

    grpc_server.wait_for_termination()