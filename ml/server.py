import grpc
import pickle

import concurrent.futures


from proto.service_pb2_grpc import (
    SwitchmlServiceServicer,
    add_SwitchmlServiceServicer_to_server,
)
from proto.service_pb2 import FetchWeightsResponse, SendWeightsResponse
from .parameter import parameters_to_weights, weights_to_parameters
from redis import Redis


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
redis.flushall()


class SwitchmlServer(SwitchmlServiceServicer):
    def __init__(self, server):
        self.server = server
        self.strategy = server.strategy

    def FetchWeights(self, request, context):

        state = redis.get(self.strategy.name)

        if state is None:
            state = {}
        else:
            state = pickle.loads(state)

        weights = state.get(
            "weights", parameters_to_weights(self.strategy.initial_parameters)
        )

        params = weights_to_parameters(weights)

        config = {
            **self.server.config,
            **self.strategy.on_fit_config_fn(1),
            **self.strategy.on_evaluate_config_fn(1),
        }

        return FetchWeightsResponse(
            parameters=params,
            config=config,
        )

    def SendWeights(self, request, context):

        round = request.round
        self.server.push(round, request)

        round_status = True

        while round_status:
            rounds = redis.get(self.strategy.name)
            if rounds is None:
                state = {}
            else:
                state = pickle.loads(rounds)
            if "aggregated_weights" in state.get(round, {}):
                round_status = False

        config = {
            **self.server.config,
            **self.strategy.on_fit_config_fn(int(round.replace("round-", ""))),
            **self.strategy.on_evaluate_config_fn(int(round.replace("round-", ""))),
        }

        params = weights_to_parameters(state.get(round)["aggregated_weights"])

        yield SendWeightsResponse(
            parameters=params,
            config=config,
        )


class CustomServer:
    def __init__(self, strategy, config) -> None:

        self.rounds = {}
        self.strategy = strategy
        self.config = config

    def push(self, round, data):

        clients = self.rounds.get(round, [])
        clients.append(data)

        self.rounds[round] = clients

        if len(clients) >= self.config.get("min_available_clients", 2):

            parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
                clients
            )

            loss_aggregated, metrics_aggregated = self.strategy.aggregate_evaluate(
                clients
            )

            print(f"{round} AGGREGATED EVAL LOSS:", loss_aggregated, "\n")

            data = {
                "clients": [self.parse_fit_eval(client) for client in clients],
                "aggregated_weights": parameters_aggregated,
            }

            self.rounds[round] = data

            redis.set(self.strategy.name, pickle.dumps(self.rounds))

            self.rounds = {}

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
                "weights": weights,
                "num_examples": fit_examples,
                "metrics": fit_metrics,
            },
            "eval_res": {
                "loss": eval_loss,
                "num_examples": eval_examples,
                "metrics": eval_metrics,
            },
        }


def start_grpc_server(manager):
    server_address = "[::]:8000"
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=10),
        # Set the maximum number of concurrent RPCs this server will service before
        # returning RESOURCE_EXHAUSTED status, or None to indicate no limit.
        maximum_concurrent_rpcs=10,
        options=channel_options,
    )

    servicer = SwitchmlServer(manager)

    add_SwitchmlServiceServicer_to_server(servicer, server)

    server.add_insecure_port(server_address)

    server.start()

    return server


def start_server(config, strategy):

    cs = CustomServer(strategy, config)

    grpc_server = start_grpc_server(cs)

    # Evaluate with strategy initial parameters
    loss, metrics = strategy.evaluate(parameters=strategy.initial_parameters)

    print("LOSS:", loss)

    print("METRICS:", metrics)

    print("SWITCHML LISTENING FOR CLIENTS")

    grpc_server.wait_for_termination()
