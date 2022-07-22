import grpc

from proto.service_pb2 import EvalRes, FetchWeightsRequest, FitRes, SendWeightsRequest
from proto.service_pb2_grpc import SwitchmlServiceStub
from ml.parameter import parameters_to_weights, weights_to_parameters


GRPC_MAX_MESSAGE_LENGTH = 536_870_912

channel_options = [
    ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH),
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.http2.max_ping_strikes", 0),
]


class SwitchmlClient:
    def __init__(self, target):
        channel = grpc.insecure_channel(target, options=channel_options)
        self.stub = SwitchmlServiceStub(channel)

    def FetchWeights(self):
        req = FetchWeightsRequest()
        response = self.stub.FetchWeights(req)
        return response

    def SendWeights(self, fit_res, eval_res, round):
        req = SendWeightsRequest(fit_res=fit_res, eval_res=eval_res, round=round)

        return self.stub.SendWeights(req)


def start_client(server_address, client):

    service = SwitchmlClient(server_address)

    res = service.FetchWeights()

    weights = parameters_to_weights(res.parameters)

    config = res.config

    num_rounds = int(config.get("num_rounds"))

    for round in range(1, num_rounds + 1):

        agg_weights, fit_examples, fit_metrics = client.fit(weights, config)

        print(f"\nROUND-{round} FIT METRICS: ", fit_metrics)

        loss, eval_examples, eval_metrics = client.evaluate(agg_weights, config)

        print(f"ROUND-{round} EVAL LOSS: ", loss)
        print(f"ROUND-{round} EVAL METRICS: ", eval_metrics, "\n")

        params = weights_to_parameters(agg_weights)

        fit_res = FitRes(
            parameters=params,
            num_examples=fit_examples,
            metrics=fit_metrics,
        )

        eval_res = EvalRes(num_examples=eval_examples, metrics=eval_metrics, loss=loss)

        response = service.SendWeights(fit_res, eval_res, f"round-{round}")

        for res in response:
            config = res.config

    agg_weights, fit_examples, fit_metrics = client.fit(weights, config)

    loss, eval_examples, eval_metrics = client.evaluate(agg_weights, config)
