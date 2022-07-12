from concurrent import futures
import time
import grpc
import sys
import os
import redis
import json

import pickle
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from ml import service_pb2, service_pb2_grpc, utils, strategy, parameter
from ml.config import get_grpc_options

from torch.utils.data import DataLoader
from threading import Thread


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

model = None

# CLIENTS_DATA = []

r = redis.Redis(host="localhost", port=6379, db=0)

r.flushdb()


def load_and_evaluate(dry_test=False, agg_weights=None):
    global model

    model = utils.load_efficientnet(classes=10)

    if dry_test:
        _, testset = utils.get_dry_sets()
        valLoader = DataLoader(testset, batch_size=1)
    else:
        trainset, _, _ = utils.load_data()
        testset = torch.utils.data.Subset(
            trainset, range(len(trainset) - 500, len(trainset))
        )
        valLoader = DataLoader(testset, batch_size=16)

    weights = agg_weights if agg_weights else utils.get_model_params(model)

    loss, results = utils.evaluate(model, weights, valLoader)

    print("Fit Loss: ", loss)

    print("Fit Results: ", results)

    print("\n")


def client_weights(CLIENTS_DATA):
    average_weights, average_metrics = strategy.federated_average(
        CLIENTS_DATA, strategy.metrics_average
    )

    print("Aggregated Metrics: ", average_metrics)

    load_and_evaluate(True, agg_weights=average_weights)

    return average_weights


def worker(data, round):
    state = r.get(f"round-{round}")

    if state is None:
        state = {"clients": [], "status": False}
    else:
        state = pickle.loads(state)

    state["clients"] += [data]
    r.set(f"round-{round}", pickle.dumps(state))

    if len(state["clients"]) >= 2:
        agg_weights = client_weights(state["clients"])
        state["aggregated_weights"] = agg_weights
        state["status"] = True
        r.set(f"round-{round}", pickle.dumps(state))


class WeightService(service_pb2_grpc.SwitchmlWeightsServiceServicer):
    def __init__(self):
        from queue import Queue

        self.queue = Queue()
        self.rounds = {}

    def SendWeights(self, request, context):

        print(f"INCOMING WEIGHTS")

        start = time.time()

        round = request.round

        end = time.time()

        print(f"ROUND-{round} WEIGHTS RECEIVED IN :", end - start, " Sec")

        fit_res = request.fit_res

        data = {
            "num_examples": fit_res.num_examples,
            "metrics": {
                "train_accuracy": fit_res.metrics.get("train_accuracy", 0),
                "train_loss": fit_res.metrics.get("train_loss", 0),
                "val_loss": fit_res.metrics.get("val_loss", 0),
                "val_accuracy": fit_res.metrics.get("val_accuracy", 0),
            },
            "weights": parameter.parameters_to_weights(fit_res.parameters),
        }
        t = Thread(target=worker, args=[data, round])
        t.daemon = True
        t.start()

        return service_pb2.SendWeightsResponse()

    def FetchWeights(self, request, context):

        # print(f"RETRIEVING {request.request} WEIGHTS FROM MODEL...")

        status = False

        if request.request == "initial":

            weights = utils.get_model_params(model)

            print("SENDING INITIAL WEIGHTS")

            params = parameter.weights_to_parameters(weights)

            status = True
        else:
            round = request.request

            state = r.get(round)

            if state:
                state = pickle.loads(state)
                if state["status"]:
                    params = parameter.weights_to_parameters(
                        state["aggregated_weights"]
                    )
                    status = True
                else:
                    params = parameter.weights_to_parameters([])
            else:
                params = parameter.weights_to_parameters([])

        params = service_pb2.Parameters(
            tensors=params.tensors, tensor_type=params.tensor_type
        )

        return service_pb2.FetchWeightsResponse(parameters=params, status=status)


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=get_grpc_options(),
    )

    service_pb2_grpc.add_SwitchmlWeightsServiceServicer_to_server(
        WeightService(), server
    )

    print("Starting server. Listening on port 8000.")

    server.add_insecure_port("0.0.0.0:8000")

    # Model Initialization

    load_and_evaluate(dry_test=True)

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
