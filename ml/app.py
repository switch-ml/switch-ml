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
        trainset,_,_ = utils.load_data()
        testset = torch.utils.data.Subset(trainset, range(len(trainset) - 500, len(trainset)))
        valLoader = DataLoader(testset, batch_size=16)

    weights = agg_weights if agg_weights else utils.get_model_params(model)

    loss, results = utils.evaluate(model, weights, valLoader)

    print("Fit Loss: ", loss)

    print("Fit Results: ", results)


def client_weights(CLIENTS_DATA):
    average_weights, average_metrics = strategy.federated_average(
        CLIENTS_DATA, strategy.metrics_average
    )

    print("Aggregated Metrics: ", average_metrics)

    load_and_evaluate(agg_weights=average_weights)

    return average_weights

    # params = parameter.weights_to_parameters(average_weights)

    # params = service_pb2.Parameters(
    #     tensors=params.tensors, tensor_type=params.tensor_type
    # )
    # return params   


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

        print(f"ROUND-{round} WEIGHTS RECEIVED IN :", end-start," Sec")

        client_id = request.client_id

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

        self.queue.put(data)

        state = r.get(f"round-{round}")
            
        if state is None:
            state = {"clients": [], "status": False}
        else:
            state = pickle.loads(state)

        if self.queue.qsize() >= 2:
            weights = client_weights([self.queue.get() for i in range(self.queue.qsize())])
            self.rounds[f"round-{round}"] = weights
            # self.rounds[f"round-{round}"] = True
            state["aggregated_weights"] = weights

            state["status"] = True
            r.set(f"round-{round}", pickle.dumps(state))
            
        
        
        while self.rounds.get(f"round-{round}") is None:
            pass

        params = parameter.weights_to_parameters(self.rounds.get(f"round-{round}"))
        # state = pickle.loads(r.get(f"round-{round}"))
        # params = parameter.weights_to_parameters(state["clients"])
        params = service_pb2.Parameters(
            tensors=params.tensors, tensor_type=params.tensor_type
        )

        print(f"ROUND-{round} AGGREGATION COMPLETED")
        return service_pb2.SendWeightsResponse(parameters=params)


            

    def FetchWeights(self, request, context):

        print(f"RETRIEVING INITIAL WEIGHTS FROM MODEL...")

        weights = utils.get_model_params(model)

        print("SENDING INITIAL WEIGHTS")

        params = parameter.weights_to_parameters(weights)

        params = service_pb2.Parameters(
            tensors=params.tensors, tensor_type=params.tensor_type
        )

        return service_pb2.FetchWeightsResponse(parameters=params)


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
