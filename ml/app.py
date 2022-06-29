from concurrent import futures
import time
import grpc
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from ml import service_pb2, service_pb2_grpc, utils, strategy, parameter
from ml.config import get_grpc_options

from torch.utils.data import DataLoader


_ONE_DAY_IN_SECONDS = 60 * 60 * 24

model = None

CLIENTS_DATA = []


def load_and_evaluate():
    global model

    model = utils.load_efficientnet(classes=10)

    _, testset = utils.get_dry_sets()

    valLoader = DataLoader(testset, batch_size=1)

    loss, results = utils.evaluate(model, utils.get_model_params(model), valLoader)

    print("Loss: ", loss)

    print("Results: ", results)


def client_weights():
    print("AGGREGATION STARTED")

    average_weights, average_metrics = strategy.federated_average(
        CLIENTS_DATA, strategy.metrics_average
    )

    print("Aggregated Metrics: ", average_metrics)

    load_and_evaluate()


class WeightService(service_pb2_grpc.SwitchmlWeightsServiceServicer):
    def SendWeights(self, request, context):

        print(f"INCOMING WEIGHTS")

        CLIENTS_DATA.append({"fit_res": request.fit_res, "eval_res": request.eval_res})

        client_weights()

        return service_pb2.SendWeightsResponse()

    def FetchWeights(self, request, context):

        print(f"FETCHING WEIGHTS FROM MODEL: {request}")

        weights = utils.get_model_params(model)

        print("SENDING WEIGHTS")

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

    load_and_evaluate()

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
