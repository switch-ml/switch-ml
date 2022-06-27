from concurrent import futures
import time
import grpc

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(SCRIPT_DIR))

from proto import weights_pb2, weights_pb2_grpc


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class WeightService(weights_pb2_grpc.WeightServiceServicer):
    def SendWeights(self, request, context):

        # metadata = dict(context.invocation_metadata())

        print(f"Request: {request.weights}")

        return weights_pb2.SendWeightsResponse()

    def FetchWeights(self, request, context):

        print(f"Request: {request}")

        return weights_pb2.FetchWeightsResponse(weights="Server Weights")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    weights_pb2_grpc.add_WeightServiceServicer_to_server(WeightService(), server)

    print("Starting server. Listening on port 8000.")

    server.add_insecure_port("0.0.0.0:8000")

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve()
