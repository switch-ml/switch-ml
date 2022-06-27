from concurrent import futures
from datetime import datetime
import time
import grpc


import service_pb2
import service_pb2_grpc

from google.protobuf.json_format import MessageToJson


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class WeightService(service_pb2_grpc.SwitchmlWeightsServiceServicer):
    def SendWeights(self, request, context):

        print(f"INCOMING WEIGHTS: {request.weights}")

        return service_pb2.SendWeightsResponse()

    def FetchWeights(self, request, context):

        print(f"FETCHING WEIGHTS: {request}")

        return service_pb2.FetchWeightsResponse(weights="Server Weights")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    service_pb2_grpc.add_SwitchmlWeightsServiceServicer_to_server(
        WeightService(), server
    )

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
