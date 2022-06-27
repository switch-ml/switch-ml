from concurrent import futures
import time
import grpc

import json

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


from ml import mlserver, service_pb2, service_pb2_grpc


from google.protobuf.json_format import MessageToJson


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class WeightService(service_pb2_grpc.SwitchmlWeightsServiceServicer):
    def SendWeights(self, request, context):

        print(f"INCOMING WEIGHTS: {request.weights}")

        return service_pb2.SendWeightsResponse()

    def FetchWeights(self, request, context):

        print(f"FETCHING WEIGHTS: {request}")

        weights = mlserver.get_server_weights()

        weights = json.dumps(weights)

        return service_pb2.FetchWeightsResponse(weights=weights)


def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ],
    )

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
