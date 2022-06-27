#!/usr/bin/env python3

import grpc


import json

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


import ml.mlserver as mlserver
import ml.service_pb2 as service_pb2
import ml.service_pb2_grpc as service_pb2_grpc


def send_weights(stub):
    weights = mlserver.get_server_weights()
    weights = json.dumps(weights)
    req = service_pb2.SendWeightsRequest(weights=weights)
    response = stub.SendWeights(req)
    print(response, "SENT WEIGHTS")


def fetch_weights(stub):
    req = service_pb2.FetchWeightsRequest()
    response = stub.FetchWeights(req)
    print(response, "RECIEVED WEIGHTS")


def run():
    with grpc.insecure_channel(
        "localhost:4000",
        options=[
            ("grpc.max_send_message_length", 512 * 1024 * 1024),
            ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ],
    ) as channel:

        stub = service_pb2_grpc.SwitchmlWeightsServiceStub(channel)

        send_weights(stub)

        fetch_weights(stub)


if __name__ == "__main__":
    run()
