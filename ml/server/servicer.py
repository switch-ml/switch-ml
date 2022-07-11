from typing import Callable, Iterator

import grpc

import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ml import service_pb2_grpc, service_pb2, utils, parameter


class SwitchmlServicer(service_pb2_grpc.SwitchmlWeightsServiceServicer):
    """SwitchmlServicer for bi-directional gRPC message stream."""

    def __init__(
        self,
        client_manager,
    ):
        self.client_manager = client_manager

    def SendWeights(self, request, context):
        pass

    def FetchWeights(self, request, context):
        model = utils.load_efficientnet(classes=10)

        weights = utils.get_model_params(model)

        print("SENDING INITIAL WEIGHTS")

        params = parameter.weights_to_parameters(weights)

        return service_pb2.FetchWeightsResponse(parameters=params)
