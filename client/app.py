from flask import Flask, Response
import grpc

import proto.weights_pb2 as weights_pb2
import proto.weights_pb2_grpc as weights_pb2_grpc

from google.protobuf.json_format import MessageToJson


app = Flask(__name__)


class SwitchMlCient:
    """
    Client for gRPC functionality
    """

    def __init__(self, host="localhost", port=8000):
        self.host = host
        self.server_port = port

        # instantiate a channel
        self.channel = grpc.insecure_channel(
            "{}:{}".format(self.host, self.server_port)
        )

        self.stub = weights_pb2_grpc.WeightServiceStub(self.channel)

    def fetch_weights(self):
        req = weights_pb2.FetchWeightsRequest()
        return self.stub.FetchWeights(req)

    def send_weights(self, weights):
        req = weights_pb2.SendWeightsRequest(weights=weights)
        return self.stub.SendWeights(req)


# app.config["client"] = SwitchMlCient()


@app.route("/")
def users_get():
    client = SwitchMlCient()
    result = client.send_weights("Client Weights")
    weights = client.fetch_weights()
    return Response(MessageToJson(weights), content_type="application/json")
