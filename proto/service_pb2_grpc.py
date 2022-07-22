# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import service_pb2 as service__pb2


class SwitchmlServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.SendWeights = channel.unary_stream(
            "/switchml.SwitchmlService/SendWeights",
            request_serializer=service__pb2.SendWeightsRequest.SerializeToString,
            response_deserializer=service__pb2.SendWeightsResponse.FromString,
        )
        self.FetchWeights = channel.unary_unary(
            "/switchml.SwitchmlService/FetchWeights",
            request_serializer=service__pb2.FetchWeightsRequest.SerializeToString,
            response_deserializer=service__pb2.FetchWeightsResponse.FromString,
        )


class SwitchmlServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def SendWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def FetchWeights(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_SwitchmlServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "SendWeights": grpc.unary_stream_rpc_method_handler(
            servicer.SendWeights,
            request_deserializer=service__pb2.SendWeightsRequest.FromString,
            response_serializer=service__pb2.SendWeightsResponse.SerializeToString,
        ),
        "FetchWeights": grpc.unary_unary_rpc_method_handler(
            servicer.FetchWeights,
            request_deserializer=service__pb2.FetchWeightsRequest.FromString,
            response_serializer=service__pb2.FetchWeightsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "switchml.SwitchmlService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class SwitchmlService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def SendWeights(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/switchml.SwitchmlService/SendWeights",
            service__pb2.SendWeightsRequest.SerializeToString,
            service__pb2.SendWeightsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def FetchWeights(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/switchml.SwitchmlService/FetchWeights",
            service__pb2.FetchWeightsRequest.SerializeToString,
            service__pb2.FetchWeightsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )