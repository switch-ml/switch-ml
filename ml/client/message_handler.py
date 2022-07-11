from typing import Tuple

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from ml.client.client import Client, has_get_properties
from ml import typings as typing


# pylint: disable=missing-function-docstring


class UnknownServerMessage(Exception):
    """Signifies that the received message is unknown."""


def handle(client, server_msg):
    field = server_msg.WhichOneof("msg")
    print(field, "field")
    return field, 0, True
    # if field == "reconnect_ins":
    #     disconnect_msg, sleep_duration = _reconnect(server_msg.reconnect_ins)
    #     return disconnect_msg, sleep_duration, False
    # if field == "get_properties_ins":
    #     return _get_properties(client, server_msg.get_properties_ins), 0, True
    # if field == "get_parameters_ins":
    #     return _get_parameters(client, server_msg.get_parameters_ins), 0, True
    # if field == "fit_ins":
    #     return _fit(client, server_msg.fit_ins), 0, True
    # if field == "evaluate_ins":
    #     return _evaluate(client, server_msg.evaluate_ins), 0, True
    # raise UnknownServerMessage()


# def _reconnect(
#     reconnect_msg,
# ):
#     # Determine the reason for sending DisconnectRes message
#     reason = Reason.ACK
#     sleep_duration = None
#     if reconnect_msg.seconds is not None:
#         reason = Reason.RECONNECT
#         sleep_duration = reconnect_msg.seconds
#     # Build DisconnectRes message
#     disconnect_res = ClientMessage.DisconnectRes(reason=reason)
#     return ClientMessage(disconnect_res=disconnect_res), sleep_duration


# def _get_properties(client: Client, get_properties_msg) -> ClientMessage:
#     # Check if client overrides get_properties
#     if not has_get_properties(client=client):
#         # If client does not override get_properties, don't call it
#         get_properties_res = typing.GetPropertiesRes(
#             status=typing.Status(
#                 code=typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED,
#                 message="Client does not implement get_properties",
#             ),
#             properties={},
#         )
#         get_properties_res_proto = serde.get_properties_res_to_proto(get_properties_res)
#         return ClientMessage(get_properties_res=get_properties_res_proto)

#     # Deserialize get_properties instruction
#     get_properties_ins = serde.get_properties_ins_from_proto(get_properties_msg)
#     # Request properties
#     get_properties_res = client.get_properties(get_properties_ins)
#     # Serialize response
#     get_properties_res_proto = serde.get_properties_res_to_proto(get_properties_res)
#     return ClientMessage(get_properties_res=get_properties_res_proto)


# def _get_parameters(
#     client: Client, get_parameters_msg: ServerMessage.GetParametersIns
# ) -> ClientMessage:
#     # Deserialize get_properties instruction
#     get_parameters_ins = serde.get_parameters_ins_from_proto(get_parameters_msg)
#     # Request parameters
#     get_parameters_res = client.get_parameters(get_parameters_ins)
#     # Serialize response
#     get_parameters_res_proto = serde.get_parameters_res_to_proto(get_parameters_res)
#     return ClientMessage(get_parameters_res=get_parameters_res_proto)


# def _fit(client: Client, fit_msg: ServerMessage.FitIns) -> ClientMessage:
#     # Deserialize fit instruction
#     fit_ins = serde.fit_ins_from_proto(fit_msg)
#     # Perform fit
#     fit_res = client.fit(fit_ins)
#     # Serialize fit result
#     fit_res_proto = serde.fit_res_to_proto(fit_res)
#     return ClientMessage(fit_res=fit_res_proto)


# def _evaluate(client: Client, evaluate_msg: ServerMessage.EvaluateIns) -> ClientMessage:
#     # Deserialize evaluate instruction
#     evaluate_ins = serde.evaluate_ins_from_proto(evaluate_msg)
#     # Perform evaluation
#     evaluate_res = client.evaluate(evaluate_ins)
#     # Serialize evaluate result
#     evaluate_res_proto = serde.evaluate_res_to_proto(evaluate_res)
#     return ClientMessage(evaluate_res=evaluate_res_proto)
