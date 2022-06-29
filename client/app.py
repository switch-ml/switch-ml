import grpc
import sys
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from client.client_manager import SwitchMlClient
from client.config import fit_config, evaluate_config, device_config, get_grpc_options
from client.parameter import parameters_to_weights, weights_to_parameters
from client.utils import load_data, load_efficientnet, get_model_params, get_dry_sets


import client.proto.service_pb2 as service_pb2
import client.proto.service_pb2_grpc as service_pb2_grpc


def send_weights(stub, train_results, eval_results):
    req = service_pb2.SendWeightsRequest(fit_res=train_results, eval_res=eval_results)
    stub.SendWeights(req)
    print("SENT WEIGHTS")


def fetch_weights(stub):
    req = service_pb2.FetchWeightsRequest()
    response = stub.FetchWeights(req)
    print("RECIEVED WEIGHTS")
    return response.parameters


def start_process(weights, dry_test=False):
    train_config = fit_config()

    test_config = evaluate_config()

    device = device_config()

    print("LOADING MODEL....")

    model = load_efficientnet(classes=10)

    if dry_test:
        trainset, testset, _ = load_data()
    else:
        trainset, testset = get_dry_sets()

    client = SwitchMlClient(trainset, testset, device)

    trained_weights, train_count, train_metrics = client.fit(weights, train_config)

    eval_loss, eval_count, eval_metrics = client.evaluate(
        get_model_params(model), test_config
    )

    print("Train Count: ", train_count)
    print("Train Results: ", train_metrics)

    print("Eval Count: ", eval_count)
    print("Eval Results: ", eval_metrics)
    print("Eval Loss: ", eval_loss)

    params = weights_to_parameters(trained_weights)

    params = service_pb2.Parameters(
        tensors=params.tensors, tensor_type=params.tensor_type
    )

    fit_res = service_pb2.FitRes(
        parameters=params, num_examples=train_count, metrics=train_metrics
    )

    eval_res = service_pb2.EvaluateRes(
        loss=eval_loss, num_examples=eval_count, metrics=eval_metrics
    )

    return fit_res, eval_res


def run():
    with grpc.insecure_channel(
        "localhost:4000",
        options=get_grpc_options(),
    ) as channel:

        print("CONNECTED TO SERVER")

        stub = service_pb2_grpc.SwitchmlWeightsServiceStub(channel)

        print("REQUESTING WEIGHTS")

        # SwitchMl Weights
        parameters = fetch_weights(stub)

        weights = parameters_to_weights(parameters)

        fit_res, eval_res = start_process(weights)

        send_weights(stub, fit_res, eval_res)


if __name__ == "__main__":
    run()
