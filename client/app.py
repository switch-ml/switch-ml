import grpc
import sys
import os
import random


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from client.client_manager import SwitchMlClient
from client.config import fit_config, evaluate_config, device_config, get_grpc_options
from client.parameter import parameters_to_weights, weights_to_parameters
from client.utils import load_data, get_dry_sets, load_partition


import client.proto.service_pb2 as service_pb2
import client.proto.service_pb2_grpc as service_pb2_grpc


def send_weights(stub, train_results, client, round):
    req = service_pb2.SendWeightsRequest(
        fit_res=train_results, round=round, client_id=client.client_id
    )
    print(f"SENDING round-{round} TRAINED WEIGHTS TO SERVER")

    res = stub.SendWeights(req)

    print(f"RECEIVED round-{round} AGGREGATED WEIGHTS FROM SERVER")

    weights = parameters_to_weights(res.parameters)

    return start_evaluation(client, weights)


def fetch_weights(stub):
    req = service_pb2.FetchWeightsRequest()

    response = stub.FetchWeights(req)

    print("RECEIVED INITIAL WEIGHTS")

    return response.parameters


def start_evaluation(client, weights):
    test_config = evaluate_config()

    eval_loss, eval_count, eval_metrics = client.evaluate(weights, test_config)

    print("Eval Results: ", eval_metrics)
    print("Eval Loss: ", eval_loss)

    return weights


def start_training(weights, client):
    train_config = fit_config()

    trained_weights, train_count, train_metrics = client.fit(weights, train_config)

    params = weights_to_parameters(trained_weights)

    params = service_pb2.Parameters(
        tensors=params.tensors, tensor_type=params.tensor_type
    )

    fit_res = service_pb2.FitRes(
        parameters=params, num_examples=train_count, metrics=train_metrics
    )

    return fit_res


def run(index):
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

        dry_test = False  # code changed

        if dry_test:
            print("RUNNING DRY TEST")
            trainset, testset = get_dry_sets()
        else:
            trainset, testset = load_partition(index)

        device = device_config()

        client = SwitchMlClient(trainset, testset, device)

        for i in range(1, 6):
            print("ROUND: ", i)

            fit_res = start_training(weights, client)

            weights = send_weights(stub, fit_res, client, i)

            print(f"ROUND {i} COMPLETED \n")


if __name__ == "__main__":
    i = random.randint(0, 10)
    run(i)
