from io import BytesIO
import json
import ast
from typing import cast

import numpy as np

from functools import reduce

from ml.parameter import parameters_to_weights, weights_to_parameters


def aggregate(results):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    count_total = sum([count for _, count in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * count for layer in weights] for weights, count in results
    ]

    weights_prime = []

    for layer_updates in zip(*weighted_weights):
        val = reduce(np.add, layer_updates)
        weights_prime.append(val / count_total)

    # Compute average weights of each layer
    return weights_prime


def metrics_average(results):

    train_accs = []
    train_loss = []
    val_loss = []
    val_accs = []
    counts = []

    for client in results:
        metrics = client[0]
        count = client[1]

        counts.append(count)
        train_accs.append(metrics.get("train_accuracy", 0))
        train_loss.append(metrics.get("train_loss", 0))
        val_loss.append(metrics.get("val_loss", 0))
        val_accs.append(metrics.get("val_accuracy", 0))

    counts_total = len(results)

    return {
        "train_loss": sum(train_loss) / counts_total,
        "train_accuracy": sum(train_accs) / counts_total,
        "val_loss": sum(val_loss) / counts_total,
        "val_accuracy": sum(val_accs) / counts_total,
        "counts": sum(counts) / len(counts),
    }


def federated_average(clients_data, fit_metrics_aggregation_fn=None):

    weights_results = []
    client_metrics = []
    metrics_aggregated = {}

    # Iteratign clients weights
    for client in clients_data:

        weights = client.get("weights")

        count = client.get("num_examples")

        train_metrics = client.get("metrics")

        weights_results.append((weights, count))

        client_metrics.append((train_metrics, count))

    # Aggregating all weights
    agg_weights = aggregate(weights_results)

    # Aggregate custom metrics if aggregation fn was provided
    if fit_metrics_aggregation_fn:
        print("AGGREGATING EVALUATION METRICS")
        metrics_aggregated = fit_metrics_aggregation_fn(client_metrics)

    return agg_weights, metrics_aggregated
