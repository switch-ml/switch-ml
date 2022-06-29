import torch


def get_grpc_options():
    return [
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
    ]


def fit_config():
    """
    Conditions to train client model on client data
    """
    config = {
        "batch_size": 1,
        "local_epochs": 1,
    }
    return config


def evaluate_config():
    """
    Conditions to test client model on client data
    """
    val_steps = 5
    return {"val_steps": val_steps}


def device_config():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device