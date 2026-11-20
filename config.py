from pathlib import Path

import torch
import yaml


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path=CONFIG_PATH):
    with path.open() as stream:
        return yaml.safe_load(stream)


def configure_self_training(config):
    config = {**config, "dataset": dict(config["dataset"])}
    self_training = config["self_training"]
    for number in range(2, self_training["iterations"] + 2):
        model_key = f"model{number}"
        config[model_key] = {
            "checkpoint_path": self_training["checkpoint_path_template"].format(
                number=number
            )
        }
        config["dataset"][f"{model_key}_sampled_save_path"] = (
            self_training["dataset_path_template"].format(number=number)
        )
    return config


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
