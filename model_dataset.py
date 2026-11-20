from pathlib import Path

import numpy as np

from config import CONFIG_PATH, load_config
from fifteen_puzzle import PuzzleWalkDataset
from solve import load_model


def load_or_generate_model_dataset(config=None, temperature=1.0, model_key="model"):
    config = config or load_config()
    if model_key == "model":
        dataset_path = Path(
            config["dataset"]["model_sampled_save_paths"][str(float(temperature))]
        )
    else:
        dataset_path = Path(config["dataset"][f"{model_key}_sampled_save_path"])
    if not dataset_path.is_absolute():
        dataset_path = CONFIG_PATH.parent / dataset_path

    if dataset_path.exists():
        with np.load(dataset_path) as saved:
            dataset = PuzzleWalkDataset(
                saved["states"],
                saved["moves"],
                saved["path_ids"],
                saved["steps"],
            )
        print(f"Loaded model-sampled dataset: {dataset_path}")
        return dataset

    model = load_model(config, model_key=model_key)
    dataset = PuzzleWalkDataset.generate_from_model(
        model,
        paths=config["walk"]["paths"],
        length=config["walk"]["T"],
        seed=config["seed"],
        temperature=temperature,
    )
    dataset.save(dataset_path)
    print(f"Saved model-sampled dataset: {dataset_path}")
    return dataset
