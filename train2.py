from config import configure_self_training, load_config
from model_dataset import load_or_generate_model_dataset
from train import train_on_dataset


def train2(config):
    config = configure_self_training(config)
    iterations = config["self_training"]["iterations"]
    temperature = config["self_training"]["temperature"]
    previous_model = "model"
    result = None

    for iteration, number in enumerate(range(2, iterations + 2), start=1):
        model_key = f"model{number}"
        print(
            f"\nIteration {iteration}/{iterations}: training {model_key} "
            f"from {previous_model} walks",
            flush=True,
        )
        dataset = load_or_generate_model_dataset(
            config, temperature=temperature, model_key=previous_model
        )
        result = train_on_dataset(
            config,
            dataset,
            model_key=model_key,
            initial_model_key=previous_model,
        )
        previous_model = model_key

    return result


def main():
    train2(load_config())


if __name__ == "__main__":
    main()
