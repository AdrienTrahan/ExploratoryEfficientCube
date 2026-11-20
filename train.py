from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG_PATH, load_config, resolve_device
from fifteen_puzzle import INVERSE_MOVES, PuzzleWalkDataset, successor_batch
from model import Jepa15PuzzleModel
from sigreg import sigreg_strong_loss, sigreg_weak_loss


def train(config):
    seed = config["seed"]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = resolve_device(config["device"])

    save_path = config["dataset"]["save_path"]
    dataset_path = Path(save_path) if save_path else None
    if dataset_path is not None and not dataset_path.is_absolute():
        dataset_path = CONFIG_PATH.parent / dataset_path

    if dataset_path is not None and dataset_path.exists():
        with np.load(dataset_path) as saved:
            dataset = PuzzleWalkDataset(
                saved["states"],
                saved["moves"],
                saved["path_ids"],
                saved["steps"],
            )
        print(f"Loaded dataset: {dataset_path}")
    else:
        dataset = PuzzleWalkDataset.generate(
            paths=config["walk"]["paths"],
            length=config["walk"]["T"],
            seed=seed,
        )
        if dataset_path is not None:
            dataset.save(dataset_path)
            print(f"Saved dataset: {dataset_path}")

    model = Jepa15PuzzleModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    writer = SummaryWriter()
    path_starts = np.concatenate(
        ([0], np.flatnonzero(dataset.path_ids[1:] != dataset.path_ids[:-1]) + 1)
    )
    path_stops = np.concatenate((path_starts[1:], [len(dataset)]))
    path_order = rng.permutation(len(path_starts))
    ordered_indices = np.concatenate(
        [np.arange(path_starts[index], path_stops[index]) for index in path_order]
    )
    path_count = len(path_order)
    path_ends = np.cumsum(path_stops[path_order] - path_starts[path_order])
    batch_size = config["training"]["batch_size"]
    last_progress = 0
    print(f"Training paths: 0/{path_count} (0.0%)", flush=True)

    for start in range(0, len(ordered_indices), batch_size):
        batch_indices = ordered_indices[start : start + batch_size]
        # sigreg_indices = dataset.sample_indices(config["sigreg"]["batch_size"], rng)
        states = dataset.states[batch_indices]
        successors, state_indices, move_indices = successor_batch(states)
        successor_states = torch.as_tensor(successors.astype(np.int64), device=device)
        state_indices = torch.as_tensor(state_indices, device=device)
        move_indices = torch.as_tensor(move_indices, device=device)
        targets = torch.as_tensor(
            np.asarray(INVERSE_MOVES)[dataset.moves[batch_indices]],
            dtype=torch.long,
            device=device,
        )
        # sigreg_states = torch.as_tensor(
        #     dataset.states[sigreg_indices].astype(np.int64), device=device
        # )

        successor_scores, _ = model(successor_states)
        logits = torch.full((len(batch_indices), 4), -torch.inf, device=device)
        logits[state_indices, move_indices] = successor_scores
        puzzle_loss = F.cross_entropy(logits, targets)
        # sigreg_loss = sigreg_strong_loss(
        #     model.encoder(sigreg_states),
        # )
        total_loss = puzzle_loss # + config["sigreg"]["weight"] * sigreg_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        processed_states = start + len(batch_indices)
        processed_paths = np.searchsorted(path_ends, processed_states, side="right")
        progress = processed_states * 1000 // len(dataset)
        if progress > last_progress:
            print(
                f"\rTraining paths: {processed_paths}/{path_count} ({progress / 10:.1f}%)",
                end="",
                flush=True,
            )
            last_progress = progress

        writer.add_scalar("loss/total", total_loss.item(), processed_states)
        writer.add_scalar("loss/efficient_puzzle", puzzle_loss.item(), processed_states)
        # writer.add_scalar("loss/sigreg", sigreg_loss.item(), processed_states)

    print(f"\rTraining paths: {path_count}/{path_count} (100%)")
    writer.flush()
    print(f"TensorBoard run: {writer.log_dir}")
    writer.close()

    checkpoint_path = Path(config["model"]["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = CONFIG_PATH.parent / checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint: {checkpoint_path}")
    return model, dataset


def main():
    train(load_config())


if __name__ == "__main__":
    main()
