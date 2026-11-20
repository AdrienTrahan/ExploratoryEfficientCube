from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fifteen_puzzle import MOVE_NAMES, apply_move, legal_moves, solved_state
from config import CONFIG_PATH, load_config, resolve_device
from model import FifteenPuzzleModel


@dataclass
class BeamNode:
    state: np.ndarray
    path: tuple[int, ...]
    score: float
    visited: frozenset[bytes]


def normalize_state(state):
    state = np.asarray(state)
    if state.shape == (1, 16):
        state = state[0]
    if state.shape == (256,):
        state = state.reshape(16, 16).argmax(axis=1)
    if state.shape == (16, 16):
        state = state.argmax(axis=1)
    if state.shape != (16,):
        raise ValueError("state must contain 16 tile values or a 256-value one-hot vector")
    if not np.array_equal(np.sort(state), np.arange(16)):
        raise ValueError("state must contain each tile from 0 to 15 exactly once")
    return state.astype(np.uint8)


def load_model(config, model_key="model"):
    device = resolve_device(config["device"])
    model = FifteenPuzzleModel().to(device)
    checkpoint_path = Path(config[model_key]["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = CONFIG_PATH.parent / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    return model


def beam_search(state, model, beam_width, max_depth):
    initial_state = normalize_state(state)
    goal = solved_state()
    if np.array_equal(initial_state, goal):
        return []
    if beam_width < 1 or max_depth < 1:
        raise ValueError("beam_width and max_depth must be positive")

    device = next(model.parameters()).device
    model.eval()
    initial_key = initial_state.tobytes()
    beam = [BeamNode(initial_state, (), 0.0, frozenset((initial_key,)))]

    with torch.inference_mode():
        for _ in range(max_depth):
            successor_states = []
            successor_moves = []
            successors_by_parent = []
            for node in beam:
                parent_successors = []
                for move in legal_moves(node.state):
                    parent_successors.append(len(successor_states))
                    successor_states.append(apply_move(node.state, move))
                    successor_moves.append(move)
                successors_by_parent.append(parent_successors)

            successor_tensor = torch.as_tensor(
                np.stack(successor_states), dtype=torch.long, device=device
            )
            successor_logits = model(successor_tensor)
            best_by_state = {}
            best_solution = None

            for node, parent_successors in zip(beam, successors_by_parent):
                probabilities = torch.softmax(
                    -successor_logits[parent_successors], dim=0
                ).cpu().numpy()
                for successor_index, probability in zip(
                    parent_successors, probabilities
                ):
                    move = successor_moves[successor_index]
                    next_state = successor_states[successor_index]
                    score = node.score + float(np.log(max(float(probability), 1e-12)))
                    if np.array_equal(next_state, goal):
                        if best_solution is None or score > best_solution[0]:
                            best_solution = (score, (*node.path, move))
                        continue

                    state_key = next_state.tobytes()
                    if state_key in node.visited:
                        continue
                    candidate = BeamNode(
                        next_state,
                        (*node.path, move),
                        score,
                        node.visited | {state_key},
                    )
                    previous = best_by_state.get(state_key)
                    if previous is None or candidate.score > previous.score:
                        best_by_state[state_key] = candidate

            if best_solution is not None:
                return [MOVE_NAMES[move] for move in best_solution[1]]
            if not best_by_state:
                break
            beam = sorted(
                best_by_state.values(), key=lambda node: node.score, reverse=True
            )[:beam_width]

    raise RuntimeError(f"No solution found within {max_depth} moves")


def solve_state(state, model=None, config=None):
    config = config or load_config()
    if model is None:
        model = load_model(config)
    return beam_search(
        state,
        model,
        beam_width=config["solver"]["beam_width"],
        max_depth=config["solver"]["max_depth"],
    )


def print_state(state):
    for row in state.reshape(4, 4):
        print(" ".join(f"{int(tile):2d}" for tile in row))


def print_solution_trace(initial_state, path):
    state = normalize_state(initial_state)
    print("Start state:")
    print_state(state)
    for step, action in enumerate(path, start=1):
        move = MOVE_NAMES.index(action)
        state = apply_move(state, move)
        print(f"\nStep {step}: blank moves {action}")
        print_state(state)
    print("\nSolved" if np.array_equal(state, solved_state()) else "Path ended")


def main():
    config = load_config()
    if config["solver"]["state"] is None:
        raise ValueError("Set solver.state in config.yaml before running solve.py")
    path = solve_state(config["solver"]["state"], config=config)
    print_solution_trace(config["solver"]["state"], path)


if __name__ == "__main__":
    main()
