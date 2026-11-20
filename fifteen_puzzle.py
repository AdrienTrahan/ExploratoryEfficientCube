from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MOVE_NAMES = ("U", "D", "L", "R")
INVERSE_MOVES = (1, 0, 3, 2)
MOVE_OFFSETS = (-4, 4, -1, 1)


def solved_state():
    return np.asarray((*range(1, 16), 0), dtype=np.uint8)


def legal_moves(state):
    blank = int(np.flatnonzero(state == 0)[0])
    row, column = divmod(blank, 4)
    moves = []
    if row > 0:
        moves.append(0)
    if row < 3:
        moves.append(1)
    if column > 0:
        moves.append(2)
    if column < 3:
        moves.append(3)
    return moves


def apply_move(state, move):
    if move not in range(len(MOVE_NAMES)):
        raise ValueError("move must be one of the four blank directions")
    if move not in legal_moves(state):
        raise ValueError(f"blank cannot move {MOVE_NAMES[move]}")
    blank = int(np.flatnonzero(state == 0)[0])
    target = blank + MOVE_OFFSETS[move]
    result = np.asarray(state).copy()
    result[blank], result[target] = result[target], result[blank]
    return result


def successor_batch(states):
    successors = []
    state_indices = []
    move_indices = []
    for state_index, state in enumerate(states):
        for move in legal_moves(state):
            successors.append(apply_move(state, move))
            state_indices.append(state_index)
            move_indices.append(move)
    return (
        np.stack(successors),
        np.asarray(state_indices, dtype=np.int64),
        np.asarray(move_indices, dtype=np.int64),
    )


def uniform_move_sampler(rng, state, step, previous_move=None):
    moves = legal_moves(state)
    if previous_move is not None:
        moves = [move for move in moves if move != INVERSE_MOVES[previous_move]]
    return int(rng.choice(moves))


def random_walk(
    initial_state,
    length,
    move_sampler=uniform_move_sampler,
    transition=apply_move,
    rng=None,
):
    if length < 1:
        raise ValueError("length must be positive")
    if rng is None:
        rng = np.random.default_rng()
    state = np.asarray(initial_state).copy()
    states = []
    moves = []
    previous_move = None
    for step in range(length):
        move = int(move_sampler(rng, state.copy(), step, previous_move))
        state = np.asarray(transition(state, move)).copy()
        states.append(state.copy())
        moves.append(move)
        previous_move = move
    return np.stack(states), np.asarray(moves, dtype=np.int64)


class PuzzleWalkDataset(Dataset):
    def __init__(self, states, moves, path_ids, steps):
        self.states = np.asarray(states, dtype=np.uint8)
        self.moves = np.asarray(moves, dtype=np.int64)
        self.path_ids = np.asarray(path_ids, dtype=np.int64)
        self.steps = np.asarray(steps, dtype=np.int64)

    @classmethod
    def generate(cls, paths, length, seed, move_sampler=uniform_move_sampler):
        if paths < 1:
            raise ValueError("paths must be positive")
        rng = np.random.default_rng(seed)
        walks = []
        last_percent = -1
        for path_id in range(paths):
            walks.append(
                random_walk(
                    solved_state(),
                    length,
                    move_sampler=move_sampler,
                    rng=rng,
                )
            )
            percent = (path_id + 1) * 100 // paths
            if percent > last_percent:
                print(
                    f"\rGenerating paths: {path_id + 1}/{paths} ({percent}%)",
                    end="",
                    flush=True,
                )
                last_percent = percent
        print()
        states = np.concatenate([walk[0] for walk in walks])
        moves = np.concatenate([walk[1] for walk in walks])
        path_ids = np.repeat(np.arange(paths), length)
        steps = np.tile(np.arange(1, length + 1), paths)
        return cls(states, moves, path_ids, steps)

    @classmethod
    def generate_from_model(cls, model, paths, length, seed, temperature=0.5):
        if paths < 1:
            raise ValueError("paths must be positive")
        if length < 1:
            raise ValueError("length must be positive")
        rng = np.random.default_rng(seed)
        device = next(model.parameters()).device
        states = np.tile(solved_state(), (paths, 1))
        path_states = np.empty((paths, length, 16), dtype=np.uint8)
        path_moves = np.empty((paths, length), dtype=np.int64)
        last_percent = -1
        model.eval()
        with torch.inference_mode():
            for step in range(length):
                successors, state_indices, move_indices = successor_batch(states)
                successor_tensor = torch.as_tensor(
                    successors, dtype=torch.long, device=device
                )
                state_index_tensor = torch.as_tensor(
                    state_indices, device=device
                )
                move_index_tensor = torch.as_tensor(
                    move_indices, device=device
                )
                successor_logits = model(successor_tensor)
                logits = torch.full(
                    (paths, len(MOVE_NAMES)), -torch.inf, device=device
                )
                logits[state_index_tensor, move_index_tensor] = successor_logits
                probabilities = torch.softmax(logits / temperature, dim=1).cpu().numpy()
                cumulative = np.cumsum(probabilities, axis=1)
                cumulative[:, -1] = 1
                moves = (rng.random(paths)[:, None] >= cumulative).sum(axis=1)

                rows = np.arange(paths)
                blanks = np.argmax(states == 0, axis=1)
                destinations = blanks + np.asarray(MOVE_OFFSETS)[moves]
                next_states = states.copy()
                next_states[rows, blanks] = states[rows, destinations]
                next_states[rows, destinations] = 0
                path_states[:, step] = next_states
                path_moves[:, step] = moves
                states = next_states

                percent = (step + 1) * 100 // length
                if percent > last_percent:
                    print(
                        f"\rGenerating model paths: {step + 1}/{length} ({percent}%)",
                        end="",
                        flush=True,
                    )
                    last_percent = percent
        print()
        path_ids = np.repeat(np.arange(paths), length)
        steps = np.tile(np.arange(1, length + 1), paths)
        return cls(
            path_states.reshape(-1, 16),
            path_moves.reshape(-1),
            path_ids,
            steps,
        )

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, index):
        return self.states[index], self.moves[index]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                states=self.states,
                moves=self.moves,
                path_ids=self.path_ids,
                steps=self.steps,
            )
