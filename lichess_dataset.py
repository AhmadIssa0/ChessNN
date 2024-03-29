
import json
import torch
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import os
from dataclasses import dataclass, field
from typing import List
import mmap


FEN_CHAR_TO_INDEX = {
    'eval': 0,
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12,
    '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21,
    '/': 22, '.': 23, ' ': 24, 'w': 25, 'W': 26,
    '-': 27, 'd': 28, 'c': 29, 'f': 30, 'g': 31, 'h': 32, 'a': 33, 'e': 34, 'F': 35, 'E': 36
}
MAX_FEN_LENGTH = 90


@dataclass
class RawIO:
    expanded_fen: str
    cp_eval: int
    contains_eval: bool
    mate: int  # number of moves till mate, if negative then black mates white. If 0 no mate was found


def remove_full_half_moves_from_fen(fen):
    parts = fen.split(' ')
    if len(parts) == 6:
        return ' '.join(parts[:-2])
    else:
        return fen


def expand_fen_string(fen):
    # Split the FEN string into its components
    parts = fen.split(' ')
    board, rest = parts[0], parts[1:]

    # Expand the first component
    expanded_board = ""
    for char in board:
        if char.isdigit():
            # Replace digit with corresponding number of dots
            expanded_board += '.' * int(char)
        else:
            # Keep non-digit characters as they are
            expanded_board += char

    # Reassemble the FEN string
    expanded_fen = ' '.join([expanded_board] + rest)
    return remove_full_half_moves_from_fen(expanded_fen)


class JSONLinesChessDataset(Dataset):
    """The lichess evaluation dataset."""
    def __init__(self, file_path, index_file='index.pkl'):
        self.file_path = file_path
        self.index_file = index_file
        self.offsets = self._load_or_build_index()
        self._map_file()

    def _map_file(self):
        # Open the file and memory-map it
        with open(self.file_path, 'r+b') as f:  # Open the file in read-binary mode for mmap
            self.data = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

    def _load_or_build_index(self):
        # Check if the index file exists; load it if it does, or build it if it doesn't
        if os.path.exists(self.index_file):
            print('Loading index from file.')
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        else:
            print('Creating dataset index file.')
            offsets = [0]  # Start from the beginning of the file
            with open(self.file_path, 'rb') as file:
                # Setup tqdm for manual updates
                pbar = tqdm(desc="Indexing", unit="lines")
                while line := file.readline():
                    offsets.append(file.tell())
                    pbar.update(1)  # Manual update by 1 for each line processed
                pbar.close()
            with open(self.index_file, 'wb') as f:
                pickle.dump(offsets, f)
                print(f'Index file saved to: {self.index_file}')
            return offsets

    def __len__(self):
        # The total number of items is the number of offsets minus one (for the end of the file)
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        # Ensure the index is within bounds
        if idx >= len(self) or idx < 0:
            raise IndexError('Index out of bounds')

        # Calculate the start and end positions for the requested item
        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        # Directly slice the mmap object to get the bytes for the desired line
        line_bytes = self.data[start:end]

        # Decode the bytes to a string, strip newline characters, and parse JSON
        line = line_bytes.decode('utf-8').strip()
        json_object = json.loads(line)

        fen_str = json_object['fen']
        expanded_fen_str = expand_fen_string(fen_str)
        pv = json_object['evals'][0]['pvs'][0]
        if 'cp' in pv:
            centipawn_eval = pv['cp']
            contains_eval = True
        else:
            centipawn_eval = 100 * 100 if pv['mate'] > 0 else -100 * 100
            contains_eval = False

        return RawIO(
            expanded_fen=expanded_fen_str,
            cp_eval=centipawn_eval,
            contains_eval=contains_eval,
            mate=pv.get('mate', 0),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the mmap object from the state
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the mmap object
        self._map_file()


class JSONLinesLichessDataset(Dataset):
    """Dataset of lichess evaluations extracted from lichess games databases."""

    def __init__(self, file_path, index_file=None):
        self.file_path = file_path
        if index_file is None:
            base_name, _ = os.path.splitext(file_path)
            self.index_file = f'{base_name}_index.pkl'
        else:
            self.index_file = index_file
        self.offsets = self._load_or_build_index()
        self._map_file()

    def _map_file(self):
        # Open the file and memory-map it
        with open(self.file_path, 'r+b') as f:  # Open the file in read-binary mode for mmap
            self.data = mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ)

    def _load_or_build_index(self):
        # Check if the index file exists; load it if it does, or build it if it doesn't
        if os.path.exists(self.index_file):
            print('Loading index from file.')
            with open(self.index_file, 'rb') as f:
                return pickle.load(f)
        else:
            print('Creating dataset index file.')
            offsets = [0]  # Start from the beginning of the file
            with open(self.file_path, 'rb') as file:
                # Setup tqdm for manual updates
                pbar = tqdm(desc="Indexing", unit="lines")
                while line := file.readline():
                    offsets.append(file.tell())
                    pbar.update(1)  # Manual update by 1 for each line processed
                pbar.close()
            with open(self.index_file, 'wb') as f:
                pickle.dump(offsets, f)
                print(f'Index file saved to: {self.index_file}')
            return offsets

    def __len__(self):
        # The total number of items is the number of offsets minus one (for the end of the file)
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        # Ensure the index is within bounds
        if idx >= len(self) or idx < 0:
            raise IndexError('Index out of bounds')

        # Calculate the start and end positions for the requested item
        start = self.offsets[idx]
        end = self.offsets[idx + 1]

        # Directly slice the mmap object to get the bytes for the desired line
        line_bytes = self.data[start:end]

        # Decode the bytes to a string, strip newline characters, and parse JSON
        line = line_bytes.decode('utf-8').strip()
        json_object = json.loads(line)

        fen_str = json_object['FEN']
        expanded_fen_str = expand_fen_string(fen_str)

        if 'cp' in json_object:
            centipawn_eval = json_object['cp']
            contains_eval = True
        else:
            centipawn_eval = 100 * 100 if json_object['mate'] > 0 else -100 * 100
            contains_eval = False

        return RawIO(
            expanded_fen=expanded_fen_str,
            cp_eval=centipawn_eval,
            contains_eval=contains_eval,
            mate=json_object.get('mate', 0),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the mmap object from the state
        del state['data']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize the mmap object
        self._map_file()


# class JSONLinesChessDataset(Dataset):
#     def __init__(self, file_path, index_file='index.pkl'):
#         self.file_path = file_path
#         self.index_file = index_file
#
#         if os.path.exists(self.index_file):
#             self.offsets = self._load_index()
#         else:
#             self.offsets = self._build_index()
#
#     def _build_index(self):
#         print('Creating dataset index file.')
#         offsets = []
#         with open(self.file_path, 'rb') as file:
#             offset = 0
#             for line in tqdm(file, desc="Indexing"):
#                 offsets.append(offset)
#                 offset += len(line)
#         # Save the index to a file for future use
#         with open(self.index_file, 'wb') as f:
#             pickle.dump(offsets, f)
#             print(f'Index file saved to: {self.index_file}')
#         return offsets
#
#     def _load_index(self):
#         # Load the index from a file
#         with open(self.index_file, 'rb') as f:
#             offsets = pickle.load(f)
#         return offsets
#
#     def __len__(self):
#         return len(self.offsets)
#
    # def __getitem__(self, idx):
    #     with open(self.file_path, 'rb') as file:
    #         file.seek(self.offsets[idx])
    #         line = file.readline()
    #         json_object = json.loads(line)
    #
    #         fen_str = json_object['fen']
    #         expanded_fen_str = expand_fen_string(fen_str)
    #         pv = json_object['evals'][0]['pvs'][0]
    #         if 'cp' in pv:
    #             centipawn_eval = pv['cp']
    #             contains_eval = True
    #         else:
    #             centipawn_eval = 30 * 100 if pv['mate'] > 0 else -30 * 100
    #             contains_eval = False
    #
    #     return RawIO(
    #         expanded_fen=expanded_fen_str,
    #         cp_eval=centipawn_eval,
    #         contains_eval=contains_eval,
    #         mate=pv.get('mate', 0),
    #     )


@dataclass
class TransformerIO:
    embedding_indices: torch.IntTensor  # (B, Length of FEN + 1). We pad FEN strings to 85 characters
    cp_evals: torch.LongTensor  # (B)
    cp_valid: torch.BoolTensor  # (B), false if cp_eval was missing (because mate was found)
    expanded_fens: List[str]
    white_win_prob: torch.Tensor = field(init=False)  # (B)

    def __post_init__(self):
        # For conversion of centipawns to win percentage see https://lichess.org/page/accuracy
        # Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * centipawns)) - 1)
        self.white_win_prob = 1.0 / (1.0 + torch.exp(-0.00368208 * self.cp_evals))

    def to(self, device):
        return TransformerIO(
            embedding_indices=self.embedding_indices.to(device=device),
            cp_evals=self.cp_evals.to(device=device),
            cp_valid=self.cp_valid.to(device=device),
            expanded_fens=self.expanded_fens
        )


def collate_fn(batch: List[RawIO]) -> TransformerIO:
    """Batch is a list of dicts with key"""
    emb_indices_batch = []
    # Use a single token for the eval
    for batch_elt in batch:
        # print(batch_elt.expanded_fen)
        try:
            indices_lst = [FEN_CHAR_TO_INDEX['eval']] + [FEN_CHAR_TO_INDEX[c] for c in batch_elt.expanded_fen]
        except KeyError:
            print(f'Failed to tokenize: {batch_elt.expanded_fen}')
            exit(1)
        # Pad to 85 characters
        indices_lst += [FEN_CHAR_TO_INDEX[' '] for _ in range(MAX_FEN_LENGTH - len(batch_elt.expanded_fen))]
        emb_indices_batch.append(
            torch.tensor(indices_lst, dtype=torch.long)
        )

    eval_batch = torch.tensor([x.cp_eval for x in batch], dtype=torch.long)
    cp_valid = torch.tensor([x.contains_eval for x in batch], dtype=torch.bool)
    embedding_indices = torch.stack(emb_indices_batch, dim=0)
    return TransformerIO(
        embedding_indices=embedding_indices,
        cp_evals=eval_batch,
        cp_valid=cp_valid,
        expanded_fens=[x.expanded_fen for x in batch],
    )