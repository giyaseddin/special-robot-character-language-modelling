from typing import Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.constants import CHAR2ID, CHAR2ID_WITH_PAD, RANDOM_STATE

from torch.utils.data import DataLoader
from src.data_loader import get_processed_train_valid_test, get_processed_train_test


def _init_fn(worker_id):
    np.random.seed(int(RANDOM_STATE))


class CharSequenceDataset(Dataset):
    """
    Dataset for character sequences and their corresponding labels.

    This dataset is tailored for models dealing with character-level predictions.
    Each sequence in the dataset is a string of characters, paired with a single-character label.
    """

    def __init__(self, sequences, labels):
        """
        Initializes the dataset with sequences and corresponding labels.

        Args:
            sequences (list of strings): Character sequences.
            labels (list of strings): Corresponding labels for each sequence.
        """
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieves a sequence-label pair by index, converting characters to their numeric encodings.

        Args:
            idx (int): Index of the desired sequence.

        Returns:
            tuple: A pair of tensors representing the encoded sequence and its label.
        """
        sequence = torch.tensor([CHAR2ID[char] for char in self.sequences[idx]], dtype=torch.long)
        label = torch.tensor(CHAR2ID[self.labels[idx]], dtype=torch.long)

        return sequence, label


def subseq_collate_fn(batch):
    input_seqs = [item[0] for item in batch]
    target_chars = [item[1] for item in batch]

    # Pad the input sequences
    input_seqs_padded = pad_sequence(input_seqs, batch_first=True, padding_value=CHAR2ID_WITH_PAD['<PAD>'])

    # Convert target characters to a tensor
    target_chars_tensor = torch.tensor(target_chars, dtype=torch.long)

    return input_seqs_padded, target_chars_tensor


class CharSubsequenceDataset(Dataset):
    """
    Dataset for generating character subsequences.

    Useful for models that predict the next character in a sequence. It processes
    a list of sequences into all possible subsequences along with their next character.
    """

    def __init__(self, sequences):
        """
        Initializes the dataset with subsequences derived from the provided sequences.

        Args:
            sequences (list of strings): Original character sequences.
        """
        self.data = self._prepare_subsequences(sequences)

    @staticmethod
    def _prepare_subsequences(sequences):
        """
        Generates subsequences and their next character from the input sequences.

        Args:
            sequences (list of strings): Character sequences.

        Returns:
            list of tuples: Each tuple contains a subsequence and its next character.
        """
        subsequences = []
        for seq in sequences:
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                next_char = seq[i]
                subsequences.append((input_seq, next_char))
        return subsequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a subsequence and its next character by index, converting them to numeric encodings.

        Args:
            idx (int): Index of the desired subsequence.

        Returns:
            tuple: A pair of tensors representing the encoded subsequence and its next character.
        """
        input_seq, next_char = self.data[idx]
        input_encoded = torch.tensor([CHAR2ID_WITH_PAD[char] for char in input_seq], dtype=torch.long)
        next_char_encoded = torch.tensor(CHAR2ID_WITH_PAD[next_char], dtype=torch.long)

        return input_encoded, next_char_encoded


def get_train_valid_test_loaders(batch_size, target_as_sequence=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoader objects for training, validation, and test datasets.

    Args:
        batch_size (int): The batch size for the DataLoader.
        target_as_sequence (bool): Whether the target is a sequence or a single character.

    Returns:
        tuple: A tuple containing DataLoader objects for training, validation, and test datasets.
    """
    # Load the processed datasets
    train_X, train_y, val_X, val_y, test_X, test_y = get_processed_train_valid_test(
        target_as_sequence=target_as_sequence
    )

    # Create datasets
    if target_as_sequence:
        train_nextchar_dataset = CharSubsequenceDataset(train_X.to_list())
        val_nextchar_dataset = CharSubsequenceDataset(val_X.to_list())
        test_nextchar_dataset = CharSubsequenceDataset(test_X.to_list())

        train_nextchar_loader = DataLoader(train_nextchar_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=subseq_collate_fn, worker_init_fn=_init_fn)
        val_nextchar_loader = DataLoader(val_nextchar_dataset, batch_size=batch_size, shuffle=True,
                                         collate_fn=subseq_collate_fn, worker_init_fn=_init_fn)
        test_nextchar_loader = DataLoader(test_nextchar_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=subseq_collate_fn, worker_init_fn=_init_fn)

        return train_nextchar_loader, val_nextchar_loader, test_nextchar_loader

    else:
        train_dataset = CharSequenceDataset(train_X.to_list(), train_y.to_list())
        val_dataset = CharSequenceDataset(val_X.to_list(), val_y.to_list())
        test_dataset = CharSequenceDataset(test_X.to_list(), test_y.to_list())

        # Create DataLoader objects
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn)

        return train_loader, val_loader, test_loader


def get_full_train_test_loaders(batch_size, target_as_sequence=False) -> Tuple[DataLoader, DataLoader]:
    """
    Creates DataLoader objects for full training and test datasets.

    Args:
        batch_size (int): The batch size for the DataLoader.
        target_as_sequence (bool): Whether the target is a sequence or a single character.

    Returns:
        tuple: A tuple containing DataLoader objects for full training and test datasets.
    """
    # Load the processed datasets
    train_X, train_y, test_X, test_y = get_processed_train_test(target_as_sequence=target_as_sequence)

    # Create full training dataset
    full_train_dataset = CharSequenceDataset(train_X.append(test_X), train_y.append(test_y))

    # Create DataLoader objects
    full_train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn)
    test_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=_init_fn)

    return full_train_loader, test_loader
