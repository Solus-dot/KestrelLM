import numpy as np
import torch
from torch.utils.data import Dataset

from config import (
    CONTEXT_LENGTH,
    TRAIN_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
)


# Provides fixed-length language-model training examples from a binary token file.
# The underlying file is memory-mapped instead of being loaded completely into RAM.
# Each sample returns X and Y, where Y is X shifted forward by exactly one token.
class BinaryTokenDataset(Dataset):
    def __init__(self, token_file, context_length=CONTEXT_LENGTH):
        self.token_file = token_file
        self.context_length = context_length

        # The preprocessing step stored every token as an unsigned 16-bit integer.
        self.tokens = np.memmap(
            token_file,
            dtype=np.uint16,
            mode="r",
        )

        # Each training example needs T + 1 tokens:
        # T tokens for X and the following T tokens for Y.
        #
        # Samples are spaced T tokens apart so we avoid creating billions
        # of almost-identical overlapping training examples.
        self.num_samples = (
            len(self.tokens) - 1
        ) // self.context_length

    # Returns the number of complete training examples available.
    def __len__(self):
        return self.num_samples

    # Retrieves one contiguous T + 1 token chunk.
    # X contains the first T tokens and Y contains the same sequence shifted by one.
    def __getitem__(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError("Dataset index out of range.")

        start = index * self.context_length
        end = start + self.context_length + 1

        chunk = self.tokens[start:end]

        # Embedding layers expect token indices to use torch.long (int64).
        # torch.tensor also copies the data out of the read-only memory map.
        x = torch.tensor(
            chunk[:-1],
            dtype=torch.long,
        )

        y = torch.tensor(
            chunk[1:],
            dtype=torch.long,
        )

        return x, y


# Loads the preprocessed training token stream.
def create_train_dataset():
    return BinaryTokenDataset(
        TRAIN_TOKENS_PATH
    )


# Loads the preprocessed validation token stream.
def create_validation_dataset():
    return BinaryTokenDataset(
        VALIDATION_TOKENS_PATH
    )


# Performs basic checks on one training example.
# This verifies shapes, data types, and the one-token target shift.
def main():
    train_dataset = create_train_dataset()
    validation_dataset = create_validation_dataset()

    print("Training samples:")
    print(f"{len(train_dataset):,}")

    print("\nValidation samples:")
    print(f"{len(validation_dataset):,}")

    x, y = train_dataset[0]

    print("\nX shape:")
    print(x.shape)

    print("\nY shape:")
    print(y.shape)

    print("\nX dtype:")
    print(x.dtype)

    print("\nY dtype:")
    print(y.dtype)

    print("\nFirst 10 X tokens:")
    print(x[:10])

    print("\nFirst 10 Y tokens:")
    print(y[:10])

    assert x.shape == (CONTEXT_LENGTH,)
    assert y.shape == (CONTEXT_LENGTH,)

    assert x.dtype == torch.long
    assert y.dtype == torch.long

    assert torch.equal(
        x[1:],
        y[:-1],
    )

    print("\nDataset test: PASSED")


if __name__ == "__main__":
    main()