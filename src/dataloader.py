import torch
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
)
from dataset import (
    create_train_dataset,
    create_validation_dataset,
)

# Creates a PyTorch DataLoader for a token dataset.
# The DataLoader groups individual (T,) samples into batches of shape (B, T).
# num_workers=0 keeps loading simple and reliable while we are developing.
def create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )

# Creates the training DataLoader.
# Training samples are shuffled so batches do not always appear in file order.
def create_train_dataloader():
    dataset = create_train_dataset()

    return create_dataloader(
        dataset,
        shuffle=True,
    )


# Creates the validation DataLoader.
# Validation does not need shuffling because no parameter updates happen there.
def create_validation_dataloader():
    dataset = create_validation_dataset()

    return create_dataloader(
        dataset,
        shuffle=False,
    )

# Loads one training batch and verifies its shape and target alignment.
def main():
    train_loader = create_train_dataloader()

    x, y = next(
        iter(train_loader)
    )

    print("X shape:")
    print(x.shape)

    print("\nY shape:")
    print(y.shape)

    print("\nX dtype:")
    print(x.dtype)

    print("\nY dtype:")
    print(y.dtype)

    print("\nFirst 10 tokens of first X sequence:")
    print(x[0, :10])

    print("\nFirst 10 tokens of first Y sequence:")
    print(y[0, :10])

    assert x.shape == (
        BATCH_SIZE,
        CONTEXT_LENGTH,
    )

    assert y.shape == (
        BATCH_SIZE,
        CONTEXT_LENGTH,
    )

    assert x.dtype == torch.long
    assert y.dtype == torch.long

    assert torch.equal(
        x[:, 1:],
        y[:, :-1],
    )

    print("\nDataLoader test: PASSED")

if __name__ == "__main__":
    main()