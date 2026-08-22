from datasets import load_dataset

from config import (
    DATA_DIR,
    HF_CACHE_DIR,
    TINYSTORIES_DIR,
)


DATASET_NAME = "roneneldan/TinyStories"


# Downloads TinyStories once and stores the processed dataset
# directly inside the KestrelLM data directory.
def download_dataset():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading TinyStories...")

    dataset = load_dataset(
        DATASET_NAME,
        cache_dir=str(HF_CACHE_DIR),
    )

    dataset.save_to_disk(
        str(TINYSTORIES_DIR)
    )

    print()
    print("TinyStories saved locally.")
    print(f"Path: {TINYSTORIES_DIR}")


if __name__ == "__main__":
    download_dataset()