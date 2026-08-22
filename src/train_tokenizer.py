from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from config import (
    SPECIAL_TOKENS,
    TOKENIZER_DIR,
    TOKENIZER_PATH,
    UNK_TOKEN,
    VOCAB_SIZE,
)


DATASET_NAME = "roneneldan/TinyStories"
DATASET_SPLIT = "train"


# Loads the TinyStories training split.
# Each row contains one story in its "text" field.
def load_training_dataset():
    print(f"Loading {DATASET_NAME}...")

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
    )

    print(f"Loaded {len(dataset):,} stories.")

    return dataset


# Yields one story at a time to the tokenizer trainer.
# This avoids creating another large list containing every story.
def story_iterator(dataset):
    for example in dataset:
        yield example["text"]


# Creates an empty byte-level BPE tokenizer.
# The vocabulary and merge rules are learned later from TinyStories.
def create_tokenizer():
    tokenizer = Tokenizer(
        BPE(
            unk_token=UNK_TOKEN,
        )
    )

    # Byte-level preprocessing gives the tokenizer coverage over raw bytes.
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
    )

    # Converts byte-level representations back into normal text.
    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


# Configures how the BPE vocabulary is learned.
# The trainer learns up to VOCAB_SIZE tokens and preserves our special tokens.
def create_trainer():
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )


# Trains the tokenizer on TinyStories.
# Saves the complete trained tokenizer to tokenizer/tokenizer.json.
def train_tokenizer():
    dataset = load_training_dataset()

    tokenizer = create_tokenizer()
    trainer = create_trainer()

    print(
        f"Training BPE tokenizer with target "
        f"vocabulary size {VOCAB_SIZE:,}..."
    )

    tokenizer.train_from_iterator(
        story_iterator(dataset),
        trainer=trainer,
        length=len(dataset),
    )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(
        str(TOKENIZER_PATH)
    )

    print()
    print("Tokenizer training complete.")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")


# Starts tokenizer training only when this file is executed directly.
if __name__ == "__main__":
    train_tokenizer()