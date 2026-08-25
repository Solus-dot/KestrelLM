import numpy as np
from datasets import load_from_disk
from tokenizers import Tokenizer
from tqdm import tqdm

from config import (
    EOS_TOKEN,
    TINYSTORIES_DIR,
    TOKENIZED_DATA_DIR,
    TOKENIZER_PATH,
    TRAIN_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
    VOCAB_SIZE,
)


# Loads the trained BPE tokenizer used by KestrelLM.
def load_tokenizer():
    return Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )


# Loads the locally stored TinyStories dataset.
# This does not contact Hugging Face.
def load_dataset():
    return load_from_disk(
        str(TINYSTORIES_DIR)
    )


# Converts one TinyStory into token IDs.
# An <eos> token is appended so the model can learn document boundaries.
def encode_story(tokenizer, text, eos_id):
    token_ids = tokenizer.encode(text).ids
    token_ids.append(eos_id)

    return token_ids


# Tokenizes one dataset split and writes the IDs directly to a binary file.
# Token IDs are stored as uint16 because our vocabulary is smaller than 65,536.
# Writing each story incrementally avoids keeping the entire dataset in RAM.
def preprocess_split(tokenizer, split, output_path):
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    total_tokens = 0

    print(f"\nProcessing {output_path.name}...")

    with open(output_path, "wb") as output_file:
        for example in tqdm(split):
            token_ids = encode_story(
                tokenizer,
                example["text"],
                eos_id,
            )

            tokens = np.asarray(
                token_ids,
                dtype=np.uint16,
            )

            tokens.tofile(output_file)

            total_tokens += len(tokens)

    print(f"Saved {total_tokens:,} tokens.")
    print(f"Output: {output_path}")

    return total_tokens


# Preprocesses both the training and validation splits.
# The resulting .bin files will later be memory-mapped by the PyTorch dataset.
def main():
    if VOCAB_SIZE >= 2**16:
        raise ValueError(
            "VOCAB_SIZE is too large for uint16 token storage."
        )

    tokenizer = load_tokenizer()
    dataset = load_dataset()

    TOKENIZED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_tokens = preprocess_split(
        tokenizer,
        dataset["train"],
        TRAIN_TOKENS_PATH,
    )

    validation_tokens = preprocess_split(
        tokenizer,
        dataset["validation"],
        VALIDATION_TOKENS_PATH,
    )

    print("\nPreprocessing complete.")
    print(f"Training tokens:   {train_tokens:,}")
    print(f"Validation tokens: {validation_tokens:,}")


if __name__ == "__main__":
    main()