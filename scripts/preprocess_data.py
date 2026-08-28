import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

from config import (
    EOS_TOKEN,
    TOKENIZED_DATA_DIR,
    TOKENIZER_PATH,
    TRAIN_TOKENS_PATH,
    VALIDATION_TOKENS_PATH,
    VOCAB_SIZE,
)


# The main pretraining corpus contains exactly 1,199,996,928 Kestrel tokens.
# The mixture is measured after tokenization so the final proportions are exact.
TRAIN_SOURCES = [
    {
        "name": "FineWeb-Edu-Dedup",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "fineweb-edu-dedup",
        "tokens": 839_997_850,
        "seed": 42,
    },
    {
        "name": "Cosmopedia v2",
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "cosmopedia-v2",
        "tokens": 239_999_386,
        "seed": 43,
    },
    {
        "name": "FineWiki English",
        "dataset": "HuggingFaceFW/finewiki",
        "config": "en",
        "tokens": 119_999_692,
        "seed": 44,
    },
]

# A separate 10M-token validation stream follows the same 70/20/10 mixture.
# Validation documents are consumed before training documents from each stream,
# which prevents overlap between the two local token files.
VALIDATION_TOKENS = {
    "FineWeb-Edu-Dedup": 7_000_000,
    "Cosmopedia v2": 2_000_000,
    "FineWiki English": 1_000_000,
}

SHUFFLE_BUFFER_SIZE = 10_000


# Loads the frozen general-domain BPE tokenizer.
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    return Tokenizer.from_file(str(TOKENIZER_PATH))


# Opens one upstream corpus as a deterministic shuffled stream.
# Only the documents needed to satisfy the local token quota are downloaded.
def load_source_stream(source):
    print(
        f"\nOpening {source['name']} "
        f"({source['dataset']} [{source['config']}])..."
    )

    dataset = load_dataset(
        source["dataset"],
        source["config"],
        split="train",
        streaming=True,
    )

    return dataset.shuffle(
        seed=source["seed"],
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )


# Encodes one document and appends EOS so packed documents retain boundaries.
def encode_document(tokenizer, text, eos_id):
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    ).ids

    token_ids.append(eos_id)

    return token_ids


# Consumes documents from an existing source iterator until exactly token_quota
# Kestrel token IDs have been written. The same iterator can then continue into
# the training split, ensuring its documents do not overlap with validation.
def write_token_quota(
    tokenizer,
    source_iterator,
    output_file,
    token_quota,
    eos_id,
    description,
):
    tokens_written = 0

    progress = tqdm(
        total=token_quota,
        desc=description,
        unit="tok",
        unit_scale=True,
    )

    while tokens_written < token_quota:
        try:
            example = next(source_iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"{description}: source stream ended before reaching "
                f"{token_quota:,} tokens."
            ) from error

        text = example.get("text")

        if not text or not text.strip():
            continue

        token_ids = encode_document(
            tokenizer,
            text,
            eos_id,
        )

        remaining = token_quota - tokens_written

        if len(token_ids) > remaining:
            token_ids = token_ids[:remaining]

            # Preserve a document boundary even when the final document must be
            # truncated to make the source token quota exact.
            if token_ids:
                token_ids[-1] = eos_id

        tokens = np.asarray(
            token_ids,
            dtype=np.uint16,
        )

        tokens.tofile(output_file)

        tokens_written += len(tokens)
        progress.update(len(tokens))

    progress.close()

    return tokens_written


# Streams one source once: held-out validation tokens are consumed first, then
# training tokens continue from the same iterator with no document reuse.
def preprocess_source(
    tokenizer,
    source,
    train_file,
    validation_file,
    eos_id,
):
    dataset = load_source_stream(source)
    source_iterator = iter(dataset)

    validation_quota = VALIDATION_TOKENS[source["name"]]

    validation_tokens = write_token_quota(
        tokenizer,
        source_iterator,
        validation_file,
        validation_quota,
        eos_id,
        f"{source['name']} validation",
    )

    training_tokens = write_token_quota(
        tokenizer,
        source_iterator,
        train_file,
        source["tokens"],
        eos_id,
        f"{source['name']} training",
    )

    return training_tokens, validation_tokens


# Builds the final packed binary token streams atomically.
# Temporary files are renamed only after every source reaches its quota.
def main():
    if VOCAB_SIZE >= 2**16:
        raise ValueError(
            "VOCAB_SIZE is too large for uint16 token storage."
        )

    tokenizer = load_tokenizer()
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    if eos_id is None:
        raise ValueError(f"EOS token not found in tokenizer: {EOS_TOKEN}")

    TOKENIZED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_temporary_path = TRAIN_TOKENS_PATH.with_name(
        TRAIN_TOKENS_PATH.name + ".tmp"
    )

    validation_temporary_path = VALIDATION_TOKENS_PATH.with_name(
        VALIDATION_TOKENS_PATH.name + ".tmp"
    )

    total_training_tokens = 0
    total_validation_tokens = 0

    print("Building general-domain KestrelLM token corpus.")
    print(
        "Training target: "
        f"{sum(source['tokens'] for source in TRAIN_SOURCES):,} tokens"
    )
    print(
        "Validation target: "
        f"{sum(VALIDATION_TOKENS.values()):,} tokens"
    )

    with (
        open(train_temporary_path, "wb") as train_file,
        open(validation_temporary_path, "wb") as validation_file,
    ):
        for source in TRAIN_SOURCES:
            training_tokens, validation_tokens = preprocess_source(
                tokenizer,
                source,
                train_file,
                validation_file,
                eos_id,
            )

            total_training_tokens += training_tokens
            total_validation_tokens += validation_tokens

    expected_training_tokens = sum(
        source["tokens"]
        for source in TRAIN_SOURCES
    )

    expected_validation_tokens = sum(
        VALIDATION_TOKENS.values()
    )

    if total_training_tokens != expected_training_tokens:
        raise RuntimeError(
            f"Expected {expected_training_tokens:,} training tokens, "
            f"wrote {total_training_tokens:,}."
        )

    if total_validation_tokens != expected_validation_tokens:
        raise RuntimeError(
            f"Expected {expected_validation_tokens:,} validation tokens, "
            f"wrote {total_validation_tokens:,}."
        )

    os.replace(
        train_temporary_path,
        TRAIN_TOKENS_PATH,
    )

    os.replace(
        validation_temporary_path,
        VALIDATION_TOKENS_PATH,
    )

    print("\nPreprocessing complete.")
    print(f"Training tokens:   {total_training_tokens:,}")
    print(f"Validation tokens: {total_validation_tokens:,}")
    print(f"Training file:     {TRAIN_TOKENS_PATH}")
    print(f"Validation file:   {VALIDATION_TOKENS_PATH}")


if __name__ == "__main__":
    main()
