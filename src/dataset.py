from datasets import load_dataset
from tokenizers import Tokenizer

from config import (
    EOS_TOKEN,
    TOKENIZER_PATH,
    CONTEXT_LENGTH
)

DATASET_NAME = "roneneldan/TinyStories"
DATASET_SPLIT = "train"

# Loads the trained KestrelLM tokenizer.
def load_tokenizer():
    return Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )

# Loads the TinyStories training split
from datasets import load_from_disk
from config import TINYSTORIES_DIR

# Loads our local TinyStories dataset
def load_training_dataset():
    dataset = load_from_disk(
        str(TINYSTORIES_DIR)
    )

    return dataset["train"]

# Converts one story into token IDs
# Appends <eos> so document boundaries are visible to the model
def encode_story(tokenizer, text):
    token_ids = tokenizer.encode(text).ids

    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    token_ids.append(eos_id)

    return token_ids

# Encodes several stories and concatenates them into one token stream
# The <eos> token remains between stories as a document separator
def build_token_stream(tokenizer, dataset, num_stories):
    token_stream = []

    for i in range(num_stories):
        story = dataset[i]["text"]

        story_tokens = encode_story(
            tokenizer,
            story,
        )

        token_stream.extend(story_tokens)

    return token_stream

# Takes CONTEXT_LENGTH + 1 consecutive tokens from the token stream
# Returns an input sequence X and its one-token-shifted target sequence Y
def create_training_pair(token_stream, start_index):
    chunk = token_stream[
        start_index : start_index + CONTEXT_LENGTH + 1
    ]

    if len(chunk) != CONTEXT_LENGTH + 1:
        raise ValueError(
            "Not enough tokens remaining to create a full training pair."
        )

    x = chunk[:-1]
    y = chunk[1:]

    return x, y

# Builds a small token stream and verifies that input and target
# sequences are exactly one token apart.
def main():
    tokenizer = load_tokenizer()
    dataset = load_training_dataset()

    token_stream = build_token_stream(
        tokenizer,
        dataset,
        num_stories=10,
    )

    x, y = create_training_pair(
        token_stream,
        start_index=0,
    )

    print("Input length:")
    print(len(x))

    print("\nTarget length:")
    print(len(y))

    print("\nFirst 10 input IDs:")
    print(x[:10])

    print("\nFirst 10 target IDs:")
    print(y[:10])

    assert len(x) == CONTEXT_LENGTH
    assert len(y) == CONTEXT_LENGTH

    # Every target token should be the token immediately after
    # the corresponding input token in the original stream.
    assert x[1:] == y[:-1]

    print("\nTraining pair test: PASSED")

if __name__ == "__main__":
    main()