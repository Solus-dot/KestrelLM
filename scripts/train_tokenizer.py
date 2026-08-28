from itertools import islice

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


# Representative 70/20/10 sample of the final pretraining mixture.
TOKENIZER_DOCUMENTS = 100_000

FINEWEB_DOCUMENTS = 70_000
COSMOPEDIA_DOCUMENTS = 20_000
FINEWIKI_DOCUMENTS = 10_000

SHUFFLE_BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42


# Opens one Hugging Face dataset as a shuffled streaming dataset.
# Streaming prevents the tokenizer-training step from downloading the full corpus.
def load_stream(dataset_name, dataset_config):
    print(f"Opening {dataset_name} [{dataset_config}]...")

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split="train",
        streaming=True,
    )

    return dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )


# Yields up to document_count non-empty text documents from one dataset stream.
def text_iterator(dataset, document_count):
    documents_yielded = 0

    for example in dataset:
        text = example["text"]

        if not text or not text.strip():
            continue

        yield text

        documents_yielded += 1

        if documents_yielded >= document_count:
            break


# Produces the representative 70/20/10 tokenizer-training corpus.
def training_text_iterator():
    fineweb = load_stream(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
    )

    cosmopedia = load_stream(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
    )

    finewiki = load_stream(
        "HuggingFaceFW/finewiki",
        "en",
    )

    print(f"FineWeb-Edu-Dedup documents: {FINEWEB_DOCUMENTS:,}")
    yield from text_iterator(fineweb, FINEWEB_DOCUMENTS)

    print(f"Cosmopedia v2 documents: {COSMOPEDIA_DOCUMENTS:,}")
    yield from text_iterator(cosmopedia, COSMOPEDIA_DOCUMENTS)

    print(f"FineWiki documents: {FINEWIKI_DOCUMENTS:,}")
    yield from text_iterator(finewiki, FINEWIKI_DOCUMENTS)


# Creates an empty byte-level BPE tokenizer.
# Byte-level tokenization guarantees coverage for arbitrary input text.
def create_tokenizer():
    tokenizer = Tokenizer(
        BPE(
            unk_token=UNK_TOKEN,
        )
    )

    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=False,
    )

    tokenizer.decoder = ByteLevelDecoder()

    return tokenizer


# Configures vocabulary learning and reserves the model's control tokens.
def create_trainer():
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )


# Trains the general-domain KestrelLM tokenizer and saves tokenizer.json.
def train_tokenizer():
    tokenizer = create_tokenizer()
    trainer = create_trainer()

    print(f"Training on {TOKENIZER_DOCUMENTS:,} streamed documents.")
    print(f"Target vocabulary size: {VOCAB_SIZE:,}\n")

    tokenizer.train_from_iterator(
        training_text_iterator(),
        trainer=trainer,
        length=TOKENIZER_DOCUMENTS,
    )

    TOKENIZER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(str(TOKENIZER_PATH))

    print()
    print("Tokenizer training complete.")
    print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
    print(f"Saved tokenizer to: {TOKENIZER_PATH}")


# Starts tokenizer training only when this file is executed directly.
if __name__ == "__main__":
    train_tokenizer()
