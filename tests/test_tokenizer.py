from tokenizers import Tokenizer

from config import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    TOKENIZER_PATH,
    UNK_TOKEN,
    VOCAB_SIZE,
)


TEST_TEXT = "Once upon a time, there was a little fox."


# Loads the trained tokenizer from disk.
# Raises a clear error if the tokenizer has not been trained yet.
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}.\n"
            "Run: uv run python scripts/train_tokenizer.py"
        )

    return Tokenizer.from_file(
        str(TOKENIZER_PATH)
    )


# Encodes a sample sentence and decodes it again.
# The decoded text should exactly match the original input.
def test_encode_decode(tokenizer):
    encoded = tokenizer.encode(TEST_TEXT)
    decoded = tokenizer.decode(encoded.ids)

    print("Original:")
    print(TEST_TEXT)

    print("\nTokens:")
    print(encoded.tokens)

    print("\nToken IDs:")
    print(encoded.ids)

    print("\nDecoded:")
    print(decoded)

    assert decoded == TEST_TEXT, (
        "Encode/decode round-trip failed."
    )

    print("\nEncode/decode test: PASSED")


# Checks that the tokenizer vocabulary matches our configured size.
# The model embedding table will later depend on this value.
def test_vocab_size(tokenizer):
    actual_vocab_size = tokenizer.get_vocab_size()

    print("\nVocabulary size:")
    print(actual_vocab_size)

    assert actual_vocab_size == VOCAB_SIZE, (
        f"Expected vocabulary size {VOCAB_SIZE}, "
        f"but found {actual_vocab_size}."
    )

    print("Vocabulary size test: PASSED")


# Checks that every required special token exists.
# Also verifies that each special token has a unique token ID.
def test_special_tokens(tokenizer):
    print("\nSpecial token IDs:")

    special_token_ids = {}

    for token in SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)

        print(f"{token}: {token_id}")

        assert token_id is not None, (
            f"Special token {token} is missing."
        )

        special_token_ids[token] = token_id

    assert len(set(special_token_ids.values())) == len(SPECIAL_TOKENS), (
        "Two or more special tokens share the same ID."
    )

    print("Special token test: PASSED")


# Verifies that the configured special-token strings have not changed.
def test_special_token_names():
    assert PAD_TOKEN == "<pad>"
    assert UNK_TOKEN == "<unk>"
    assert BOS_TOKEN == "<bos>"
    assert EOS_TOKEN == "<eos>"


# Runs all tokenizer sanity checks.
def main():
    tokenizer = load_tokenizer()

    test_special_token_names()
    test_encode_decode(tokenizer)
    test_vocab_size(tokenizer)
    test_special_tokens(tokenizer)

    print("\nAll tokenizer tests passed.")


if __name__ == "__main__":
    main()