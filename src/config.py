from pathlib import Path


# Absolute path to the root of the KestrelLM repository.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directory and file used to store the trained tokenizer.
TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"

# Directory used for datasets and other local training data.
DATA_DIR = PROJECT_ROOT / "data"

# Local copy of the processed TinyStories dataset.
TINYSTORIES_DIR = DATA_DIR / "tinystories"

# Temporary Hugging Face cache used only while downloading the dataset.
HF_CACHE_DIR = DATA_DIR / "hf_cache"

# Number of unique tokens in the tokenizer vocabulary.
VOCAB_SIZE = 8192

# Special tokens used by the tokenizer and training pipeline.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]

# Maximum number of tokens the model can process in one sequence.
CONTEXT_LENGTH = 512

# Hidden dimension of every token representation.
D_MODEL = 512

# Number of transformer blocks.
N_LAYERS = 6

# Number of attention heads in each transformer block.
N_HEADS = 8

# Hidden dimension handled by each attention head.
D_HEAD = D_MODEL // N_HEADS

# Intermediate dimension of the feed-forward network.
D_FF = 2048

# Multi-head attention requires the hidden dimension to split evenly
# across all attention heads.
assert D_MODEL % N_HEADS == 0, (
    "D_MODEL must be evenly divisible by N_HEADS."
)
