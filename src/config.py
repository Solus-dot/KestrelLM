from dataclasses import dataclass
from pathlib import Path


# Project root and file-system paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKENIZER_DIR = PROJECT_ROOT / "tokenizer"
TOKENIZER_PATH = TOKENIZER_DIR / "tokenizer.json"

DATA_DIR = PROJECT_ROOT / "data"
TINYSTORIES_DIR = DATA_DIR / "tinystories"
HF_CACHE_DIR = DATA_DIR / "hf_cache"
TOKENIZED_DATA_DIR = DATA_DIR / "tokenized"

TRAIN_TOKENS_PATH = TOKENIZED_DATA_DIR / "train.bin"
VALIDATION_TOKENS_PATH = TOKENIZED_DATA_DIR / "validation.bin"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"


# Tokenizer configuration.
VOCAB_SIZE = 8192

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


# Stores the architectural dimensions for one KestrelLM model size.
@dataclass(frozen=True)
class ModelConfig:
    name: str
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    def validate(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"{self.name}: d_model must be divisible by n_heads.")

        if self.d_head != 64:
            raise ValueError(f"{self.name}: expected d_head=64, got {self.d_head}.")

        if self.d_ff != 4 * self.d_model:
            raise ValueError(f"{self.name}: d_ff must equal 4 * d_model.")


# Scaling-study model configurations.
# Depth, context length, and head dimension stay fixed while model width changes.
KESTREL_SMALL = ModelConfig(
    name="Kestrel-S",
    context_length=512,
    d_model=256,
    n_layers=6,
    n_heads=4,
    d_ff=1024,
)

KESTREL_MEDIUM = ModelConfig(
    name="Kestrel-M",
    context_length=512,
    d_model=512,
    n_layers=6,
    n_heads=8,
    d_ff=2048,
)

KESTREL_LARGE = ModelConfig(
    name="Kestrel-L",
    context_length=512,
    d_model=768,
    n_layers=6,
    n_heads=12,
    d_ff=3072,
)

MODEL_CONFIGS = [KESTREL_SMALL, KESTREL_MEDIUM, KESTREL_LARGE]

for model_config in MODEL_CONFIGS:
    model_config.validate()


# Kestrel-M remains the default architecture.
# These aliases keep all existing code and checkpoints compatible.
ACTIVE_MODEL_CONFIG = KESTREL_MEDIUM

CONTEXT_LENGTH = ACTIVE_MODEL_CONFIG.context_length
D_MODEL = ACTIVE_MODEL_CONFIG.d_model
N_LAYERS = ACTIVE_MODEL_CONFIG.n_layers
N_HEADS = ACTIVE_MODEL_CONFIG.n_heads
D_HEAD = ACTIVE_MODEL_CONFIG.d_head
D_FF = ACTIVE_MODEL_CONFIG.d_ff


# Training configuration.
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-5
WARMUP_STEPS = 500

ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.95
WEIGHT_DECAY = 0.1

MAX_GRAD_NORM = 1.0

# Approximately 600M total training tokens.
TRAINING_STEPS = 36_622
RUN_UNTIL_STEP = 36_622

VALIDATION_INTERVAL = 250
VALIDATION_BATCHES = 20

CHECKPOINT_INTERVAL = 250
MILESTONE_CHECKPOINT_INTERVAL = 5_000

# Set to LATEST_CHECKPOINT to resume from the newest rolling checkpoint.
RESUME_CHECKPOINT = None
