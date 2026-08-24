from pathlib import Path

import torch

from config import (
    CONTEXT_LENGTH,
    D_FF,
    D_HEAD,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
)
from model import KestrelLM, count_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "final_600m.pt"
RELEASE_DIR = PROJECT_ROOT / "release"
OUTPUT_CHECKPOINT = RELEASE_DIR / "kestrel_30m.pt"


# Loads the final training checkpoint and verifies that its weights still match
# the current KestrelLM architecture before creating the distributable model.
def load_and_validate_model():
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {SOURCE_CHECKPOINT}")

    checkpoint = torch.load(
        SOURCE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = KestrelLM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


# Creates an inference-only checkpoint containing model weights and useful
# architecture/training metadata, but no AdamW optimizer state.
def create_release_checkpoint(model, training_checkpoint):
    return {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_head": D_HEAD,
            "d_ff": D_FF,
        },
        "parameter_count": count_parameters(model),
        "training": {
            "global_step": training_checkpoint["global_step"],
            "tokens_processed": training_checkpoint["tokens_processed"],
            "dataset": "TinyStories",
        },
        "evaluation": {
            "validation_tokens": 4_690_944,
            "validation_loss": 1.5031,
            "validation_perplexity": 4.4957,
        },
    }


# Exports the trained model into a smaller checkpoint intended for inference
# and public distribution.
def main():
    model, training_checkpoint = load_and_validate_model()

    release_checkpoint = create_release_checkpoint(
        model,
        training_checkpoint,
    )

    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        release_checkpoint,
        OUTPUT_CHECKPOINT,
    )

    size_mb = OUTPUT_CHECKPOINT.stat().st_size / 1024**2

    print(f"Source checkpoint: {SOURCE_CHECKPOINT}")
    print(f"Release checkpoint: {OUTPUT_CHECKPOINT}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training step: {training_checkpoint['global_step']:,}")
    print(f"Training tokens: {training_checkpoint['tokens_processed']:,}")
    print(f"Release size: {size_mb:.1f} MiB")
    print("Model export: PASSED")


if __name__ == "__main__":
    main()
