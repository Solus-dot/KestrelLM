import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model

from config import (
    PROJECT_ROOT,
    TOKENIZER_PATH,
    VOCAB_SIZE,
    ModelConfig,
)
from model import KestrelLM, count_parameters


# Reads the trusted training checkpoint and desired public release stage.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted final PyTorch training checkpoint.",
    )

    parser.add_argument(
        "--stage",
        choices={"base", "instruct"},
        required=True,
        help="Public release stage.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional release directory override.",
    )

    return parser.parse_args()


# Reconstructs the exact Kestrel architecture recorded by the training run.
def load_training_checkpoint(checkpoint_path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_config" not in checkpoint:
        raise KeyError(
            "Training checkpoint does not contain model_config."
        )

    model_config = ModelConfig(
        **checkpoint["model_config"]
    )

    model = KestrelLM(
        config=model_config
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, model_config, checkpoint


# Writes architecture and training provenance separately from tensor weights.
def create_release_config(
    model,
    model_config,
    checkpoint,
    stage,
):
    release_config = {
        "model_type": "kestrel",
        "stage": stage,
        "vocab_size": VOCAB_SIZE,
        "parameter_count": count_parameters(model),
        "tie_word_embeddings": True,
        **asdict(model_config),
    }

    if "global_step" in checkpoint:
        release_config["global_step"] = checkpoint["global_step"]

    if "tokens_processed" in checkpoint:
        release_config["tokens_processed"] = checkpoint["tokens_processed"]

    return release_config


# Reloads the SafeTensors file into a fresh model to verify the public artifact.
def validate_safetensors(weights_path, model_config):
    validation_model = KestrelLM(
        config=model_config
    )

    load_model(
        validation_model,
        str(weights_path),
        strict=True,
        device="cpu",
    )

    validation_model.eval()


# Exports inference-only weights, JSON configuration, and tokenizer.
def main():
    args = parse_args()

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "release"
            / f"kestrel-{args.stage}"
        )

    weights_path = output_dir / "model.safetensors"
    config_path = output_dir / "config.json"
    tokenizer_output_path = output_dir / "tokenizer.json"

    model, model_config, checkpoint = load_training_checkpoint(
        args.checkpoint
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # save_model handles KestrelLM's tied embedding/LM-head storage correctly.
    save_model(
        model,
        str(weights_path),
        metadata={
            "format": "pt",
            "model_type": "kestrel",
            "stage": args.stage,
        },
    )

    release_config = create_release_config(
        model,
        model_config,
        checkpoint,
        args.stage,
    )

    config_path.write_text(
        json.dumps(
            release_config,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_PATH}"
        )

    shutil.copy2(
        TOKENIZER_PATH,
        tokenizer_output_path,
    )

    validate_safetensors(
        weights_path,
        model_config,
    )

    size_mb = (
        weights_path.stat().st_size
        / 1024**2
    )

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Stage: {args.stage}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Weights: {weights_path}")
    print(f"Config: {config_path}")
    print(f"Tokenizer: {tokenizer_output_path}")
    print(f"SafeTensors size: {size_mb:.1f} MiB")
    print("SafeTensors reload validation: PASSED")
    print("Model export: PASSED")


if __name__ == "__main__":
    main()
