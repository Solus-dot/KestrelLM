import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm

from config import KESTREL_MEDIUM, ModelConfig
from dataloader import create_validation_dataloader
from model import KestrelLM, compute_loss, count_parameters


# Reads the checkpoint to evaluate from the command line.
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    return parser.parse_args()


# Chooses an AMD/NVIDIA GPU first, Apple MPS second, and CPU otherwise.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Prints the accelerator backend and physical device used for evaluation.
def print_device_info(device):
    print(f"Device: {device}")

    if device.type == "cuda":
        backend = "ROCm" if torch.version.hip is not None else "CUDA"

        print(f"GPU backend: {backend}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        if torch.version.hip is not None:
            print(f"HIP version: {torch.version.hip}")

    elif device.type == "mps":
        print("GPU backend: Apple MPS")

    else:
        print("GPU backend: CPU")


# Loads the architecture stored in newer checkpoints.
# Older Kestrel-M checkpoints fall back to the original medium configuration.
def get_model_config(checkpoint):
    config_data = checkpoint.get("model_config")

    if config_data is None:
        return KESTREL_MEDIUM

    return ModelConfig(**config_data)


# Loads only the trained model parameters needed for evaluation.
def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_config = get_model_config(checkpoint)

    model = KestrelLM(config=model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    global_step = checkpoint["global_step"]
    tokens_processed = checkpoint["tokens_processed"]

    return (
        model,
        model_config,
        global_step,
        tokens_processed,
    )


# Evaluates every validation batch and computes token-weighted average loss.
def evaluate(model, validation_loader, device):
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(
        validation_loader,
        desc="Evaluating",
        unit="batch",
    )

    with torch.inference_mode():
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = compute_loss(logits, y)

            batch_tokens = y.numel()

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            running_loss = total_loss / total_tokens

            progress_bar.set_postfix(
                loss=f"{running_loss:.4f}",
                tokens=f"{total_tokens:,}",
            )

    validation_loss = total_loss / total_tokens
    perplexity = math.exp(validation_loss)

    return validation_loss, perplexity, total_tokens


# Loads one checkpoint and evaluates it over the complete validation set.
def main():
    args = parse_args()
    device = get_device()

    print_device_info(device)
    print(f"Checkpoint: {args.checkpoint}\n")

    (
        model,
        model_config,
        global_step,
        training_tokens,
    ) = load_model(
        args.checkpoint,
        device,
    )

    print(f"Model: {model_config.name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Checkpoint step: {global_step:,}")
    print(f"Training tokens: {training_tokens:,}")

    validation_loader = create_validation_dataloader()

    print(f"Validation batches: {len(validation_loader):,}")
    print("\nRunning full validation evaluation...\n")

    validation_loss, perplexity, validation_tokens = evaluate(
        model,
        validation_loader,
        device,
    )

    print("\nFinal evaluation")
    print(f"Model: {model_config.name}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Training tokens: {training_tokens:,}")
    print(f"Validation tokens: {validation_tokens:,}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("Evaluation: PASSED")


if __name__ == "__main__":
    main()
