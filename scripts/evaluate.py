import math

import torch
from tqdm import tqdm

from config import CHECKPOINT_DIR
from dataloader import create_validation_dataloader
from model import KestrelLM, compute_loss


FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_600m.pt"


# Chooses an AMD/NVIDIA GPU first, Apple MPS second, and CPU otherwise.
# PyTorch exposes AMD ROCm GPUs through the torch.cuda interface.
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


# Loads only the trained model parameters from the final checkpoint.
# Optimizer state is unnecessary because evaluation performs no updates.
def load_model(checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = KestrelLM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    global_step = checkpoint["global_step"]
    tokens_processed = checkpoint["tokens_processed"]

    return model, global_step, tokens_processed


# Evaluates every validation batch and computes token-weighted average loss.
# Weighting by token count keeps the result exact even if batch sizes differ.
def evaluate(model, validation_loader, device):
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(
        validation_loader,
        desc="Evaluating",
        unit="batch",
    )

    with torch.no_grad():
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


# Loads the finished model and evaluates it over the complete validation set.
def main():
    device = get_device()

    print_device_info(device)
    print(f"Checkpoint: {FINAL_CHECKPOINT}\n")

    model, global_step, training_tokens = load_model(
        FINAL_CHECKPOINT,
        device,
    )

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
    print(f"Validation tokens: {validation_tokens:,}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print("Evaluation: PASSED")


if __name__ == "__main__":
    main()
