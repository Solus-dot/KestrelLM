import math
import os

import torch
from tqdm import tqdm

from config import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_INTERVAL,
    CONTEXT_LENGTH,
    GRADIENT_ACCUMULATION_STEPS,
    LATEST_CHECKPOINT,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MILESTONE_CHECKPOINT_INTERVAL,
    MIN_LEARNING_RATE,
    RESUME_CHECKPOINT,
    RUN_UNTIL_STEP,
    TRAINING_STEPS,
    VALIDATION_BATCHES,
    VALIDATION_INTERVAL,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from dataloader import create_train_dataloader, create_validation_dataloader
from model import KestrelLM, compute_loss


# Chooses an AMD/NVIDIA GPU first, Apple MPS second, and CPU otherwise.
# PyTorch exposes AMD ROCm GPUs through the torch.cuda interface.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Prints the accelerator backend and physical device used for training.
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


# Separates matrix-like parameters from 1D parameters for AdamW weight decay.
def create_optimizer(model):
    decay_parameters = []
    no_decay_parameters = []

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        if parameter.dim() >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=LEARNING_RATE,
        betas=(ADAM_BETA_1, ADAM_BETA_2),
    )


# Uses linear warmup followed by cosine decay over the complete training schedule.
def get_learning_rate(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS

    decay_steps = TRAINING_STEPS - WARMUP_STEPS
    decay_progress = (step - WARMUP_STEPS) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


# Updates the learning rate for every AdamW parameter group.
def set_learning_rate(optimizer, learning_rate):
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


# Computes average validation loss without gradients or parameter updates.
def evaluate(model, validation_loader, device):
    model.eval()

    total_loss = 0.0
    batches_evaluated = 0

    with torch.no_grad():
        for x, y in validation_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = compute_loss(logits, y)

            total_loss += loss.item()
            batches_evaluated += 1

            if batches_evaluated >= VALIDATION_BATCHES:
                break

    return total_loss / batches_evaluated


# Builds all state required to continue the training run later.
def create_checkpoint(model, optimizer, global_step, data_pass, tokens_processed):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "data_pass": data_pass,
        "tokens_processed": tokens_processed,
    }


# Writes to a temporary file before atomically replacing the destination.
# An interrupted save therefore cannot corrupt an existing valid checkpoint.
def atomic_save(checkpoint, checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


# Updates latest.pt and creates permanent milestone checkpoints when configured.
def save_checkpoint(model, optimizer, global_step, data_pass, tokens_processed):
    checkpoint = create_checkpoint(
        model, optimizer, global_step, data_pass, tokens_processed
    )

    atomic_save(checkpoint, LATEST_CHECKPOINT)

    milestone_path = None

    if global_step % MILESTONE_CHECKPOINT_INTERVAL == 0:
        milestone_path = CHECKPOINT_DIR / f"step_{global_step}.pt"
        atomic_save(checkpoint, milestone_path)

    return milestone_path


# Moves all tensor-valued optimizer state to the selected accelerator.
# This makes checkpoints portable between MPS, ROCm, CUDA, and CPU systems.
def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


# Restores model parameters, AdamW state, and training counters.
# The checkpoint is first loaded onto CPU so checkpoints remain cross-platform.
def load_checkpoint(model, optimizer, checkpoint_path, device):
    if checkpoint_path is None:
        return 0, 0, 0

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Training was configured to resume, so refusing to start from scratch."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)

    global_step = checkpoint["global_step"]
    data_pass = checkpoint.get("data_pass", checkpoint.get("epoch", 0))
    tokens_processed = checkpoint["tokens_processed"]

    return global_step, data_pass, tokens_processed


# Restores the configured checkpoint or starts fresh when resume is explicitly disabled.
def initialize_training_state(model, optimizer, device):
    return load_checkpoint(model, optimizer, RESUME_CHECKPOINT, device)


# Runs pretraining with gradient accumulation, LR scheduling, validation,
# gradient clipping, checkpointing, and safe interruption handling.
def main():
    device = get_device()

    train_loader = create_train_dataloader()
    validation_loader = create_validation_dataloader()

    model = KestrelLM().to(device)
    optimizer = create_optimizer(model)

    global_step, data_pass, tokens_processed = initialize_training_state(
        model, optimizer, device
    )

    microbatch_tokens = BATCH_SIZE * CONTEXT_LENGTH
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    tokens_per_optimizer_step = microbatch_tokens * GRADIENT_ACCUMULATION_STEPS
    total_training_tokens = TRAINING_STEPS * tokens_per_optimizer_step
    remaining_steps = RUN_UNTIL_STEP - global_step
    remaining_tokens = total_training_tokens - tokens_processed

    if RUN_UNTIL_STEP > TRAINING_STEPS:
        raise ValueError("RUN_UNTIL_STEP cannot exceed TRAINING_STEPS.")

    if global_step >= RUN_UNTIL_STEP:
        raise ValueError(
            f"Checkpoint is already at step {global_step}, "
            f"but this invocation stops at step {RUN_UNTIL_STEP}."
        )

    print_device_info(device)

    if RESUME_CHECKPOINT is None:
        print("Starting new training run")
    else:
        print(f"Resumed from: {RESUME_CHECKPOINT}")
        print(f"Starting optimizer step: {global_step:,}")
        print(f"Previously processed tokens: {tokens_processed:,}")

    print(f"Physical batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Tokens per microbatch: {microbatch_tokens:,}")
    print(f"Tokens per optimizer step: {tokens_per_optimizer_step:,}")
    print(f"Full training steps: {TRAINING_STEPS:,}")
    print(f"Full training tokens: {total_training_tokens:,}")
    print(f"Remaining optimizer steps: {remaining_steps:,}")
    print(f"Remaining training tokens: {remaining_tokens:,}")
    print(f"Peak learning rate: {LEARNING_RATE}")
    print(f"Minimum learning rate: {MIN_LEARNING_RATE}")
    print(f"Warmup steps: {WARMUP_STEPS:,}")
    print(f"Adam betas: ({ADAM_BETA_1}, {ADAM_BETA_2})")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Maximum gradient norm: {MAX_GRAD_NORM}")
    print(f"Validation interval: {VALIDATION_INTERVAL:,}")
    print(f"Validation batches: {VALIDATION_BATCHES:,}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL:,}")
    print(f"Milestone interval: {MILESTONE_CHECKPOINT_INTERVAL:,}\n")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    accumulation_step = 0
    accumulated_loss = 0.0
    accumulated_tokens = 0

    progress_bar = tqdm(
        total=RUN_UNTIL_STEP, initial=global_step, desc="Training", unit="step"
    )

    try:
        while global_step < RUN_UNTIL_STEP:
            data_pass += 1

            for x, y in train_loader:
                if global_step >= RUN_UNTIL_STEP:
                    break

                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = compute_loss(logits, y)

                scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
                scaled_loss.backward()

                accumulated_loss += loss.item()
                accumulated_tokens += x.numel()
                accumulation_step += 1

                if accumulation_step < GRADIENT_ACCUMULATION_STEPS:
                    continue

                global_step += 1

                learning_rate = get_learning_rate(global_step)
                set_learning_rate(optimizer, learning_rate)

                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                tokens_processed += accumulated_tokens
                mean_loss = accumulated_loss / GRADIENT_ACCUMULATION_STEPS

                assert math.isfinite(mean_loss)
                assert torch.isfinite(gradient_norm)

                progress_bar.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{learning_rate:.2e}",
                    grad_norm=f"{gradient_norm.item():.2f}",
                    tokens=f"{tokens_processed:,}",
                )
                progress_bar.update(1)

                accumulation_step = 0
                accumulated_loss = 0.0
                accumulated_tokens = 0

                if global_step % VALIDATION_INTERVAL == 0:
                    validation_loss = evaluate(model, validation_loader, device)

                    progress_bar.write(
                        f"Step {global_step:,} | "
                        f"Training loss: {mean_loss:.4f} | "
                        f"Validation loss: {validation_loss:.4f} | "
                        f"LR: {learning_rate:.2e} | "
                        f"Gradient norm: {gradient_norm.item():.2f} | "
                        f"Tokens: {tokens_processed:,}"
                    )

                    assert math.isfinite(validation_loss)
                    model.train()

                if global_step % CHECKPOINT_INTERVAL == 0:
                    milestone_path = save_checkpoint(
                        model, optimizer, global_step, data_pass, tokens_processed
                    )

                    progress_bar.write(f"Updated checkpoint: {LATEST_CHECKPOINT}")

                    if milestone_path is not None:
                        progress_bar.write(f"Saved milestone: {milestone_path}")

    except KeyboardInterrupt:
        progress_bar.write("\nTraining interrupted.")

        milestone_path = save_checkpoint(
            model, optimizer, global_step, data_pass, tokens_processed
        )

        progress_bar.write(
            f"Saved completed state at step {global_step:,} "
            f"and {tokens_processed:,} tokens."
        )
        progress_bar.write(f"Latest checkpoint: {LATEST_CHECKPOINT}")

        if milestone_path is not None:
            progress_bar.write(f"Milestone checkpoint: {milestone_path}")

        progress_bar.close()
        return

    milestone_path = save_checkpoint(
        model, optimizer, global_step, data_pass, tokens_processed
    )

    progress_bar.close()

    print(f"\nFinal optimizer step: {global_step:,}")
    print(f"Total tokens processed: {tokens_processed:,}")
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")

    if milestone_path is not None:
        print(f"Milestone checkpoint: {milestone_path}")

    print("Training run: PASSED")


if __name__ == "__main__":
    main()
