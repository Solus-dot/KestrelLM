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


# Chooses Apple's MPS backend when available and otherwise falls back to the CPU.
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Separates parameters into groups with and without weight decay.
# Matrix-like weights are decayed, while 1D RMSNorm scales are not.
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


# Uses linear warmup followed by cosine decay over the complete training run.
def get_learning_rate(step):
    if step <= WARMUP_STEPS:
        return LEARNING_RATE * step / WARMUP_STEPS

    decay_steps = TRAINING_STEPS - WARMUP_STEPS
    decay_progress = (step - WARMUP_STEPS) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))

    return MIN_LEARNING_RATE + cosine * (LEARNING_RATE - MIN_LEARNING_RATE)


# Updates the learning rate for every optimizer parameter group.
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


# Builds the training state stored inside every checkpoint.
def create_checkpoint(model, optimizer, global_step, epoch, tokens_processed):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "tokens_processed": tokens_processed,
    }


# Writes to a temporary file before atomically replacing the target checkpoint.
def atomic_save(checkpoint, checkpoint_path):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, checkpoint_path)


# Updates latest.pt and optionally creates a permanent milestone checkpoint.
def save_checkpoint(model, optimizer, global_step, epoch, tokens_processed):
    checkpoint = create_checkpoint(model, optimizer, global_step, epoch, tokens_processed)

    atomic_save(checkpoint, LATEST_CHECKPOINT)

    milestone_path = None

    if global_step % MILESTONE_CHECKPOINT_INTERVAL == 0:
        milestone_path = CHECKPOINT_DIR / f"step_{global_step}.pt"
        atomic_save(checkpoint, milestone_path)

    return milestone_path


# Restores model parameters, AdamW state, and training counters.
def load_checkpoint(model, optimizer, checkpoint_path, device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    global_step = checkpoint["global_step"]
    epoch = checkpoint["epoch"]
    tokens_processed = checkpoint["tokens_processed"]

    return global_step, epoch, tokens_processed


# Starts fresh when no checkpoint is configured or resumes from the given checkpoint.
def initialize_training_state(model, optimizer, device):
    if RESUME_CHECKPOINT is None:
        return 0, 0, 0

    return load_checkpoint(model, optimizer, RESUME_CHECKPOINT, device)


# Accumulates several microbatches before every optimizer update.
# The current invocation stops at RUN_UNTIL_STEP while the LR schedule uses TRAINING_STEPS.
def main():
    device = get_device()

    train_loader = create_train_dataloader()
    validation_loader = create_validation_dataloader()

    model = KestrelLM().to(device)
    optimizer = create_optimizer(model)

    global_step, epoch, tokens_processed = initialize_training_state(model, optimizer, device)

    microbatch_tokens = BATCH_SIZE * CONTEXT_LENGTH
    effective_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    tokens_per_optimizer_step = microbatch_tokens * GRADIENT_ACCUMULATION_STEPS

    total_training_tokens = TRAINING_STEPS * tokens_per_optimizer_step
    pilot_tokens = RUN_UNTIL_STEP * tokens_per_optimizer_step

    if RUN_UNTIL_STEP > TRAINING_STEPS:
        raise ValueError("RUN_UNTIL_STEP cannot exceed TRAINING_STEPS.")

    if global_step >= RUN_UNTIL_STEP:
        raise ValueError(
            f"Checkpoint is already at step {global_step}, "
            f"but this invocation stops at step {RUN_UNTIL_STEP}."
        )

    print(f"Device: {device}")

    if RESUME_CHECKPOINT is None:
        print("Starting new training run")
    else:
        print(f"Resumed from: {RESUME_CHECKPOINT}")
        print(f"Starting optimizer step: {global_step:,}")
        print(f"Previously processed tokens: {tokens_processed:,}")

    print(f"Physical batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Tokens per optimizer step: {tokens_per_optimizer_step:,}")
    print(f"Full training steps: {TRAINING_STEPS:,}")
    print(f"Full training tokens: {total_training_tokens:,}")
    print(f"This run stops at step: {RUN_UNTIL_STEP:,}")
    print(f"Tokens by end of this run: {pilot_tokens:,}")
    print(f"Warmup steps: {WARMUP_STEPS:,}")
    print(f"Validation interval: {VALIDATION_INTERVAL:,}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL:,}\n")

    model.train()

    accumulation_step = 0
    accumulated_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        total=RUN_UNTIL_STEP,
        initial=global_step,
        desc="Training",
        unit="step",
    )

    while global_step < RUN_UNTIL_STEP:
        epoch += 1

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
            accumulation_step += 1
            tokens_processed += x.numel()

            if accumulation_step < GRADIENT_ACCUMULATION_STEPS:
                continue

            global_step += 1

            learning_rate = get_learning_rate(global_step)
            set_learning_rate(optimizer, learning_rate)

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            mean_loss = accumulated_loss / GRADIENT_ACCUMULATION_STEPS

            progress_bar.set_postfix(
                epoch=epoch,
                loss=f"{mean_loss:.4f}",
                lr=f"{learning_rate:.2e}",
                grad_norm=f"{gradient_norm.item():.2f}",
                tokens=f"{tokens_processed:,}",
            )
            progress_bar.update(1)

            assert math.isfinite(mean_loss)
            assert torch.isfinite(gradient_norm)

            accumulation_step = 0
            accumulated_loss = 0.0

            if global_step % VALIDATION_INTERVAL == 0:
                validation_loss = evaluate(model, validation_loader, device)

                progress_bar.write(
                    f"Step {global_step:,} | "
                    f"Epoch {epoch} | "
                    f"Training loss: {mean_loss:.4f} | "
                    f"Validation loss: {validation_loss:.4f} | "
                    f"LR: {learning_rate:.2e} | "
                    f"Gradient norm: {gradient_norm.item():.2f} | "
                    f"Tokens: {tokens_processed:,}"
                )

                model.train()

            if global_step % CHECKPOINT_INTERVAL == 0:
                milestone_path = save_checkpoint(model, optimizer, global_step, epoch, tokens_processed)
                progress_bar.write(f"Updated checkpoint: {LATEST_CHECKPOINT}")

                if milestone_path is not None:
                    progress_bar.write(f"Saved milestone: {milestone_path}")

    milestone_path = save_checkpoint(model, optimizer, global_step, epoch, tokens_processed)

    progress_bar.close()

    print(f"\nFinal optimizer step: {global_step:,}")
    print(f"Total tokens processed: {tokens_processed:,}")
    print(f"Latest checkpoint: {LATEST_CHECKPOINT}")

    if milestone_path is not None:
        print(f"Milestone checkpoint: {milestone_path}")

    print("Pilot training run: PASSED")


if __name__ == "__main__":
    main()