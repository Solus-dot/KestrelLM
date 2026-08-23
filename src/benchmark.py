import time
import torch

from config import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    CONTEXT_LENGTH,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
)
from dataloader import create_dataloader
from dataset import create_train_dataset
from model import KestrelLM, compute_loss


BATCH_SIZE = 2
ACCUMULATION_STEPS = 16

WARMUP_STEPS = 3
BENCHMARK_STEPS = 200
REPORT_INTERVAL = 20


# Chooses Apple's MPS backend when available and otherwise falls back to the CPU.
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Waits for all queued MPS work to finish before reading the clock.
def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


# Creates the same AdamW configuration used during real training.
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


# Returns the next batch and restarts the dataloader if necessary.
def get_next_batch(data_iterator, data_loader):
    try:
        batch = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        batch = next(data_iterator)

    return batch, data_iterator


# Performs one complete optimizer update over ACCUMULATION_STEPS microbatches.
def run_optimizer_step(model, optimizer, data_loader, data_iterator, device):
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for _ in range(ACCUMULATION_STEPS):
        (x, y), data_iterator = get_next_batch(data_iterator, data_loader)

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = compute_loss(logits, y)

        scaled_loss = loss / ACCUMULATION_STEPS
        scaled_loss.backward()

        total_loss += loss.item()

    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    optimizer.step()

    mean_loss = total_loss / ACCUMULATION_STEPS

    return mean_loss, gradient_norm.item(), data_iterator


# Returns MPS memory usage when available.
def get_memory_usage(device):
    if device.type != "mps":
        return None

    allocated = torch.mps.current_allocated_memory()
    driver = torch.mps.driver_allocated_memory()

    return allocated, driver


# Measures sustained training throughput and reports rolling performance.
def main():
    device = get_device()

    dataset = create_train_dataset()
    data_loader = create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    data_iterator = iter(data_loader)

    torch.manual_seed(42)

    model = KestrelLM().to(device)
    optimizer = create_optimizer(model)

    effective_batch_size = BATCH_SIZE * ACCUMULATION_STEPS
    tokens_per_step = effective_batch_size * CONTEXT_LENGTH

    print(f"Device: {device}")
    print(f"Physical batch size: {BATCH_SIZE}")
    print(f"Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Tokens per optimizer step: {tokens_per_step:,}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measured steps: {BENCHMARK_STEPS}")
    print(f"Report interval: {REPORT_INTERVAL}\n")

    print("Warming up...")

    for _ in range(WARMUP_STEPS):
        _, _, data_iterator = run_optimizer_step(
            model, optimizer, data_loader, data_iterator, device
        )

    synchronize(device)

    print("Starting sustained benchmark...\n")

    benchmark_start = time.perf_counter()
    interval_start = benchmark_start

    for step in range(1, BENCHMARK_STEPS + 1):
        loss, gradient_norm, data_iterator = run_optimizer_step(
            model, optimizer, data_loader, data_iterator, device
        )

        if step % REPORT_INTERVAL != 0:
            continue

        synchronize(device)

        now = time.perf_counter()

        interval_time = now - interval_start
        total_time = now - benchmark_start

        interval_tokens = REPORT_INTERVAL * tokens_per_step
        total_tokens = step * tokens_per_step

        interval_tokens_per_second = interval_tokens / interval_time
        total_tokens_per_second = total_tokens / total_time

        interval_seconds_per_step = interval_time / REPORT_INTERVAL

        memory = get_memory_usage(device)

        line = (
            f"Step {step:3d} | "
            f"{interval_seconds_per_step:.3f} s/step | "
            f"{interval_tokens_per_second:,.0f} tok/s | "
            f"avg {total_tokens_per_second:,.0f} tok/s | "
            f"loss {loss:.4f} | "
            f"grad {gradient_norm:.2f}"
        )

        if memory is not None:
            allocated, driver = memory

            line += (
                f" | MPS allocated {allocated / 1024**3:.2f} GB"
                f" | driver {driver / 1024**3:.2f} GB"
            )

        print(line)

        interval_start = now

    synchronize(device)

    total_time = time.perf_counter() - benchmark_start
    total_tokens = BENCHMARK_STEPS * tokens_per_step
    tokens_per_second = total_tokens / total_time
    seconds_per_step = total_time / BENCHMARK_STEPS

    print("\nFinal sustained result")
    print(f"Time: {total_time / 60:.2f} minutes")
    print(f"Seconds per optimizer step: {seconds_per_step:.3f}")
    print(f"Throughput: {tokens_per_second:,.0f} tokens/s")

    estimated_full_hours = 600_000_000 / tokens_per_second / 3600

    print(f"Estimated 600M-token runtime: {estimated_full_hours:.1f} hours")


if __name__ == "__main__":
    main()

