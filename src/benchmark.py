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


# All configurations preserve an effective batch size of 32 sequences.
BENCHMARK_CONFIGS = [
    (2, 16),
    (4, 8),
    (8, 4),
    (16, 2),
    (32, 1),
]

WARMUP_STEPS = 3
BENCHMARK_STEPS = 20


# Requires an AMD ROCm GPU and refuses to silently fall back to the CPU.
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm GPU is not available to PyTorch. "
            "Refusing to run the benchmark on CPU."
        )

    if torch.version.hip is None:
        raise RuntimeError(
            "PyTorch sees a CUDA-style device, but this is not a ROCm build."
        )

    return torch.device("cuda:0")


# Waits for all queued ROCm operations to complete before reading the clock.
def synchronize():
    torch.cuda.synchronize()


# Clears cached GPU memory between benchmark configurations.
def empty_cache():
    torch.cuda.empty_cache()


# Prints the ROCm and GPU information used for this benchmark.
def print_device_info(device):
    print(f"Device: {device}")
    print("GPU backend: ROCm")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HIP version: {torch.version.hip}")
    print(f"PyTorch: {torch.__version__}")


# Creates the same AdamW optimizer configuration used during real training.
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


# Returns another batch and restarts the dataloader if it is exhausted.
def get_next_batch(data_iterator, data_loader):
    try:
        batch = next(data_iterator)

    except StopIteration:
        data_iterator = iter(data_loader)
        batch = next(data_iterator)

    return batch, data_iterator


# Performs one complete optimizer update using gradient accumulation.
def run_optimizer_step(
    model,
    optimizer,
    data_loader,
    data_iterator,
    device,
    accumulation_steps,
):
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0

    for _ in range(accumulation_steps):
        (x, y), data_iterator = get_next_batch(data_iterator, data_loader)

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = compute_loss(logits, y)

        scaled_loss = loss / accumulation_steps
        scaled_loss.backward()

        total_loss += loss.item()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM,
    )

    optimizer.step()

    mean_loss = total_loss / accumulation_steps

    return mean_loss, gradient_norm.item(), data_iterator


# Benchmarks one physical-batch and accumulation configuration.
def benchmark_configuration(device, dataset, batch_size, accumulation_steps):
    effective_batch_size = batch_size * accumulation_steps
    tokens_per_step = effective_batch_size * CONTEXT_LENGTH

    print(
        f"\nB={batch_size}, "
        f"accumulation={accumulation_steps}, "
        f"effective batch={effective_batch_size}"
    )

    print(f"Tokens per optimizer step: {tokens_per_step:,}")

    torch.manual_seed(42)

    model = None
    optimizer = None
    data_loader = None
    data_iterator = None

    try:
        data_loader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        data_iterator = iter(data_loader)

        model = KestrelLM().to(device)
        optimizer = create_optimizer(model)
        model.train()

        model_device = next(model.parameters()).device

        if model_device.type != "cuda":
            raise RuntimeError(
                f"Model ended up on {model_device} instead of the ROCm GPU."
            )

        print(f"Model device: {model_device}")
        print(f"Warming up for {WARMUP_STEPS} steps...")

        for _ in range(WARMUP_STEPS):
            _, _, data_iterator = run_optimizer_step(
                model,
                optimizer,
                data_loader,
                data_iterator,
                device,
                accumulation_steps,
            )

        synchronize()

        print(f"Benchmarking {BENCHMARK_STEPS} steps...")

        start_time = time.perf_counter()

        for _ in range(BENCHMARK_STEPS):
            loss, gradient_norm, data_iterator = run_optimizer_step(
                model,
                optimizer,
                data_loader,
                data_iterator,
                device,
                accumulation_steps,
            )

        synchronize()

        elapsed_time = time.perf_counter() - start_time

        seconds_per_step = elapsed_time / BENCHMARK_STEPS
        steps_per_second = BENCHMARK_STEPS / elapsed_time
        tokens_per_second = tokens_per_step / seconds_per_step

        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory reserved: {memory_reserved:.2f} GB")

        return {
            "batch_size": batch_size,
            "accumulation_steps": accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "seconds_per_step": seconds_per_step,
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_per_second,
            "loss": loss,
            "gradient_norm": gradient_norm,
        }

    except torch.OutOfMemoryError:
        print("OUT OF MEMORY - skipping this configuration.")
        return None

    finally:
        del model
        del optimizer
        del data_loader
        del data_iterator

        empty_cache()


# Runs every configuration and ranks them by training throughput.
def main():
    device = get_device()
    dataset = create_train_dataset()

    print_device_info(device)

    print(f"Context length: {CONTEXT_LENGTH}")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measured steps: {BENCHMARK_STEPS}")

    results = []

    for batch_size, accumulation_steps in BENCHMARK_CONFIGS:
        result = benchmark_configuration(
            device,
            dataset,
            batch_size,
            accumulation_steps,
        )

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("Every benchmark configuration failed.")

    results.sort(
        key=lambda result: result["tokens_per_second"],
        reverse=True,
    )

    print("\nBenchmark results\n")

    print(
        f"{'Batch':>6} "
        f"{'Accum':>6} "
        f"{'Eff. Batch':>10} "
        f"{'Sec/Step':>10} "
        f"{'Steps/s':>10} "
        f"{'Tokens/s':>12}"
    )

    print("-" * 62)

    for result in results:
        print(
            f"{result['batch_size']:>6} "
            f"{result['accumulation_steps']:>6} "
            f"{result['effective_batch_size']:>10} "
            f"{result['seconds_per_step']:>10.3f} "
            f"{result['steps_per_second']:>10.3f} "
            f"{result['tokens_per_second']:>12,.0f}"
        )

    fastest = results[0]

    remaining_tokens = 600_014_848 - 159_744_000
    remaining_hours = remaining_tokens / fastest["tokens_per_second"] / 3600

    print(
        f"\nFastest configuration: "
        f"B={fastest['batch_size']}, "
        f"accumulation={fastest['accumulation_steps']}"
    )

    print(f"Throughput: {fastest['tokens_per_second']:,.0f} tokens/s")
    print(f"Estimated remaining training time: {remaining_hours:.1f} hours")


if __name__ == "__main__":
    main()
