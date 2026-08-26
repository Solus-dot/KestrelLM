import time

import torch

from config import (
    ADAM_BETA_1,
    ADAM_BETA_2,
    BATCH_SIZE,
    KESTREL_LARGE,
    KESTREL_MEDIUM,
    KESTREL_SMALL,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
)
from dataloader import create_dataloader
from dataset import create_train_dataset
from model import KestrelLM, compute_loss, count_parameters


MODEL_CONFIGS = [
    KESTREL_SMALL,
    KESTREL_MEDIUM,
    KESTREL_LARGE,
]

WARMUP_STEPS = 3
BENCHMARK_STEPS = 20


# Requires the ROCm GPU used for KestrelLM training.
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm GPU is not available to PyTorch.")

    if torch.version.hip is None:
        raise RuntimeError("PyTorch is not using a ROCm build.")

    return torch.device("cuda:0")


# Waits for queued ROCm operations to finish before reading timings.
def synchronize():
    torch.cuda.synchronize()


# Releases cached allocator memory between model-size benchmarks.
def clear_gpu_memory():
    torch.cuda.empty_cache()


# Prints the accelerator environment used for the benchmark.
def print_device_info(device):
    print(f"Device: {device}")
    print("GPU backend: ROCm")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"HIP version: {torch.version.hip}")
    print(f"PyTorch: {torch.__version__}")


# Creates the same AdamW configuration used during normal training.
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


# Returns another batch and restarts the dataloader if necessary.
def get_next_batch(data_iterator, data_loader):
    try:
        batch = next(data_iterator)
    except StopIteration:
        data_iterator = iter(data_loader)
        batch = next(data_iterator)

    return batch, data_iterator


# Performs one complete forward, backward, and optimizer update.
def run_training_step(model, optimizer, data_loader, data_iterator, device):
    optimizer.zero_grad(set_to_none=True)

    (x, y), data_iterator = get_next_batch(data_iterator, data_loader)

    x = x.to(device)
    y = y.to(device)

    logits = model(x)
    loss = compute_loss(logits, y)

    loss.backward()

    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM,
    )

    optimizer.step()

    return loss.item(), gradient_norm.item(), data_iterator


# Benchmarks one model size using the same physical batch and sequence length.
def benchmark_model(config, device, dataset):
    print(f"\n{config.name}")
    print(f"Parameters: {count_parameters(KestrelLM(config=config)):,}")
    print(f"d_model: {config.d_model}")
    print(f"Layers: {config.n_layers}")
    print(f"Heads: {config.n_heads}")
    print(f"d_head: {config.d_head}")
    print(f"d_ff: {config.d_ff}")

    torch.manual_seed(42)
    clear_gpu_memory()

    model = None
    optimizer = None
    data_loader = None
    data_iterator = None

    try:
        data_loader = create_dataloader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        data_iterator = iter(data_loader)

        model = KestrelLM(config=config).to(device)
        optimizer = create_optimizer(model)
        model.train()

        parameter_count = count_parameters(model)
        tokens_per_step = BATCH_SIZE * config.context_length

        print(f"Batch size: {BATCH_SIZE}")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Warming up for {WARMUP_STEPS} steps...")

        for _ in range(WARMUP_STEPS):
            _, _, data_iterator = run_training_step(
                model,
                optimizer,
                data_loader,
                data_iterator,
                device,
            )

        synchronize()

        # The optimizer state has now been initialized, so peak memory measured
        # below represents steady-state training rather than first-step setup.
        torch.cuda.reset_peak_memory_stats()

        print(f"Benchmarking {BENCHMARK_STEPS} steps...")

        start_time = time.perf_counter()

        for _ in range(BENCHMARK_STEPS):
            loss, gradient_norm, data_iterator = run_training_step(
                model,
                optimizer,
                data_loader,
                data_iterator,
                device,
            )

        synchronize()

        elapsed_time = time.perf_counter() - start_time

        seconds_per_step = elapsed_time / BENCHMARK_STEPS
        steps_per_second = BENCHMARK_STEPS / elapsed_time
        tokens_per_second = tokens_per_step / seconds_per_step
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

        return {
            "name": config.name,
            "parameters": parameter_count,
            "seconds_per_step": seconds_per_step,
            "steps_per_second": steps_per_second,
            "tokens_per_second": tokens_per_second,
            "peak_memory_gb": peak_memory_gb,
            "loss": loss,
            "gradient_norm": gradient_norm,
        }

    except torch.OutOfMemoryError:
        print("OUT OF MEMORY")

        return {
            "name": config.name,
            "parameters": count_parameters(model) if model is not None else 0,
            "oom": True,
        }

    finally:
        del model
        del optimizer
        del data_loader
        del data_iterator

        clear_gpu_memory()


# Compares training throughput and peak VRAM across the three width-scaled models.
def main():
    device = get_device()
    dataset = create_train_dataset()

    print_device_info(device)

    print(f"\nBatch size: {BATCH_SIZE}")
    print("Context length: 512")
    print(f"Warmup steps: {WARMUP_STEPS}")
    print(f"Measured steps: {BENCHMARK_STEPS}")

    results = []

    for config in MODEL_CONFIGS:
        results.append(
            benchmark_model(
                config,
                device,
                dataset,
            )
        )

    print("\nScaling benchmark results\n")

    print(
        f"{'Model':>10} "
        f"{'Params':>12} "
        f"{'Sec/Step':>10} "
        f"{'Tokens/s':>12} "
        f"{'Peak GB':>10}"
    )

    print("-" * 60)

    for result in results:
        if result.get("oom"):
            print(
                f"{result['name']:>10} "
                f"{result['parameters']:>12,} "
                f"{'OOM':>10} "
                f"{'OOM':>12} "
                f"{'OOM':>10}"
            )

            continue

        print(
            f"{result['name']:>10} "
            f"{result['parameters']:>12,} "
            f"{result['seconds_per_step']:>10.3f} "
            f"{result['tokens_per_second']:>12,.0f} "
            f"{result['peak_memory_gb']:>10.2f}"
        )

    print("\nTraining scaling benchmark: PASSED")


if __name__ == "__main__":
    main()
