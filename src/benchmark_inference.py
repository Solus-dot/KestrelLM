import time

import torch
from tokenizers import Tokenizer

from config import CHECKPOINT_DIR, CONTEXT_LENGTH, TOKENIZER_PATH
from model import KestrelLM


FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_600m.pt"

PREFILL_LENGTHS = [32, 128, 256, 512]
PREFILL_WARMUP_RUNS = 3
PREFILL_MEASURED_RUNS = 20

GENERATION_PROMPT_LENGTH = 32
GENERATION_TOKENS = 200
GENERATION_WARMUP_TOKENS = 20
GENERATION_MEASURED_RUNS = 3

BENCHMARK_TEXT = (
    "Once upon a time, there was a little girl who loved to explore the forest. "
    "Every morning, she packed her small bag and walked along the winding path. "
)


# Chooses an AMD/NVIDIA GPU first, Apple MPS second, and CPU otherwise.
# PyTorch exposes AMD ROCm GPUs through the torch.cuda interface.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Waits until all queued accelerator operations have completed.
# Accurate GPU timings require synchronization before reading the clock.
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# Prints the accelerator backend and physical device used for the benchmark.
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


# Loads the tokenizer used to create the pretraining dataset.
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    return Tokenizer.from_file(str(TOKENIZER_PATH))


# Loads only the model parameters from the finished pretraining checkpoint.
def load_model(device):
    if not FINAL_CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {FINAL_CHECKPOINT}")

    checkpoint = torch.load(
        FINAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = KestrelLM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# Produces a repeatable stream of real tokenizer IDs long enough for all tests.
# Using tokenizer output keeps benchmark inputs representative of normal inference.
def create_benchmark_tokens(tokenizer):
    repeated_text = BENCHMARK_TEXT

    while True:
        token_ids = tokenizer.encode(
            repeated_text,
            add_special_tokens=False,
        ).ids

        if len(token_ids) >= CONTEXT_LENGTH:
            return token_ids[:CONTEXT_LENGTH]

        repeated_text += BENCHMARK_TEXT


# Measures one forward pass over an existing prompt.
# This is the prompt-processing or prefill phase of autoregressive inference.
def benchmark_prefill(model, token_ids, device):
    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(PREFILL_WARMUP_RUNS):
            model(input_ids)

        synchronize(device)

        start_time = time.perf_counter()

        for _ in range(PREFILL_MEASURED_RUNS):
            model(input_ids)

        synchronize(device)

    elapsed_time = time.perf_counter() - start_time

    seconds_per_run = elapsed_time / PREFILL_MEASURED_RUNS
    tokens_per_second = len(token_ids) / seconds_per_run

    return seconds_per_run, tokens_per_second


# Generates tokens using the current naive inference implementation.
# Every new token causes the entire visible context to be recomputed.
def generate_naive(model, prompt_ids, device, new_tokens):
    generated_ids = list(prompt_ids)

    with torch.inference_mode():
        for _ in range(new_tokens):
            context_ids = generated_ids[-CONTEXT_LENGTH:]

            input_ids = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=device,
            )

            logits = model(input_ids)
            next_token_id = torch.argmax(logits[0, -1]).item()

            generated_ids.append(next_token_id)

    return generated_ids


# Measures autoregressive decoding throughput without a KV cache.
# EOS is intentionally ignored so every run generates exactly the same token count.
def benchmark_generation(model, prompt_ids, device):
    generate_naive(
        model,
        prompt_ids,
        device,
        GENERATION_WARMUP_TOKENS,
    )

    synchronize(device)

    measured_times = []

    for _ in range(GENERATION_MEASURED_RUNS):
        synchronize(device)
        start_time = time.perf_counter()

        generate_naive(
            model,
            prompt_ids,
            device,
            GENERATION_TOKENS,
        )

        synchronize(device)
        elapsed_time = time.perf_counter() - start_time

        measured_times.append(elapsed_time)

    average_time = sum(measured_times) / len(measured_times)
    tokens_per_second = GENERATION_TOKENS / average_time
    milliseconds_per_token = average_time / GENERATION_TOKENS * 1000

    return average_time, tokens_per_second, milliseconds_per_token


# Prints current accelerator memory usage when supported by the backend.
def print_memory_usage(device):
    if device.type != "cuda":
        return

    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    reserved_gb = torch.cuda.memory_reserved() / 1024**3

    print(f"GPU memory allocated: {allocated_gb:.2f} GB")
    print(f"GPU memory reserved: {reserved_gb:.2f} GB")


# Runs prompt-processing and naive autoregressive-generation benchmarks.
def main():
    device = get_device()

    print_device_info(device)
    print(f"Checkpoint: {FINAL_CHECKPOINT}")
    print("Inference mode: naive full-context recomputation")
    print("KV cache: disabled\n")

    tokenizer = load_tokenizer()
    model = load_model(device)

    benchmark_tokens = create_benchmark_tokens(tokenizer)

    print("Prompt processing\n")

    print(
        f"{'Tokens':>8} "
        f"{'Time (ms)':>12} "
        f"{'Tokens/s':>12}"
    )

    print("-" * 36)

    for prompt_length in PREFILL_LENGTHS:
        prompt_ids = benchmark_tokens[:prompt_length]

        seconds_per_run, tokens_per_second = benchmark_prefill(
            model,
            prompt_ids,
            device,
        )

        milliseconds = seconds_per_run * 1000

        print(
            f"{prompt_length:>8} "
            f"{milliseconds:>12.3f} "
            f"{tokens_per_second:>12,.0f}"
        )

    generation_prompt = benchmark_tokens[:GENERATION_PROMPT_LENGTH]

    print("\nNaive autoregressive generation\n")
    print(f"Prompt length: {GENERATION_PROMPT_LENGTH} tokens")
    print(f"Generated tokens per run: {GENERATION_TOKENS}")
    print(f"Measured runs: {GENERATION_MEASURED_RUNS}")

    average_time, tokens_per_second, milliseconds_per_token = benchmark_generation(
        model,
        generation_prompt,
        device,
    )

    print(f"\nAverage generation time: {average_time:.3f} s")
    print(f"Generation throughput: {tokens_per_second:,.2f} tokens/s")
    print(f"Latency per generated token: {milliseconds_per_token:.3f} ms")

    print_memory_usage(device)

    print("\nBaseline inference benchmark: PASSED")


if __name__ == "__main__":
    main()
