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
GENERATION_MEASURED_RUNS = 5

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
# GPU work is asynchronous, so synchronization is required for accurate timing.
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


# Produces enough real tokenizer IDs for every benchmark context length.
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


# Measures one ordinary forward pass over an existing prompt.
# This represents the prefill or prompt-processing phase of inference.
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


# Generates text using the original inference method.
# Every generated token recomputes the entire visible context from scratch.
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


# Generates text using the KV cache.
# The prompt is processed once and later steps process only the newest token.
def generate_cached(model, prompt_ids, device, new_tokens):
    if len(prompt_ids) + new_tokens > CONTEXT_LENGTH:
        raise ValueError(
            "Cached benchmark sequence exceeds the model context length."
        )

    generated_ids = list(prompt_ids)

    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits, kv_cache = model(
            input_ids,
            use_cache=True,
        )

        next_token_id = torch.argmax(
            logits[0, -1]
        ).item()

        generated_ids.append(next_token_id)

        for _ in range(new_tokens - 1):
            input_ids = torch.tensor(
                [[next_token_id]],
                dtype=torch.long,
                device=device,
            )

            logits, kv_cache = model(
                input_ids,
                kv_cache=kv_cache,
                use_cache=True,
            )

            next_token_id = torch.argmax(
                logits[0, -1]
            ).item()

            generated_ids.append(next_token_id)

    return generated_ids


# Checks that both generation implementations produce exactly the same tokens.
def verify_generation_parity(model, prompt_ids, device):
    test_tokens = 50

    naive_ids = generate_naive(
        model,
        prompt_ids,
        device,
        test_tokens,
    )

    cached_ids = generate_cached(
        model,
        prompt_ids,
        device,
        test_tokens,
    )

    if naive_ids != cached_ids:
        raise RuntimeError(
            "Naive and KV-cached generation produced different token sequences."
        )

    print(
        f"Generation parity: PASSED "
        f"({test_tokens} generated tokens)"
    )


# Measures one generation implementation over several identical runs.
def benchmark_generation_method(
    generation_function,
    model,
    prompt_ids,
    device,
):
    generation_function(
        model,
        prompt_ids,
        device,
        GENERATION_WARMUP_TOKENS,
    )

    synchronize(device)

    measured_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(GENERATION_MEASURED_RUNS):
        synchronize(device)

        start_time = time.perf_counter()

        generation_function(
            model,
            prompt_ids,
            device,
            GENERATION_TOKENS,
        )

        synchronize(device)

        elapsed_time = time.perf_counter() - start_time
        measured_times.append(elapsed_time)

    average_time = sum(measured_times) / len(measured_times)

    tokens_per_second = (
        GENERATION_TOKENS / average_time
    )

    milliseconds_per_token = (
        average_time / GENERATION_TOKENS * 1000
    )

    peak_memory_gb = None

    if device.type == "cuda":
        peak_memory_gb = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )

    return {
        "time": average_time,
        "tokens_per_second": tokens_per_second,
        "milliseconds_per_token": milliseconds_per_token,
        "peak_memory_gb": peak_memory_gb,
    }


# Runs prompt-processing and naive-versus-KV-cache generation benchmarks.
def main():
    device = get_device()

    print_device_info(device)
    print(f"Checkpoint: {FINAL_CHECKPOINT}")
    print(f"Context length: {CONTEXT_LENGTH}\n")

    tokenizer = load_tokenizer()
    model = load_model(device)

    benchmark_tokens = create_benchmark_tokens(
        tokenizer
    )

    print("Prompt processing\n")

    print(
        f"{'Tokens':>8} "
        f"{'Time (ms)':>12} "
        f"{'Tokens/s':>12}"
    )

    print("-" * 36)

    for prompt_length in PREFILL_LENGTHS:
        prompt_ids = benchmark_tokens[
            :prompt_length
        ]

        seconds_per_run, tokens_per_second = (
            benchmark_prefill(
                model,
                prompt_ids,
                device,
            )
        )

        milliseconds = seconds_per_run * 1000

        print(
            f"{prompt_length:>8} "
            f"{milliseconds:>12.3f} "
            f"{tokens_per_second:>12,.0f}"
        )

    generation_prompt = benchmark_tokens[
        :GENERATION_PROMPT_LENGTH
    ]

    print("\nGeneration benchmark\n")
    print(
        f"Prompt length: "
        f"{GENERATION_PROMPT_LENGTH} tokens"
    )
    print(
        f"Generated tokens per run: "
        f"{GENERATION_TOKENS}"
    )
    print(
        f"Measured runs: "
        f"{GENERATION_MEASURED_RUNS}\n"
    )

    verify_generation_parity(
        model,
        generation_prompt,
        device,
    )

    print("\nBenchmarking naive generation...")

    naive_result = benchmark_generation_method(
        generate_naive,
        model,
        generation_prompt,
        device,
    )

    print("Benchmarking KV-cached generation...")

    cached_result = benchmark_generation_method(
        generate_cached,
        model,
        generation_prompt,
        device,
    )

    speedup = (
        cached_result["tokens_per_second"]
        / naive_result["tokens_per_second"]
    )

    print("\nGeneration results\n")

    print(
        f"{'Method':>12} "
        f"{'Time (s)':>10} "
        f"{'Tokens/s':>12} "
        f"{'ms/token':>12} "
        f"{'Peak GB':>10}"
    )

    print("-" * 62)

    print(
        f"{'Naive':>12} "
        f"{naive_result['time']:>10.3f} "
        f"{naive_result['tokens_per_second']:>12,.2f} "
        f"{naive_result['milliseconds_per_token']:>12.3f} "
        f"{naive_result['peak_memory_gb']:>10.3f}"
    )

    print(
        f"{'KV cache':>12} "
        f"{cached_result['time']:>10.3f} "
        f"{cached_result['tokens_per_second']:>12,.2f} "
        f"{cached_result['milliseconds_per_token']:>12.3f} "
        f"{cached_result['peak_memory_gb']:>10.3f}"
    )

    print(f"\nKV-cache speedup: {speedup:.2f}x")

    if speedup > 1:
        reduction = (
            1
            - cached_result["milliseconds_per_token"]
            / naive_result["milliseconds_per_token"]
        ) * 100

        print(
            f"Latency reduction: "
            f"{reduction:.1f}%"
        )

    print("\nInference comparison benchmark: PASSED")


if __name__ == "__main__":
    main()
