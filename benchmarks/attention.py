import time

import torch

from config import CHECKPOINT_DIR, CONTEXT_LENGTH, VOCAB_SIZE
from model import KestrelLM


FINAL_CHECKPOINT = CHECKPOINT_DIR / "final.pt"

SEQUENCE_LENGTHS = [32, 128, 256, 512]

WARMUP_RUNS = 5
MEASURED_RUNS = 20

GENERATION_PROMPT_LENGTH = 32
GENERATION_TOKENS = 400
GENERATION_WARMUP_RUNS = 2
GENERATION_MEASURED_RUNS = 5


# Chooses an AMD/NVIDIA GPU first, Apple MPS second, and CPU otherwise.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Waits for asynchronous accelerator work to finish before recording timings.
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

    elif device.type == "mps":
        torch.mps.synchronize()


# Prints the accelerator and backend used for the benchmark.
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


# Loads manual-attention and SDPA models using exactly the same trained weights.
def load_models(device):
    checkpoint = torch.load(
        FINAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    manual_model = KestrelLM(attention_backend="manual")
    sdpa_model = KestrelLM(attention_backend="sdpa")

    manual_model.load_state_dict(checkpoint["model_state_dict"])
    sdpa_model.load_state_dict(checkpoint["model_state_dict"])

    manual_model.to(device)
    sdpa_model.to(device)

    manual_model.eval()
    sdpa_model.eval()

    return manual_model, sdpa_model


# Measures full-sequence inference throughput for one attention backend.
def benchmark_forward(model, token_ids, device):
    with torch.inference_mode():
        for _ in range(WARMUP_RUNS):
            model(token_ids)

        synchronize(device)

        start_time = time.perf_counter()

        for _ in range(MEASURED_RUNS):
            model(token_ids)

        synchronize(device)

    elapsed_time = time.perf_counter() - start_time
    seconds_per_run = elapsed_time / MEASURED_RUNS

    sequence_length = token_ids.shape[1]
    tokens_per_second = sequence_length / seconds_per_run

    return seconds_per_run, tokens_per_second


# Generates tokens greedily using KV-cached autoregressive decoding.
def generate_cached(model, prompt_ids, new_tokens):
    input_ids = prompt_ids

    with torch.inference_mode():
        logits, kv_cache = model(
            input_ids,
            use_cache=True,
        )

        next_token_id = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated_tokens = [next_token_id]

        for _ in range(new_tokens - 1):
            logits, kv_cache = model(
                next_token_id,
                kv_cache=kv_cache,
                use_cache=True,
            )

            next_token_id = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated_tokens.append(next_token_id)

    return torch.cat(generated_tokens, dim=1)


# Measures KV-cached decoding throughput for one attention backend.
def benchmark_generation(model, prompt_ids, device):
    for _ in range(GENERATION_WARMUP_RUNS):
        generate_cached(
            model,
            prompt_ids,
            GENERATION_TOKENS,
        )

    synchronize(device)

    measured_times = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(GENERATION_MEASURED_RUNS):
        synchronize(device)

        start_time = time.perf_counter()

        generate_cached(
            model,
            prompt_ids,
            GENERATION_TOKENS,
        )

        synchronize(device)

        measured_times.append(
            time.perf_counter() - start_time
        )

    average_time = sum(measured_times) / len(measured_times)
    tokens_per_second = GENERATION_TOKENS / average_time
    milliseconds_per_token = average_time / GENERATION_TOKENS * 1000

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


# Benchmarks handwritten attention against PyTorch SDPA using identical weights.
def main():
    torch.manual_seed(42)

    device = get_device()

    print_device_info(device)
    print(f"Checkpoint: {FINAL_CHECKPOINT}\n")

    manual_model, sdpa_model = load_models(device)

    print("Full-sequence inference\n")

    print(
        f"{'Tokens':>8} "
        f"{'Manual tok/s':>16} "
        f"{'SDPA tok/s':>16} "
        f"{'Speedup':>10}"
    )

    print("-" * 54)

    for sequence_length in SEQUENCE_LENGTHS:
        token_ids = torch.randint(
            0,
            VOCAB_SIZE,
            (1, sequence_length),
            device=device,
        )

        _, manual_tokens_per_second = benchmark_forward(
            manual_model,
            token_ids,
            device,
        )

        _, sdpa_tokens_per_second = benchmark_forward(
            sdpa_model,
            token_ids,
            device,
        )

        speedup = (
            sdpa_tokens_per_second
            / manual_tokens_per_second
        )

        print(
            f"{sequence_length:>8} "
            f"{manual_tokens_per_second:>16,.0f} "
            f"{sdpa_tokens_per_second:>16,.0f} "
            f"{speedup:>9.2f}x"
        )

    prompt_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (1, GENERATION_PROMPT_LENGTH),
        device=device,
    )

    with torch.inference_mode():
        manual_tokens = generate_cached(
            manual_model,
            prompt_ids,
            50,
        )

        sdpa_tokens = generate_cached(
            sdpa_model,
            prompt_ids,
            50,
        )

    if not torch.equal(manual_tokens, sdpa_tokens):
        raise RuntimeError(
            "Manual and SDPA backends produced different greedy tokens."
        )

    print("\nGeneration parity: PASSED (50 generated tokens)")

    print("\nKV-cached generation\n")
    print(f"Prompt length: {GENERATION_PROMPT_LENGTH}")
    print(f"Generated tokens: {GENERATION_TOKENS}")
    print(f"Measured runs: {GENERATION_MEASURED_RUNS}\n")

    manual_result = benchmark_generation(
        manual_model,
        prompt_ids,
        device,
    )

    sdpa_result = benchmark_generation(
        sdpa_model,
        prompt_ids,
        device,
    )

    speedup = (
        sdpa_result["tokens_per_second"]
        / manual_result["tokens_per_second"]
    )

    latency_reduction = (
        1
        - sdpa_result["milliseconds_per_token"]
        / manual_result["milliseconds_per_token"]
    ) * 100

    print(
        f"{'Backend':>12} "
        f"{'Time (s)':>10} "
        f"{'Tokens/s':>12} "
        f"{'ms/token':>12} "
        f"{'Peak GB':>10}"
    )

    print("-" * 62)

    print(
        f"{'Manual':>12} "
        f"{manual_result['time']:>10.3f} "
        f"{manual_result['tokens_per_second']:>12,.2f} "
        f"{manual_result['milliseconds_per_token']:>12.3f} "
        f"{manual_result['peak_memory_gb']:>10.3f}"
    )

    print(
        f"{'SDPA':>12} "
        f"{sdpa_result['time']:>10.3f} "
        f"{sdpa_result['tokens_per_second']:>12,.2f} "
        f"{sdpa_result['milliseconds_per_token']:>12.3f} "
        f"{sdpa_result['peak_memory_gb']:>10.3f}"
    )

    print(f"\nSDPA generation speedup: {speedup:.2f}x")
    print(f"SDPA latency reduction: {latency_reduction:.1f}%")
    print("\nAttention backend benchmark: PASSED")


if __name__ == "__main__":
    main()
