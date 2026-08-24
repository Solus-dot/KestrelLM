import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from config import CHECKPOINT_DIR, CONTEXT_LENGTH, EOS_TOKEN, PROJECT_ROOT, TOKENIZER_PATH
from model import KestrelLM


HF_REPO_ID = "SolusBolus/KestrelLM"
HF_MODEL_FILENAME = "kestrel-30m.pt"

LOCAL_RELEASE_CHECKPOINT = PROJECT_ROOT / "release" / HF_MODEL_FILENAME
LOCAL_TRAINING_CHECKPOINT = CHECKPOINT_DIR / "final_600m.pt"


# Chooses an AMD/NVIDIA GPU first, Apple MPS second, and CPU otherwise.
# PyTorch exposes AMD ROCm GPUs through the torch.cuda interface.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Prints the accelerator backend and physical device used for generation.
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


# Loads the custom BPE tokenizer used during pretraining.
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    return Tokenizer.from_file(str(TOKENIZER_PATH))


# Finds a local model checkpoint when available.
# The smaller inference-only release checkpoint is preferred over the training checkpoint.
def find_local_checkpoint():
    if LOCAL_RELEASE_CHECKPOINT.exists():
        return LOCAL_RELEASE_CHECKPOINT

    if LOCAL_TRAINING_CHECKPOINT.exists():
        return LOCAL_TRAINING_CHECKPOINT

    return None


# Returns a usable checkpoint path.
# If no local checkpoint exists, the published model is downloaded and cached by Hugging Face.
def resolve_checkpoint():
    local_checkpoint = find_local_checkpoint()

    if local_checkpoint is not None:
        print(f"Using local checkpoint: {local_checkpoint}")
        return local_checkpoint

    print(f"Local checkpoint not found.")
    print(f"Downloading {HF_MODEL_FILENAME} from {HF_REPO_ID}...")

    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
    )

    return Path(downloaded_path)


# Loads the pretrained KestrelLM weights for inference.
# Optimizer state is unnecessary because generation performs no training updates.
def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = KestrelLM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


# Applies top-k filtering by keeping only the k highest-scoring tokens.
def apply_top_k(logits, top_k):
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits

    threshold = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


# Samples from the smallest high-probability token set whose cumulative
# probability reaches top_p. A value of 1.0 disables nucleus filtering.
def sample_top_p(probabilities, top_p):
    if top_p >= 1.0:
        return torch.multinomial(probabilities, num_samples=1)

    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

    remove_mask = cumulative_probabilities > top_p

    # Keeps the first token that crosses the top-p threshold.
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    sorted_probabilities[remove_mask] = 0.0
    sorted_probabilities /= sorted_probabilities.sum()

    sampled_position = torch.multinomial(sorted_probabilities, num_samples=1)

    return sorted_indices[sampled_position]


# Converts the model's next-token logits into one sampled token.
# Temperature 0 performs deterministic greedy decoding.
def sample_next_token(logits, temperature, top_k, top_p):
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)

    probabilities = torch.softmax(logits, dim=-1)

    return sample_top_p(probabilities, top_p)


# Processes the current visible context and creates a fresh KV cache.
# This is used once for the initial prompt and again only if the context fills up.
def prefill_context(model, token_ids, device):
    context_ids = token_ids[-CONTEXT_LENGTH:]

    input_ids = torch.tensor(
        [context_ids],
        dtype=torch.long,
        device=device,
    )

    logits, kv_cache = model(
        input_ids,
        use_cache=True,
    )

    return logits, kv_cache


# Generates text autoregressively while reusing each layer's cached keys and values.
# Once the 512-token context fills, the cache is rebuilt from the newest 512 tokens.
def generate(
    model,
    tokenizer,
    prompt,
    device,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
):
    prompt_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
    ).ids

    if not prompt_ids:
        raise ValueError("Prompt must contain at least one token.")

    generated_ids = list(prompt_ids)

    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)

    if eos_token_id is None:
        raise ValueError(f"EOS token not found in tokenizer: {EOS_TOKEN}")

    with torch.inference_mode():
        logits, kv_cache = prefill_context(
            model,
            generated_ids,
            device,
        )

        for _ in range(max_new_tokens):
            next_token_logits = logits[0, -1]

            next_token = sample_next_token(
                next_token_logits,
                temperature,
                top_k,
                top_p,
            )

            next_token_id = next_token.item()
            generated_ids.append(next_token_id)

            if next_token_id == eos_token_id:
                break

            cache_length = model.get_cache_length(kv_cache)

            if cache_length < CONTEXT_LENGTH:
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

            else:
                # Learned positional embeddings only cover positions 0-511.
                # Rebuild the cache from the newest context window once it is full.
                logits, kv_cache = prefill_context(
                    model,
                    generated_ids,
                    device,
                )

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


# Defines command-line options for prompts and sampling parameters.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text with the pretrained KestrelLM model."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text used to begin generation.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Keep only the top-k tokens. Use 0 to disable.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability. Use 1.0 to disable.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )

    return parser.parse_args()


# Validates generation settings before loading and running the model.
def validate_arguments(args):
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be greater than 0.")

    if args.temperature < 0:
        raise ValueError("temperature cannot be negative.")

    if args.top_k < 0:
        raise ValueError("top-k cannot be negative.")

    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be greater than 0 and at most 1.")


# Loads the pretrained model and tokenizer, generates text, and prints the result.
def main():
    args = parse_arguments()
    validate_arguments(args)

    torch.manual_seed(args.seed)

    device = get_device()

    print_device_info(device)

    checkpoint_path = resolve_checkpoint()

    print(f"Checkpoint: {checkpoint_path}")
    print("KV cache: enabled")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Max new tokens: {args.max_new_tokens}\n")

    tokenizer = load_tokenizer()
    model = load_model(checkpoint_path, device)

    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print("Generated text\n")
    print(generated_text)


if __name__ == "__main__":
    main()
