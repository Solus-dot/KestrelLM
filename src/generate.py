import argparse

import torch
from tokenizers import Tokenizer

from config import CHECKPOINT_DIR, CONTEXT_LENGTH, EOS_TOKEN, TOKENIZER_PATH
from model import KestrelLM


FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_600m.pt"


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

    elif device.type == "mps":
        print("GPU backend: Apple MPS")

    else:
        print("GPU backend: CPU")


# Loads the custom BPE tokenizer used during pretraining.
def load_tokenizer():
    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

    return Tokenizer.from_file(str(TOKENIZER_PATH))


# Loads the final pretrained KestrelLM checkpoint for inference.
# Optimizer state is ignored because generation performs no training updates.
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

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    remove_mask = cumulative_probabilities > top_p

    # Shift the mask so the first token crossing the threshold is retained.
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    sorted_probabilities[remove_mask] = 0.0
    sorted_probabilities /= sorted_probabilities.sum()

    sampled_position = torch.multinomial(
        sorted_probabilities,
        num_samples=1,
    )

    return sorted_indices[sampled_position]


# Converts the model's next-token logits into one sampled token.
# Temperature 0 performs greedy decoding instead of random sampling.
def sample_next_token(logits, temperature, top_k, top_p):
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    logits = apply_top_k(logits, top_k)

    probabilities = torch.softmax(logits, dim=-1)

    return sample_top_p(probabilities, top_p)


# Generates tokens autoregressively from a text prompt.
# Only the most recent CONTEXT_LENGTH tokens are passed to the model.
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

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context_ids = generated_ids[-CONTEXT_LENGTH:]

            input_ids = torch.tensor(
                [context_ids],
                dtype=torch.long,
                device=device,
            )

            logits = model(input_ids)

            # Only the final position predicts the next token.
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


# Loads the final model and tokenizer, generates text, and prints the result.
def main():
    args = parse_arguments()
    validate_arguments(args)

    torch.manual_seed(args.seed)

    device = get_device()

    print_device_info(device)
    print(f"Checkpoint: {FINAL_CHECKPOINT}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Max new tokens: {args.max_new_tokens}\n")

    tokenizer = load_tokenizer()
    model = load_model(device)

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
