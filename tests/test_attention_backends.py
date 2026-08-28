import torch

from config import CHECKPOINT_DIR, VOCAB_SIZE
from model import KestrelLM


FINAL_CHECKPOINT = CHECKPOINT_DIR / "final.pt"

SEQUENCE_LENGTH = 32
PREFILL_LENGTH = 8


# Chooses the available accelerator for the attention-backend parity test.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Creates manual-attention and SDPA models with exactly the same trained weights.
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


# Runs token-by-token cached decoding and concatenates the logits from all positions.
def run_cached(model, token_ids):
    prefill_logits, kv_cache = model(
        token_ids[:, :PREFILL_LENGTH],
        use_cache=True,
    )

    cached_logits = [prefill_logits]

    for position in range(PREFILL_LENGTH, SEQUENCE_LENGTH):
        token_logits, kv_cache = model(
            token_ids[:, position:position + 1],
            kv_cache=kv_cache,
            use_cache=True,
        )

        cached_logits.append(token_logits)

    return torch.cat(cached_logits, dim=1)


# Verifies that manual attention and PyTorch SDPA produce numerically equivalent
# logits during both ordinary full-sequence inference and KV-cached decoding.
def main():
    torch.manual_seed(42)

    device = get_device()
    manual_model, sdpa_model = load_models(device)

    token_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (1, SEQUENCE_LENGTH),
        device=device,
    )

    with torch.inference_mode():
        manual_logits = manual_model(token_ids)
        sdpa_logits = sdpa_model(token_ids)

        manual_cached_logits = run_cached(
            manual_model,
            token_ids,
        )

        sdpa_cached_logits = run_cached(
            sdpa_model,
            token_ids,
        )

    full_difference = (
        manual_logits - sdpa_logits
    ).abs().max().item()

    cached_difference = (
        manual_cached_logits - sdpa_cached_logits
    ).abs().max().item()

    sdpa_cache_difference = (
        sdpa_logits - sdpa_cached_logits
    ).abs().max().item()

    torch.testing.assert_close(
        sdpa_logits,
        manual_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        sdpa_cached_logits,
        manual_cached_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        sdpa_cached_logits,
        sdpa_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    print(f"Device: {device}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Prefill length: {PREFILL_LENGTH}")
    print(f"Manual vs SDPA full difference: {full_difference:.8f}")
    print(f"Manual vs SDPA cached difference: {cached_difference:.8f}")
    print(f"SDPA full vs cached difference: {sdpa_cache_difference:.8f}")
    print("Attention backend parity test: PASSED")


if __name__ == "__main__":
    main()
