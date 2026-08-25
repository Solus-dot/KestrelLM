import torch

from config import CHECKPOINT_DIR, VOCAB_SIZE
from model import KestrelLM


FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_600m.pt"

SEQUENCE_LENGTH = 32
PREFILL_LENGTH = 8


# Chooses the available accelerator for the parity test.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# Loads the final pretrained model without optimizer state.
def load_model(device):
    checkpoint = torch.load(
        FINAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model = KestrelLM()
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


# Compares ordinary full-sequence inference with prefill plus cached decoding.
def main():
    torch.manual_seed(42)

    device = get_device()
    model = load_model(device)

    token_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (1, SEQUENCE_LENGTH),
        device=device,
    )

    with torch.inference_mode():
        full_logits = model(token_ids)

        prefill_logits, kv_cache = model(
            token_ids[:, :PREFILL_LENGTH],
            use_cache=True,
        )

        cached_logits = [prefill_logits]

        for position in range(
            PREFILL_LENGTH,
            SEQUENCE_LENGTH,
        ):
            token_logits, kv_cache = model(
                token_ids[:, position:position + 1],
                kv_cache=kv_cache,
                use_cache=True,
            )

            cached_logits.append(token_logits)

        cached_logits = torch.cat(
            cached_logits,
            dim=1,
        )

    maximum_difference = (
        full_logits - cached_logits
    ).abs().max().item()

    torch.testing.assert_close(
        cached_logits,
        full_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    print(f"Device: {device}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Prefill length: {PREFILL_LENGTH}")
    print(f"Maximum logit difference: {maximum_difference:.8f}")
    print("KV-cache parity test: PASSED")


if __name__ == "__main__":
    main()
