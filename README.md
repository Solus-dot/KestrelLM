# KestrelLM

KestrelLM is a small decoder-only Transformer language model implemented from scratch in PyTorch. The project covers the full language-model lifecycle rather than only the model architecture: custom ByteLevel BPE tokenization, streamed corpus construction, binary token packing, causal pretraining, checkpointing, evaluation, autoregressive generation, KV-cached inference, controlled systems experiments, supervised instruction tuning, and SafeTensors release.

The current target is **Kestrel-L**, a 63.3M-parameter model that will be pretrained from scratch on approximately **1.2B general-domain tokens** and then post-trained into **Kestrel-Instruct**. An earlier 29.6M TinyStories model remains useful as a historical baseline, but TinyStories is no longer part of the active training pipeline.

```text
general-domain corpus → custom BPE tokenizer → packed train/validation streams → Kestrel-L pretraining → Kestrel-Base → supervised instruction tuning → Kestrel-Instruct → SafeTensors release
```

## Current model family

All Kestrel variants use six decoder blocks, a 512-token context, 64-dimensional attention heads, RMSNorm, SwiGLU, learned positional embeddings, causal multi-head self-attention, and tied token-embedding / language-model-head weights. Width is the primary scaling dimension.

| Model | Parameters | d_model | Heads | d_head | d_ff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kestrel-S | 8,523,008 | 256 | 4 | 64 | 1,024 |
| Kestrel-M | 29,628,928 | 512 | 8 | 64 | 2,048 |
| Kestrel-L | 63,317,760 | 768 | 12 | 64 | 3,072 |

The forward path is:

```text
token IDs → token + learned positional embeddings → 6 × [RMSNorm → causal attention → residual → RMSNorm → SwiGLU → residual] → final RMSNorm → tied LM head → logits
```

Each block applies pre-normalized attention followed by a pre-normalized SwiGLU feed-forward network:

\[
H' = H + \mathrm{Attention}(\mathrm{RMSNorm}(H))
\]

\[
H'' = H' + \mathrm{SwiGLU}(\mathrm{RMSNorm}(H'))
\]

with

\[
\mathrm{SwiGLU}(H)=\left(\mathrm{SiLU}(HW_g)\odot HW_u\right)W_d.
\]

The reference attention implementation is handwritten in PyTorch rather than delegated to `nn.Transformer`. It performs explicit Q/K/V projections, head splitting, scaled \(QK^\top\), causal masking, softmax, value mixing, head recombination, and output projection. PyTorch SDPA is also available as an optional experimental backend.

## General-domain tokenizer and corpus

The active tokenizer is a custom **8,192-token ByteLevel BPE** trained from scratch on a representative sample of the new pretraining distribution. The tokenizer-training sample contains 100,000 streamed documents: 70,000 FineWeb-Edu-Dedup documents, 20,000 Cosmopedia v2 documents, and 10,000 FineWiki English documents.

The reserved vocabulary contains:

```text
<pad>  <unk>  <bos>  <eos>  <|system|>  <|user|>  <|assistant|>
```

The chat-role tokens are reserved before base pretraining so the same vocabulary and embedding matrix can later be reused during supervised instruction tuning.

The finalized pretraining mixture is measured after tokenization:

| Source | Share | Exact training tokens |
| --- | ---: | ---: |
| FineWeb-Edu-Dedup | 70% | 839,997,850 |
| Cosmopedia v2 | 20% | 239,999,386 |
| FineWiki English | 10% | 119,999,692 |
| **Total** | **100%** | **1,199,996,928** |

The source datasets are streamed rather than downloaded in full. Documents are deterministically shuffled, tokenized with the frozen Kestrel tokenizer, terminated with `<eos>`, and written until the source reaches its exact Kestrel-token quota.

```text
FineWeb-Edu-Dedup + Cosmopedia v2 + FineWiki → streamed documents → Kestrel BPE → <eos> boundaries → uint16 token stream
```

The resulting local files are `data/tokenized/train.bin` and `data/tokenized/validation.bin`. Token IDs are stored as `uint16` because the vocabulary is far below 65,536 entries. `BinaryTokenDataset` memory-maps these files and returns fixed-length next-token-prediction examples, so the core training code is independent of the original source corpus.

## Pretraining target

The main Kestrel-L run is designed around a fixed batch and exact token budget:

| Setting | Value |
| --- | ---: |
| Physical batch size | 32 |
| Context length | 512 |
| Gradient accumulation | 1 |
| Effective batch size | 32 |
| Tokens per optimizer step | 16,384 |
| Optimizer steps | 73,242 |
| Total training tokens | 1,199,996,928 |
| Optimizer | AdamW |
| Peak learning rate | \(1\times10^{-4}\) |
| Minimum learning rate | \(1\times10^{-5}\) |
| Warmup | 500 steps |
| Schedule | Linear warmup + cosine decay |
| Adam β₁ / β₂ | 0.9 / 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |

The exact training budget is

\[
73{,}242 \times 32 \times 512 = 1{,}199{,}996{,}928.
\]

Training checkpoints remain trusted local `.pt` files because they contain model weights, optimizer state, counters, processed-token totals, and model configuration required for exact resume. Public inference artifacts are exported separately as SafeTensors.

## Hardware scaling benchmark

Training throughput and peak allocated VRAM were measured on an **AMD Radeon RX 7900 XTX** through ROCm at batch size 32 and context length 512.

| Model | Seconds/step | Throughput | Peak VRAM |
| --- | ---: | ---: | ---: |
| Kestrel-S | 0.091 | 180,876 tok/s | 5.39 GB |
| Kestrel-M | 0.227 | 72,049 tok/s | 8.86 GB |
| Kestrel-L | 0.402 | 40,800 tok/s | 12.48 GB |

All three variants fit at the same physical batch size. Keeping the batch fixed makes scaling comparisons cleaner because every optimizer update contains the same 16,384 tokens. At the measured Kestrel-L throughput, 1.2B tokens correspond to roughly 8.2 hours of pure training compute before validation and checkpoint overhead.

## KV-cached inference

KestrelLM supports autoregressive generation with dynamic per-layer KV caches. After prompt prefill, each decoding step computes projections only for the newly generated token while reusing previous keys and values.

```text
prompt prefill → cached K/V + next-token logits → generate one token → append new K/V → repeat
```

For the earlier Kestrel-M model with a 32-token prompt and 400 generated tokens:

| Method | Throughput | Latency |
| --- | ---: | ---: |
| Full-context recomputation | 308.37 tok/s | 3.243 ms/token |
| Dynamic KV cache | **456.50 tok/s** | **2.191 ms/token** |

Dynamic KV caching therefore produced a **1.48× speedup** and **32.4% lower token latency**.

A fixed-capacity preallocated KV cache was also implemented and verified, but it measured **421.84 tok/s** at the same 400-token horizon, approximately 8% slower than the dynamic implementation. The dynamic cache was therefore retained.

## Manual attention vs SDPA

PyTorch scaled dot-product attention was added as an interchangeable backend and numerically compared against the handwritten implementation. Full-sequence and cached-logit parity passed with maximum differences on the order of \(10^{-5}\).

| Sequence length | Manual | SDPA | SDPA / Manual |
| ---: | ---: | ---: | ---: |
| 32 | 13,485 tok/s | 13,972 tok/s | 1.04× |
| 128 | 41,156 tok/s | 44,021 tok/s | 1.07× |
| 256 | 83,960 tok/s | 85,026 tok/s | 1.01× |
| 512 | **106,145 tok/s** | 98,545 tok/s | 0.93× |

For 400-token cached decoding, manual attention reached **479.73 tok/s** while SDPA reached **464.96 tok/s**. Manual attention therefore remains the default for the measured ROCm workload, with SDPA retained as an optional comparison backend.

## Post-training plan

After general-domain base pretraining, Kestrel-Base will be supervised-fine-tuned into Kestrel-Instruct. The first baseline will use **full-parameter SFT** rather than LoRA because the 63.3M-parameter model fits comfortably on the available GPU. PEFT can later be evaluated as a controlled memory/throughput experiment rather than being required for feasibility.

A conversation is serialized using the reserved role tokens:

```text
<|system|> system message <|user|> user request <|assistant|> assistant response <eos>
```

The entire conversation is visible to the model, but loss is applied only to assistant-response targets. System/user/control positions use `-100` labels so PyTorch cross entropy ignores them.

```text
system/user tokens: no supervised loss → assistant tokens: supervised next-token loss → full-parameter update
```

The exact SFT corpus will be selected separately with attention to quality, licensing, breadth, and evaluation contamination.

## SafeTensors release standard

Training state and public inference state are deliberately separated:

```text
trusted resumable training checkpoint = .pt   |   public inference weights = .safetensors
```

The intended release layout is:

```text
release/kestrel-base/     → model.safetensors | config.json | tokenizer.json
release/kestrel-instruct/ → model.safetensors | config.json | tokenizer.json
```

Because KestrelLM ties the input embedding and LM-head weights, export uses the model-aware SafeTensors save/load helpers so shared tensor storage is represented correctly. A release is validated by loading the SafeTensors weights into a fresh model with the exact recorded architecture before publication.

## Legacy TinyStories baseline

The original Kestrel-M model was trained on TinyStories as an implementation proof before the project moved to general-domain pretraining. That historical run remains useful for comparison but is not part of the active corpus pipeline.

| Metric | Legacy Kestrel-M |
| --- | ---: |
| Parameters | 29,628,928 |
| Training tokens | 600,014,848 |
| Optimizer steps | 36,622 |
| Final training loss | 1.5309 |
| Validation tokens | 4,690,944 |
| Validation loss | 1.5031 |
| Validation perplexity | 4.4957 |

The legacy model demonstrated that the architecture, training loop, checkpointing, evaluation, generation, and KV cache worked end to end. The current general-domain Kestrel-L run supersedes it as the primary model-development path.

## Repository structure

```text
src/
  config.py
  dataset.py
  dataloader.py
  model.py
  train.py
  generate.py

scripts/
  train_tokenizer.py
  preprocess_data.py
  evaluate.py
  export_model.py

benchmarks/
  training.py
  inference.py
  attention.py

tests/
  test_tokenizer.py
  test_kv_cache.py
  test_attention_backends.py
```

The core `src/` directory is intentionally small. Corpus acquisition and preprocessing live in `scripts/`; performance experiments live in `benchmarks/`; correctness checks live in `tests/`.

## Environment

KestrelLM uses Python 3.12+ and `uv`. Standard PyTorch device interfaces are used for NVIDIA CUDA, AMD ROCm through `torch.cuda`, Apple MPS, and CPU.

On systems where ROCm PyTorch is installed manually into the project environment, use `uv run --no-sync ...` for execution so dependency synchronization does not replace the ROCm build with a portable/CPU PyTorch wheel.

## Current status

The Transformer implementation, S/M/L parameterization, training loop, checkpoint/resume path, generation code, dynamic KV caching, optional SDPA backend, scaling benchmark, new general-domain tokenizer, and finalized pretraining mixture are in place. The current work is transitioning the repository fully away from the old TinyStories preprocessing path, generating the new packed 1.2B-token corpus, pretraining Kestrel-L, evaluating Kestrel-Base, then implementing full supervised instruction tuning and SafeTensors release.
