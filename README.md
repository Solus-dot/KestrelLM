# KestrelLM

KestrelLM is a decoder-only Transformer language model built from scratch in PyTorch.

The project covers the complete language-model pipeline: custom BPE tokenization, binary dataset preprocessing, causal Transformer implementation, pretraining, checkpointing, evaluation, autoregressive sampling, KV-cached inference, performance benchmarking, and controlled model-scaling experiments.

The primary released model, **Kestrel-M**, contains **29.6M parameters** and was pretrained on **600M TinyStories tokens**, reaching a validation perplexity of **4.50**.

Pretrained weights are published on Hugging Face under:

```text
SolusBolus/KestrelLM
```

A fresh clone can run generation without manually downloading the checkpoint.

## Results

### Released model

| Metric | Result |
| --- | ---: |
| Parameters | 29,628,928 |
| Training tokens | 600,014,848 |
| Optimizer steps | 36,622 |
| Final training loss | 1.5309 |
| Validation tokens | 4,690,944 |
| Validation loss | 1.5031 |
| Validation perplexity | **4.4957** |
| Context length | 512 |
| Vocabulary size | 8,192 |

The validation metrics were measured over the complete held-out validation token stream rather than the smaller validation samples used during training.

### Training performance

The final training configuration reached approximately **72K tokens/s** on an AMD Radeon RX 7900 XTX using ROCm.

The final segment of pretraining, from approximately 160M to 600M tokens, completed in about **1 hour 37 minutes**.

### Inference performance

With a 32-token prompt and 400 generated tokens on the RX 7900 XTX:

| Method | Throughput | Latency |
| --- | ---: | ---: |
| Full-context recomputation | 308.37 tokens/s | 3.243 ms/token |
| KV-cached decoding | **456.50 tokens/s** | **2.191 ms/token** |

This corresponds to:

```text
KV-cache speedup:   1.48×
Latency reduction:  32.4%
```

## Scaling Study

KestrelLM was width-scaled into three model sizes to study the effect of model capacity under a fixed training-token budget.

The experiment keeps the following constant:

- 6 Transformer layers
- 512-token context
- 64-dimensional attention heads
- 4× feed-forward expansion
- 8,192-token tokenizer
- TinyStories dataset
- AdamW optimizer
- batch size 32
- 16,384 tokens per optimizer step
- 6,104 optimizer steps
- 100,007,936 training tokens per model

Only model width changes.

| Model | Parameters | \(d_\text{model}\) | Heads | \(d_\text{ff}\) | Training tokens | Validation loss | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kestrel-S | 8,523,008 | 256 | 4 | 1,024 | 100,007,936 | 2.4405 | 11.4790 |
| Kestrel-M | 29,628,928 | 512 | 8 | 2,048 | 100,007,936 | 2.0780 | 7.9883 |
| Kestrel-L | 63,317,760 | 768 | 12 | 3,072 | 100,007,936 | **1.9178** | **6.8060** |

![KestrelLM width scaling](assets/scaling.png)

At the same 100M-token budget, increasing model width consistently improved held-out validation performance.

From Kestrel-S to Kestrel-L, parameter count increased by **7.43×** while perplexity fell from **11.4790 to 6.8060**, a reduction of approximately **40.7%**.

The experiment also shows diminishing returns: scaling from 8.5M to 29.6M parameters produced a larger improvement than scaling from 29.6M to 63.3M.

### Scaling cost

All three configurations were benchmarked at the same batch size and context length on an AMD Radeon RX 7900 XTX.

| Model | Parameters | Training throughput | Peak VRAM |
| --- | ---: | ---: | ---: |
| Kestrel-S | 8.52M | 181,232 tokens/s | 5.39 GB |
| Kestrel-M | 29.63M | 72,206 tokens/s | 8.86 GB |
| Kestrel-L | 63.32M | 40,779 tokens/s | 12.48 GB |

The result exposes the expected quality-compute tradeoff: larger models achieve lower validation loss under the same data budget but require more training time and accelerator memory.

The scaling study also provides a second comparison for Kestrel-M:

| Kestrel-M training budget | Validation loss | Perplexity |
| ---: | ---: | ---: |
| 100M tokens | 2.0780 | 7.9883 |
| 600M tokens | **1.5031** | **4.4957** |

This demonstrates improvement along two distinct scaling axes:

```text
More parameters at fixed training tokens  → lower validation loss
More training tokens at fixed parameters  → lower validation loss
```

## Architecture

KestrelLM uses a pre-normalized decoder-only Transformer.

The released Kestrel-M configuration is:

| Component | Configuration |
| --- | --- |
| Transformer layers | 6 |
| Model dimension | 512 |
| Attention heads | 8 |
| Head dimension | 64 |
| Feed-forward dimension | 2,048 |
| Context length | 512 |
| Vocabulary size | 8,192 |
| Normalization | RMSNorm |
| Feed-forward network | SwiGLU |
| Positional representation | Learned embeddings |
| Attention | Multi-head causal self-attention |
| LM head | Tied to token embeddings |

Each Transformer block computes:

\[
H' = H + \operatorname{Attention}(\operatorname{RMSNorm}(H))
\]

followed by:

\[
H'' = H' + \operatorname{SwiGLU}(\operatorname{RMSNorm}(H')).
\]

The feed-forward network is:

\[
\operatorname{SwiGLU}(H)
=
\left(
\operatorname{SiLU}(HW_g)
\odot
HW_u
\right)W_d.
\]

### Attention

The primary attention implementation is written directly in PyTorch rather than using `nn.Transformer`.

For each layer:

```text
hidden states
    ↓
Q / K / V projections
    ↓
split into attention heads
    ↓
scaled QKᵀ scores
    ↓
causal mask
    ↓
softmax
    ↓
weighted value aggregation
    ↓
combine heads
    ↓
output projection
```

For one attention head:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_h}} + M
\right)V
\]

where \(M\) is the causal mask.

PyTorch scaled dot-product attention is also implemented as an optional backend for controlled performance comparisons.

## Tokenizer and Dataset

KestrelLM is trained on **TinyStories**.

A custom Byte-Level BPE tokenizer is trained using the Hugging Face `tokenizers` library with a vocabulary size of 8,192.

Special tokens are:

```text
<pad>
<unk>
<bos>
<eos>
```

Each story receives an `<eos>` token before all tokenized stories are packed into continuous token streams.

The preprocessed dataset is stored as `uint16` binary files and accessed using NumPy memory mapping rather than loading the complete dataset into RAM.

Training samples are contiguous windows of \(T+1\) tokens:

\[
[x_1,x_2,\ldots,x_T,x_{T+1}]
\]

which are split into:

\[
X=[x_1,\ldots,x_T]
\]

and:

\[
Y=[x_2,\ldots,x_{T+1}].
\]

The model therefore learns standard next-token prediction.

## Training

The training objective is token-level cross-entropy:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p(y_i).
\]

The released 29.6M-parameter model used:

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| Peak learning rate | \(1\times10^{-4}\) |
| Minimum learning rate | \(1\times10^{-5}\) |
| Warmup | 500 steps |
| Schedule | Linear warmup + cosine decay |
| Adam \(\beta_1\) | 0.9 |
| Adam \(\beta_2\) | 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Physical batch size | 32 |
| Gradient accumulation | 1 |
| Effective batch size | 32 |
| Context length | 512 |
| Tokens per update | 16,384 |
| Training steps | 36,622 |
| Total tokens | 600,014,848 |

Parameters with two or more dimensions receive AdamW weight decay. One-dimensional parameters such as RMSNorm scales are excluded.

### Cross-device training

Pretraining initially ran on Apple MPS and was later migrated to an AMD Radeon RX 7900 XTX using ROCm.

Training checkpoints store:

```text
model parameters
AdamW optimizer state
global optimizer step
data-pass counter
processed-token count
```

Checkpoints are first loaded onto CPU before model and optimizer tensors are transferred to the active device.

This allowed the same training run to move between accelerator backends without restarting pretraining.

Checkpoint writes are atomic: a temporary checkpoint is written first and then replaces the rolling checkpoint.

## KV-Cached Inference

Naive autoregressive generation recomputes the complete visible sequence for every new token.

Without caching:

```text
process prompt
    ↓
generate token
    ↓
process prompt + generated token
    ↓
generate token
    ↓
process the entire sequence again
    ↓
...
```

KestrelLM instead stores each layer's previously computed keys and values:

\[
K_{\text{cache}}
=
[K_{\text{past}};K_{\text{new}}]
\]

\[
V_{\text{cache}}
=
[V_{\text{past}};V_{\text{new}}].
\]

After prompt prefill, each decoding step only computes projections for the newly generated token.

### Correctness

Cached and full-context inference were compared directly.

```text
Sequence length:           32
Prefill length:            8
Maximum logit difference:  0.00002861
KV-cache parity test:      PASSED
```

Greedy cached and uncached generation also produced identical token sequences.

### Long-horizon decode benchmark

For a 32-token prompt followed by 400 generated tokens:

| Method | Throughput | Latency | Peak memory |
| --- | ---: | ---: | ---: |
| Full-context recomputation | 308.37 tokens/s | 3.243 ms/token | 0.180 GB |
| KV cache | **456.50 tokens/s** | **2.191 ms/token** | 0.161 GB |

The KV cache provides a **1.48× decoding speedup** and a **32.4% latency reduction**.

Its advantage grows with sequence length because the uncached implementation repeatedly processes an increasingly large prefix.

## Inference Optimization Experiments

KestrelLM includes several controlled implementation experiments where an optimization was benchmarked rather than assumed to be beneficial.

### Dynamic vs preallocated KV cache

The original KV cache grows its key/value tensors using `torch.cat`.

A fixed-capacity preallocated implementation was also built and verified to reuse the same underlying storage throughout decoding.

At a 400-token decode horizon:

| KV-cache implementation | Throughput | Latency |
| --- | ---: | ---: |
| Dynamic concatenation | **456.50 tokens/s** | **2.191 ms/token** |
| Preallocated cache | 421.84 tokens/s | 2.371 ms/token |

Preallocation was approximately **8% slower** on this model and hardware, despite slightly reducing peak memory.

The dynamic implementation was therefore retained.

### Manual attention vs PyTorch SDPA

PyTorch scaled dot-product attention was implemented as an interchangeable backend using the same learned parameters.

Numerical parity was verified first:

```text
Manual vs SDPA full difference:    0.00002718
Manual vs SDPA cached difference:  0.00002503
SDPA full vs cached difference:    0.00002146
Attention backend parity test:     PASSED
```

Full-sequence throughput:

| Sequence length | Manual attention | SDPA | SDPA speedup |
| ---: | ---: | ---: | ---: |
| 32 | 13,485 tokens/s | 13,972 tokens/s | 1.04× |
| 128 | 41,156 tokens/s | 44,021 tokens/s | 1.07× |
| 256 | 83,960 tokens/s | 85,026 tokens/s | 1.01× |
| 512 | **106,145 tokens/s** | 98,545 tokens/s | 0.93× |

In a separate 400-token KV-cached generation benchmark:

| Backend | Throughput | Latency |
| --- | ---: | ---: |
| Manual | **479.73 tokens/s** | **2.084 ms/token** |
| SDPA | 464.96 tokens/s | 2.151 ms/token |

SDPA was about 3% slower during autoregressive decoding on the tested ROCm workload, so handwritten attention remains the default backend.

These experiments are intentionally retained as negative as well as positive results: an optimization is adopted only when measurement demonstrates an advantage for the target workload.

## Generation

KestrelLM supports:

- greedy decoding
- temperature sampling
- top-k sampling
- top-p / nucleus sampling
- deterministic random seeds
- `<eos>` stopping
- KV-cached autoregressive decoding

Example:

```bash
uv run python src/generate.py \
    --prompt "Once upon a time" \
    --max-new-tokens 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95
```

If no local checkpoint is available, `generate.py` automatically downloads:

```text
SolusBolus/KestrelLM
└── kestrel_30m.pt
```

through `huggingface_hub` and reuses the normal Hugging Face cache on subsequent runs.

### Example output

Prompt:

```text
Once upon a time
```

Example output at temperature 0.8:

> Once upon a time, there was a big, big lion. He lived in a jungle with his friends. One day, he saw a little bird with a hurt wing. The lion wanted to help the bird, but he couldn't find a bandage. The lion was sad and didn't know what to do.
>
> He asked his friends for help, but they didn't know the bird's name. Suddenly, he remembered a wise owl's advice. The owl told the lion to stay with his friends and help him. The lion did what the owl said.
>
> The lion was able to help the bird and he was very happy. His friends were very grateful for the lion's help. They learned that it's important to help others and that it's okay to ask for help. The lion and his friends lived happily ever after.

KestrelLM is trained only on TinyStories and is intended as a small language-model implementation and systems project, not as a general-purpose assistant.

## Installation

KestrelLM uses Python 3.12+ and `uv`.

```bash
git clone git@github.com:Solus-dot/KestrelLM.git
cd KestrelLM
uv sync
```

Run generation:

```bash
uv run python src/generate.py \
    --prompt "Once upon a time" \
    --max-new-tokens 200
```

The first generation automatically downloads the pretrained model when no local checkpoint exists.

## Training From Scratch

Download TinyStories:

```bash
PYTHONPATH=src uv run python scripts/download_dataset.py
```

Train the BPE tokenizer:

```bash
PYTHONPATH=src uv run python scripts/train_tokenizer.py
```

Preprocess the dataset:

```bash
PYTHONPATH=src uv run python scripts/preprocess_data.py
```

Train Kestrel-M:

```bash
uv run python src/train.py \
    --model medium \
    --steps 36622
```

The model-size choices are:

```text
small   → Kestrel-S   →  8.52M parameters
medium  → Kestrel-M   → 29.63M parameters
large   → Kestrel-L   → 63.32M parameters
```

Training can be assigned a separate checkpoint directory with `--run-name`.

For example:

```bash
uv run python src/train.py \
    --model large \
    --steps 6104 \
    --run-name experiments/kestrel_l
```

Pass `--resume` to continue from that run's rolling `latest.pt` checkpoint.

## Reproducing the Scaling Study

Each model is trained for exactly 6,104 optimizer steps:

\[
6104\times32\times512
=
100,007,936
\]

tokens.

```bash
uv run python src/train.py \
    --model small \
    --steps 6104 \
    --run-name scaling/kestrel_s

uv run python src/train.py \
    --model medium \
    --steps 6104 \
    --run-name scaling/kestrel_m

uv run python src/train.py \
    --model large \
    --steps 6104 \
    --run-name scaling/kestrel_l
```

The resulting checkpoints are stored under:

```text
checkpoints/scaling/
├── kestrel_s/
├── kestrel_m/
└── kestrel_l/
```

## Evaluation

`scripts/evaluate.py` evaluates a checkpoint over the complete validation token stream.

Released Kestrel-M:

```bash
PYTHONPATH=src uv run python scripts/evaluate.py \
    --checkpoint checkpoints/final_600m.pt
```

Scaling checkpoints:

```bash
PYTHONPATH=src uv run python scripts/evaluate.py \
    --checkpoint checkpoints/scaling/kestrel_s/final.pt

PYTHONPATH=src uv run python scripts/evaluate.py \
    --checkpoint checkpoints/scaling/kestrel_m/final.pt

PYTHONPATH=src uv run python scripts/evaluate.py \
    --checkpoint checkpoints/scaling/kestrel_l/final.pt
```

Full validation evaluation requires the local tokenized validation dataset:

```text
data/tokenized/validation.bin
```

The published validation results are included in this README so downloading the training dataset is not required merely to inspect the model's reported performance.

## Model Export

The original training checkpoint is approximately 356 MB because it contains both model weights and AdamW optimizer state.

The release exporter removes optimizer state and creates an inference-only checkpoint:

```bash
PYTHONPATH=src uv run python scripts/export_model.py
```

The resulting FP32 artifact is approximately 118.5 MB:

```text
release/kestrel_30m.pt
```

It contains:

```text
model_state_dict
architecture metadata
parameter count
training metadata
evaluation metadata
```

The exported checkpoint is the model published on Hugging Face.

## Tests

Tokenizer:

```bash
PYTHONPATH=src uv run python tests/test_tokenizer.py
```

KV-cache correctness:

```bash
PYTHONPATH=src uv run python tests/test_kv_cache.py
```

Manual-attention / SDPA parity:

```bash
PYTHONPATH=src uv run python tests/test_attention_backends.py
```

## Benchmarks

Training throughput and model-scaling benchmark:

```bash
PYTHONPATH=src uv run python benchmarks/training.py
```

Inference benchmark:

```bash
PYTHONPATH=src uv run python benchmarks/inference.py
```

Attention-backend benchmark:

```bash
PYTHONPATH=src uv run python benchmarks/attention.py
```

The benchmarks cover:

```text
training throughput
training VRAM
model-size scaling
prompt processing
naive autoregressive generation
KV-cached generation
manual attention
PyTorch SDPA
```

## Accelerator Backends

KestrelLM uses standard PyTorch device interfaces:

```text
NVIDIA GPU  → CUDA
AMD GPU     → ROCm through torch.cuda
Apple GPU   → MPS
otherwise   → CPU
```

The published Python environment remains portable and does not force a ROCm-specific PyTorch build.

On an AMD system, install the appropriate ROCm-enabled PyTorch wheel for the machine.

If PyTorch was manually replaced with a ROCm build inside a `uv` environment, commands can be run with:

```bash
uv run --no-sync ...
```

to avoid dependency synchronization replacing that installation.

The primary performance measurements in this README were collected on:

```text
AMD Radeon RX 7900 XTX
ROCm / HIP 7.2
```

## Project Structure

```text
KestrelLM/
├── assets/
│   └── scaling.png
│
├── benchmarks/
│   ├── attention.py
│   ├── inference.py
│   └── training.py
│
├── scripts/
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── preprocess_data.py
│   └── train_tokenizer.py
│
├── src/
│   ├── config.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── generate.py
│   ├── model.py
│   └── train.py
│
├── tests/
│   ├── test_attention_backends.py
│   ├── test_kv_cache.py
│   └── test_tokenizer.py
│
├── tokenizer/
│   └── tokenizer.json
│
├── .gitignore
├── LICENSE
├── pyproject.toml
├── pyrightconfig.json
├── README.md
└── uv.lock
```

Large or generated artifacts remain local:

```text
data/
checkpoints/
results/
release/
```

## Implemented Directly

The project intentionally avoids using a prebuilt Transformer model implementation.

The following components are implemented within KestrelLM:

- learned token embeddings
- learned positional embeddings
- RMSNorm
- multi-head causal self-attention
- Q/K/V projections
- attention-head splitting and recombination
- scaled dot-product attention
- causal masking
- SwiGLU feed-forward networks
- residual decoder blocks
- Transformer stack
- tied input/output embeddings
- autoregressive cross-entropy training
- AdamW parameter grouping
- gradient accumulation
- gradient clipping
- linear LR warmup
- cosine LR decay
- validation
- atomic checkpointing
- cross-device checkpoint resume
- greedy decoding
- temperature sampling
- top-k sampling
- top-p sampling
- EOS termination
- per-layer KV caching
- cached/full-context parity testing
- optional SDPA attention backend
- configurable model scaling
- training and inference benchmarks
- inference-only model export
- automatic Hugging Face checkpoint retrieval

PyTorch provides tensor operations, automatic differentiation, accelerator backends, and the AdamW optimizer.

## Scope

KestrelLM is an educational and experimental language-model implementation.

The objective is not to compete with production-scale language models. The project instead focuses on understanding and measuring the complete LM stack at a scale where architecture, training, inference, and systems behavior can all be implemented and studied directly.

The project demonstrates:

```text
model implementation
        +
pretraining
        +
evaluation
        +
inference optimization
        +
controlled experiments
        +
reproducible benchmarking
```

rather than treating model training as a single black-box API call.
