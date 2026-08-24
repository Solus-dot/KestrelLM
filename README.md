# KestrelLM

KestrelLM is a small decoder-only transformer language model implemented from scratch in PyTorch.

The project covers the complete language-model pipeline: training a custom BPE tokenizer, preprocessing text into token streams, implementing a causal Transformer, pretraining, checkpointing, evaluation, autoregressive sampling, and KV-cached inference.

The final model contains **29,628,928 parameters** and was pretrained on **600,014,848 TinyStories tokens**.

The pretrained inference checkpoint is published on Hugging Face as:

```text
SolusBolus/KestrelLM
```

`generate.py` automatically downloads the checkpoint when no local copy is available.

## Results

| Metric | Result |
| --- | ---: |
| Parameters | 29,628,928 |
| Training tokens | 600,014,848 |
| Optimizer steps | 36,622 |
| Final training loss | 1.5309 |
| Validation tokens | 4,690,944 |
| Validation loss | 1.5031 |
| Validation perplexity | 4.4957 |
| Context length | 512 |
| Vocabulary size | 8,192 |

The final validation metrics were measured over the complete held-out validation token stream rather than the smaller validation samples used during training.

## Architecture

KestrelLM is a decoder-only causal Transformer with the following configuration:

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
| Position representation | Learned positional embeddings |
| Attention | Multi-head causal self-attention |
| Output head | Tied with token embeddings |

Each Transformer block uses a pre-normalization residual architecture:

\[
H' = H + \operatorname{Attention}(\operatorname{RMSNorm}(H))
\]

\[
H'' = H' + \operatorname{SwiGLU}(\operatorname{RMSNorm}(H'))
\]

The attention implementation explicitly performs:

```text
hidden states
    ↓
Q / K / V projections
    ↓
split into attention heads
    ↓
scaled QKᵀ scores
    ↓
causal masking
    ↓
softmax
    ↓
weighted value aggregation
    ↓
combine heads
    ↓
output projection
```

The Transformer architecture itself does not use a prebuilt `nn.Transformer` or Hugging Face Transformer model.

## Tokenizer

KestrelLM uses a custom **8,192-token BPE tokenizer** trained on TinyStories using the Hugging Face `tokenizers` library.

Special tokens:

```text
<pad>
<unk>
<bos>
<eos>
```

Each story is terminated with `<eos>` before the tokenized stories are packed into a continuous training stream.

The preprocessed training and validation streams are stored as `uint16` binary files and accessed through NumPy memory mapping during training.

## Training

Training uses standard autoregressive next-token prediction.

For an input token sequence

\[
x_1,x_2,\ldots,x_T
\]

the model learns to predict

\[
x_2,x_3,\ldots,x_{T+1}.
\]

The objective is token-level cross-entropy:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p(y_i).
\]

Final training configuration:

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| Peak learning rate | \(1\times10^{-4}\) |
| Minimum learning rate | \(1\times10^{-5}\) |
| Warmup | 500 optimizer steps |
| LR schedule | Linear warmup + cosine decay |
| Adam \(\beta_1\) | 0.9 |
| Adam \(\beta_2\) | 0.95 |
| Weight decay | 0.1 |
| Gradient clipping | 1.0 |
| Physical batch size | 32 |
| Gradient accumulation | 1 |
| Effective batch size | 32 sequences |
| Context length | 512 |
| Tokens per optimizer step | 16,384 |
| Optimizer steps | 36,622 |
| Total tokens | 600,014,848 |

Parameters with two or more dimensions receive AdamW weight decay. One-dimensional parameters such as RMSNorm scales are excluded from weight decay.

The learning-rate schedule consists of linear warmup followed by cosine decay.

## Accelerator Benchmarking

Training began on Apple MPS and was later moved to an AMD Radeon RX 7900 XTX using ROCm.

Before continuing the run on the AMD GPU, several physical batch sizes were benchmarked while keeping the effective batch size fixed at 32:

| Physical batch | Gradient accumulation | Effective batch | Training throughput |
| ---: | ---: | ---: | ---: |
| 2 | 16 | 32 | 55,322 tokens/s |
| 4 | 8 | 32 | 66,520 tokens/s |
| 8 | 4 | 32 | 68,400 tokens/s |
| 16 | 2 | 32 | 67,208 tokens/s |
| 32 | 1 | 32 | **72,025 tokens/s** |

The final training run therefore used:

```text
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
```

while preserving the same effective batch size and number of tokens per optimizer update.

The final segment from approximately 160M to 600M tokens completed in about **1 hour 37 minutes** on the RX 7900 XTX.

## Checkpointing

Training checkpoints contain:

```text
model parameters
optimizer state
global optimizer step
data-pass counter
processed-token count
```

Checkpoint loading first maps tensors to CPU and then transfers the model and optimizer state to the selected accelerator.

This allowed the same training run to move from Apple MPS to AMD ROCm without restarting pretraining.

Checkpoint writes are performed atomically through a temporary file before replacing the rolling checkpoint.

Large training checkpoints are intentionally excluded from Git.

## Evaluation

After pretraining, the final model was evaluated over the complete validation set.

```text
Validation tokens:      4,690,944
Validation loss:        1.5031
Validation perplexity:  4.4957
```

Perplexity is:

\[
\operatorname{PPL}=e^{\mathcal{L}}
\]

so:

\[
e^{1.5031}\approx4.4957.
\]

The validation loss being close to the final training loss indicates no obvious train/validation divergence at the end of the run.

## Generation

KestrelLM supports autoregressive generation with:

- greedy decoding
- temperature sampling
- top-k sampling
- top-p / nucleus sampling
- deterministic random seeds
- `<eos>` stopping
- KV-cached decoding

Example:

```bash
uv run python src/generate.py \
    --prompt "Once upon a time" \
    --max-new-tokens 200 \
    --temperature 0.8
```

A fresh clone does not need a local model checkpoint.

If neither the exported model nor the original training checkpoint is present, `generate.py` automatically downloads:

```text
SolusBolus/KestrelLM
└── kestrel_30m.pt
```

from Hugging Face and stores it in the normal Hugging Face cache.

### Example generation

Prompt:

```text
Once upon a time
```

Output at temperature `0.8`:

> Once upon a time, there was a big, big lion. He lived in a jungle with his friends. One day, he saw a little bird with a hurt wing. The lion wanted to help the bird, but he couldn't find a bandage. The lion was sad and didn't know what to do.
>
> He asked his friends for help, but they didn't know the bird's name. Suddenly, he remembered a wise owl's advice. The owl told the lion to stay with his friends and help him. The lion did what the owl said.
>
> The lion was able to help the bird and he was very happy. His friends were very grateful for the lion's help. They learned that it's important to help others and that it's okay to ask for help. The lion and his friends lived happily ever after.

The model learns coherent TinyStories-style structure including characters, events, simple causal relationships, resolutions, and story endings.

## KV-Cached Inference

The original generator recomputed the entire visible sequence for every generated token.

Without a KV cache:

```text
prompt
  ↓
process all tokens
  ↓
generate one token
  ↓
process all tokens again
  ↓
generate another token
  ↓
...
```

KestrelLM implements a per-layer KV cache so previously calculated keys and values can be reused.

For each attention layer:

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

After the initial prompt prefill, subsequent decoding steps only need to calculate the new token's query, key, and value projections.

### Correctness

Cached and uncached inference were compared directly.

```text
Sequence length:          32
Prefill length:           8
Maximum logit difference: 0.00002861
KV-cache parity test:     PASSED
```

The small logit difference is consistent with normal floating-point variation.

The cached and uncached generation paths also produced identical greedy token sequences during parity testing.

### Inference performance

Measured on an AMD Radeon RX 7900 XTX using ROCm:

| Method | Throughput | Latency |
| --- | ---: | ---: |
| Full-context recomputation | 364.22 tokens/s | 2.746 ms/token |
| KV cache | **461.75 tokens/s** | **2.166 ms/token** |

This produced:

```text
KV-cache speedup:     1.27×
Latency reduction:    21.1%
```

The speedup is smaller than the theoretical reduction in attention computation because KestrelLM is only about 30M parameters. Single-token inference substantially underutilizes a high-end GPU, making kernel-launch and small-operation overhead increasingly significant.

The current educational KV-cache implementation also grows cached tensors using concatenation rather than a fully preallocated production-style cache.

## Prompt Processing Performance

Single-sequence prompt processing on the RX 7900 XTX:

| Prompt length | Time | Throughput |
| ---: | ---: | ---: |
| 32 | 2.394 ms | 13,366 tokens/s |
| 128 | 3.132 ms | 40,867 tokens/s |
| 256 | 3.088 ms | 82,908 tokens/s |
| 512 | 4.837 ms | 105,845 tokens/s |

## Exported Model

The original final training checkpoint is approximately 356 MB because it contains both the model and AdamW optimizer state.

`export_model.py` creates a smaller inference-only artifact containing:

```text
model_state_dict
architecture metadata
parameter count
training metadata
evaluation metadata
```

without optimizer state.

The exported FP32 checkpoint is approximately **118.5 MB**:

```text
kestrel_30m.pt
```

It is published on Hugging Face under:

```text
SolusBolus/KestrelLM
```

## Running KestrelLM

KestrelLM uses Python 3.12+ and `uv`.

Clone the repository and install dependencies:

```bash
git clone git@github.com:Solus-dot/KestrelLM.git
cd KestrelLM
uv sync
```

Generate text:

```bash
uv run python src/generate.py \
    --prompt "Once upon a time" \
    --max-new-tokens 200
```

The first generation run automatically downloads the pretrained model if no local checkpoint exists.

Subsequent runs reuse the Hugging Face cache.

### GPU backends

The model supports the normal PyTorch device backends:

```text
NVIDIA GPU  → CUDA
AMD GPU     → ROCm through the torch.cuda API
Apple GPU   → MPS
otherwise   → CPU
```

A default `uv sync` installs the normal PyTorch dependency for portability.

ROCm users should install the appropriate ROCm-enabled PyTorch build for their machine rather than expecting the default environment to provide AMD GPU support automatically.

## Training From Scratch

The pretrained checkpoint is not required if the objective is to reproduce training.

The general pipeline is:

```bash
uv run python src/download_dataset.py
uv run python src/train_tokenizer.py
uv run python src/preprocess_data.py
uv run python src/train.py
```

The large downloaded dataset, tokenized binary streams, checkpoints, results, and exported release artifacts are excluded from Git.

By default:

```python
RESUME_CHECKPOINT = None
```

so `train.py` starts a new run.

To continue an existing training run, configure:

```python
RESUME_CHECKPOINT = LATEST_CHECKPOINT
```

after placing a compatible checkpoint in the checkpoint directory.

## Full Validation Evaluation

`evaluate.py` performs evaluation over the complete tokenized validation stream:

```bash
uv run python src/evaluate.py
```

Unlike generation, full evaluation is not self-contained in a fresh clone.

It requires:

```text
data/tokenized/validation.bin
checkpoints/final_600m.pt
```

from a completed local training setup.

The published validation metrics are included above so users do not need the training dataset simply to inspect the model's reported results.

## Tests

Tokenizer tests:

```bash
PYTHONPATH=src uv run python src/tests/test_tokenizer.py
```

KV-cache correctness test:

```bash
PYTHONPATH=src uv run python src/tests/test_kv_cache.py
```

The KV-cache test requires the final local training checkpoint because it compares inference using the trained model.

## Benchmarks

Training throughput benchmark:

```bash
uv run python src/benchmark.py
```

Inference benchmark:

```bash
uv run python src/benchmark_inference.py
```

The inference benchmark compares:

```text
prompt processing
naive autoregressive decoding
KV-cached autoregressive decoding
```

and verifies generation parity before reporting performance.

## Project Structure

```text
KestrelLM/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── pyrightconfig.json
├── src/
│   ├── benchmark.py
│   ├── benchmark_inference.py
│   ├── config.py
│   ├── dataloader.py
│   ├── dataset.py
│   ├── download_dataset.py
│   ├── evaluate.py
│   ├── export_model.py
│   ├── generate.py
│   ├── model.py
│   ├── preprocess_data.py
│   ├── train.py
│   ├── train_tokenizer.py
│   └── tests/
│       ├── test_kv_cache.py
│       └── test_tokenizer.py
└── tokenizer/
    └── tokenizer.json
```

The following are intentionally local-only:

```text
data/
checkpoints/
results/
release/
```

## What Was Implemented

The project intentionally avoids using a prebuilt Transformer model implementation.

The following components are implemented directly within KestrelLM:

- learned token embeddings
- learned positional embeddings
- RMSNorm
- multi-head causal self-attention
- query, key, and value projections
- attention-head splitting and recombination
- scaled dot-product attention
- causal attention masking
- SwiGLU feed-forward networks
- residual Transformer blocks
- decoder stack
- tied token embedding / LM-head weights
- next-token cross-entropy training
- AdamW parameter grouping
- gradient accumulation
- gradient clipping
- linear warmup
- cosine learning-rate decay
- validation
- atomic checkpointing
- cross-device checkpoint resume
- greedy decoding
- temperature sampling
- top-k sampling
- top-p sampling
- EOS termination
- per-layer KV caching
- KV-cache correctness testing
- training and inference benchmarking
- inference-only model export
- automatic Hugging Face checkpoint download

PyTorch provides tensor operations, automatic differentiation, accelerator backends, and the AdamW optimizer.

## Dataset

KestrelLM was pretrained on **TinyStories**, a dataset of short synthetic stories designed to support language-model research at relatively small model scales.

The dataset is particularly suitable for KestrelLM because meaningful language generation can emerge at tens of millions of parameters rather than requiring a multi-billion-parameter model.

KestrelLM should therefore be understood as a small educational and experimental language model, not a general-purpose assistant or knowledge model.

## Scope and Limitations

KestrelLM was trained exclusively on TinyStories.

As a result, it is specialized for short English story-like text and should not be expected to provide:

- broad factual knowledge
- reliable question answering
- instruction-following behavior
- coding ability
- long-context reasoning
- general chatbot capabilities

The model uses learned positional embeddings with a maximum context length of 512 tokens.

The current KV cache prioritizes implementation clarity over maximum inference efficiency.

## Purpose

KestrelLM was built to make the internal mechanics of Transformer language models concrete by implementing and running the complete pipeline:

```text
raw text
    ↓
custom BPE tokenizer
    ↓
packed token stream
    ↓
memory-mapped training dataset
    ↓
token + positional embeddings
    ↓
causal multi-head attention
    ↓
SwiGLU Transformer blocks
    ↓
next-token logits
    ↓
cross-entropy loss
    ↓
backpropagation + AdamW
    ↓
600M-token pretraining run
    ↓
full validation evaluation
    ↓
autoregressive generation
    ↓
KV-cached inference
    ↓
published pretrained checkpoint
```

The result is a complete small language-model implementation that can be trained from scratch, resumed across accelerator backends, quantitatively evaluated, optimized, benchmarked, exported, published, and run from a fresh clone.
