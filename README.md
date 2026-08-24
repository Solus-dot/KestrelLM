# KestrelLM

KestrelLM is a small decoder-only transformer language model built from scratch in PyTorch. It was implemented to understand the complete language-model pipeline: tokenization, transformer architecture, pretraining, evaluation, autoregressive generation, and KV-cached inference.

The final model has **29.63M parameters** and was pretrained on approximately **600M TinyStories tokens**.

## Results

| Metric | Result |
| --- | ---: |
| Parameters | 29,628,928 |
| Training tokens | 600,014,848 |
| Final training loss | 1.5309 |
| Validation tokens | 4,690,944 |
| Validation loss | 1.5031 |
| Validation perplexity | 4.4957 |
| Context length | 512 |
| Vocabulary size | 8,192 |

Full validation evaluation was performed over the complete held-out validation token stream.

## Architecture

KestrelLM is a decoder-only causal transformer with:

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
| Position encoding | Learned positional embeddings |
| Attention | Multi-head causal self-attention |
| Output head | Tied with token embeddings |

The main transformer block uses a pre-normalization residual architecture:

\[
H' = H + \operatorname{Attention}(\operatorname{RMSNorm}(H))
\]

\[
H'' = H' + \operatorname{SwiGLU}(\operatorname{RMSNorm}(H'))
\]

The attention implementation explicitly performs the query, key, and value projections, head splitting, scaled dot-product attention, causal masking, softmax, value aggregation, and output projection.

## Tokenization

KestrelLM uses a custom **8,192-token BPE tokenizer** trained on TinyStories with Hugging Face Tokenizers.

Special tokens:

```text
<pad>
<unk>
<bos>
<eos>
```

Stories are terminated with `<eos>` and packed into a continuous token stream for next-token prediction.

The preprocessed training and validation streams are stored as `uint16` binary files and memory-mapped during training.

## Training

The model was trained with next-token cross-entropy:

\[
\mathcal{L}
=
-\frac{1}{N}
\sum_{i=1}^{N}
\log p(y_i)
\]

Training configuration:

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
| Effective batch size | 32 sequences |
| Tokens per optimizer step | 16,384 |
| Optimizer steps | 36,622 |
| Total tokens | 600,014,848 |

Parameters with two or more dimensions receive AdamW weight decay, while one-dimensional parameters such as RMSNorm scales do not.

Training checkpoints contain the model parameters, optimizer state, optimizer step, and processed-token count, allowing training to resume across machines and accelerator backends.

## Evaluation

After pretraining, the final model was evaluated over the complete validation set:

```text
Validation tokens:      4,690,944
Validation loss:        1.5031
Validation perplexity:  4.4957
```

Perplexity is computed as:

\[
\operatorname{PPL}=e^{\mathcal{L}}
\]

giving:

\[
e^{1.5031}\approx4.4957
\]

## Generation

KestrelLM supports autoregressive text generation with:

- temperature sampling
- greedy decoding
- top-k sampling
- top-p / nucleus sampling
- EOS stopping
- deterministic sampling seeds
- KV-cached decoding

Example:

```bash
uv run python src/generate.py \
    --prompt "Once upon a time" \
    --max-new-tokens 200 \
    --temperature 0.8
```

Example output:

> Once upon a time, there was a big, big lion. He lived in a jungle with his friends. One day, he saw a little bird with a hurt wing. The lion wanted to help the bird, but he couldn't find a bandage. The lion was sad and didn't know what to do.
>
> He asked his friends for help, but they didn't know the bird's name. Suddenly, he remembered a wise owl's advice...

Despite its small size, the model learns coherent TinyStories-style structure, including characters, events, resolutions, and simple narrative relationships.

## KV Cache

The initial inference implementation recomputed the entire token history after every generated token.

KestrelLM later gained a per-layer KV cache that retains previously computed attention keys and values:

\[
K_{\text{cache}}
=
[K_{\text{past}};K_{\text{new}}]
\]

\[
V_{\text{cache}}
=
[V_{\text{past}};V_{\text{new}}]
\]

This allows subsequent decoding steps to process only the newly generated token.

Correctness was checked by comparing cached and uncached model logits:

```text
Sequence length:          32
Prefill length:           8
Maximum logit difference: 0.00002861
KV-cache parity test:     PASSED
```

### Inference benchmark

Measured on an AMD Radeon RX 7900 XTX using ROCm:

| Method | Throughput | Latency |
| --- | ---: | ---: |
| Full-context recomputation | 364.22 tokens/s | 2.746 ms/token |
| KV cache | 461.75 tokens/s | 2.166 ms/token |

The KV cache produced a:

- **1.27× decoding speedup**
- **21.1% reduction in per-token latency**

The relatively modest wall-clock speedup is expected for a ~30M parameter model on a high-end GPU: single-token decoding leaves much of the GPU underutilized, so kernel-launch and small-operation overhead become significant.

## Prompt Processing Performance

Measured on an AMD Radeon RX 7900 XTX:

| Prompt length | Throughput |
| ---: | ---: |
| 32 tokens | 13,366 tokens/s |
| 128 tokens | 40,867 tokens/s |
| 256 tokens | 82,908 tokens/s |
| 512 tokens | 105,845 tokens/s |

## Project Structure

```text
KestrelLM/
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── config.py
│   ├── model.py
│   ├── dataset.py
│   ├── dataloader.py
│   ├── train.py
│   ├── evaluate.py
│   ├── generate.py
│   ├── benchmark.py
│   ├── benchmark_inference.py
│   ├── download_dataset.py
│   ├── preprocess_data.py
│   ├── train_tokenizer.py
│   ├── data_stats.py
│   └── tests/
│       ├── test_tokenizer.py
│       └── test_kv_cache.py
├── tokenizer/
│   └── tokenizer.json
├── data/
│   └── tokenized/
└── checkpoints/
```

Training data and model checkpoints are intentionally excluded from Git.

## Running the Model

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Generate text:

```bash
uv run python src/generate.py \
    --prompt "Once upon a time" \
    --max-new-tokens 200
```

Run the full validation evaluation:

```bash
uv run python src/evaluate.py
```

Run the KV-cache correctness test:

```bash
PYTHONPATH=src uv run python src/tests/test_kv_cache.py
```

Run the inference benchmark:

```bash
uv run python src/benchmark_inference.py
```

## What Was Implemented From Scratch

The project intentionally avoids using a prebuilt transformer model implementation. The following components are implemented directly using PyTorch tensor operations and modules:

- learned token embeddings
- learned positional embeddings
- RMSNorm
- multi-head causal self-attention
- causal attention masking
- Q/K/V projections
- attention head splitting and recombination
- SwiGLU feed-forward networks
- residual transformer blocks
- decoder stack
- tied language-model head
- next-token cross-entropy training
- gradient accumulation
- gradient clipping
- AdamW parameter grouping
- linear warmup and cosine learning-rate decay
- checkpointing and cross-device resume
- autoregressive sampling
- top-k and top-p sampling
- KV-cached autoregressive decoding

PyTorch provides tensor operations, automatic differentiation, and the AdamW optimizer; the transformer architecture itself is implemented within the project.

## Dataset

KestrelLM is pretrained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

TinyStories consists of short synthetic stories using vocabulary and narrative structures designed to make language-model training feasible at relatively small model scales.

## Purpose

KestrelLM is an educational implementation intended to make the mechanics of transformer language models concrete.

Rather than starting with a high-level transformer library, the project follows the entire pipeline:

```text
raw text
    ↓
BPE tokenizer
    ↓
token stream
    ↓
training examples
    ↓
token + position embeddings
    ↓
causal transformer
    ↓
next-token logits
    ↓
cross-entropy loss
    ↓
backpropagation + AdamW
    ↓
pretrained language model
    ↓
autoregressive generation
```

The result is a complete small language model that can be trained, evaluated, checkpointed, resumed across accelerator backends, and used for text generation.
