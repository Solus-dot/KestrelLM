# KestrelLM Pipeline

## Summary: what to run, in order

The project is meant to be executed as a sequence of artifacts: first freeze the tokenizer, then build the token corpus with that tokenizer, then pretrain and evaluate Kestrel-Base, then post-train and evaluate Kestrel-Instruct, and finally export inference-only SafeTensors releases. On the ROCm homelab, use `uv run --no-sync` so `uv` does not replace the manually installed ROCm PyTorch build, and use `PYTHONPATH=src` because the standalone scripts import modules from `src/` directly.

| Step | File / command | Why it runs here |
| --- | --- | --- |
| 1. Train tokenizer | `PYTHONPATH=src uv run --no-sync python scripts/train_tokenizer.py` | Learns the frozen 8,192-token ByteLevel BPE vocabulary from the 70/20/10 general-domain sample. This is run once unless the tokenizer is intentionally changed. |
| 2. Verify tokenizer | `PYTHONPATH=src uv run --no-sync python tests/test_tokenizer.py` | Confirms exact encode/decode round-trip, vocabulary size, and all seven reserved token IDs before a large corpus is encoded with it. |
| 3. Build pretraining corpus | `PYTHONPATH=src uv run --no-sync python scripts/preprocess_data.py` | Streams FineWeb-Edu-Dedup, Cosmopedia v2, and FineWiki through the frozen tokenizer and writes exact `train.bin` and `validation.bin` token budgets. Any tokenizer change requires regenerating these files. |
| 4. Inspect corpus | inspect `data/tokenized/train.bin` and `data/tokenized/validation.bin` | Confirms exact token counts and decodes a few windows before spending GPU time on training. This is a data-integrity gate rather than a separate pipeline script. |
| 5. Optional hardware benchmark | `PYTHONPATH=src uv run --no-sync python benchmarks/training.py` | Rechecks Kestrel-L throughput and VRAM only when hardware, PyTorch/ROCm, batch size, or architecture changes. It is not required before every training run. |
| 6. Pretrain Kestrel-Base | `PYTHONPATH=src uv run --no-sync python src/train.py --model large --steps 73242 --run-name kestrel_base_1p2b` | Trains Kestrel-L from scratch for exactly 73,242 optimizer steps = 1,199,996,928 tokens and produces resumable `.pt` checkpoints plus `final.pt`. |
| 7. Evaluate Kestrel-Base | `PYTHONPATH=src uv run --no-sync python scripts/evaluate.py --checkpoint checkpoints/kestrel_base_1p2b/final.pt` | Measures held-out validation loss/perplexity before instruction tuning so base-model quality is separated from post-training effects. |
| 8. Supervised instruction tuning | SFT preprocessing/training files are the next implementation stage | Serializes conversations with the reserved role tokens and trains from Kestrel-Base using assistant-only loss to produce Kestrel-Instruct. |
| 9. Evaluate Kestrel-Instruct | instruction evaluation path to be finalized with the SFT stage | Measures instruction-following behavior separately from base language-model evaluation. |
| 10. Export releases | `PYTHONPATH=src uv run --no-sync python scripts/export_model.py --checkpoint <checkpoint> --stage base` or `--stage instruct` | Converts trusted `.pt` training state into inference-only `model.safetensors`, `config.json`, and `tokenizer.json`, then reload-validates the exported model. |

The dependency chain is therefore:

```text
train_tokenizer.py → test_tokenizer.py → preprocess_data.py → inspect token binaries → train.py → evaluate.py → SFT → instruct evaluation → export_model.py
```

The tokenizer and tokenized corpus are upstream dependencies: changing either invalidates every downstream model trained from them. Base evaluation happens before SFT so the effect of post-training remains measurable, and SafeTensors export happens last because public inference artifacts intentionally omit optimizer/resume state.

---

## 1. Objective

KestrelLM is a small decoder-only Transformer implemented from scratch in PyTorch. The project is no longer centered on a narrow story-generation corpus; the target is now a complete small-language-model lifecycle in which a general-domain base model is pretrained from raw text, evaluated as a language model, post-trained into an instruction-following model, and finally released in a standard inference format. The intended public artifacts are **Kestrel-Base**, representing the pretrained model before chat alignment, and **Kestrel-Instruct**, representing the same architecture after supervised instruction tuning.

The complete flow is deliberately linear and reproducible:

```text
FineWeb-Edu-Dedup + Cosmopedia v2 + FineWiki → general-domain BPE tokenizer → streamed token-budgeted corpus → packed uint16 train.bin / validation.bin → Kestrel-L pretraining → Kestrel-Base evaluation → supervised instruction tuning → Kestrel-Instruct evaluation → SafeTensors export
```

Training state and public release state are separate concerns throughout the pipeline. Resumable training checkpoints remain trusted local PyTorch `.pt` files because they contain optimizer state and counters, while public model releases use `model.safetensors` plus explicit JSON configuration and `tokenizer.json`.

---

## 2. Data design

The pretraining distribution is fixed at **70% FineWeb-Edu-Dedup, 20% Cosmopedia v2, and 10% FineWiki English**. FineWeb-Edu-Dedup supplies broad educational web text and therefore carries most of the linguistic and factual diversity. Cosmopedia v2 adds dense explanatory, textbook-like, and structured synthetic prose. FineWiki contributes a cleaner factual and entity-heavy distribution so that encyclopedic knowledge is not left entirely to web text.

| Source | Hugging Face dataset/config | Training share | Exact training-token quota |
| --- | --- | ---: | ---: |
| FineWeb-Edu-Dedup | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | 70% | 839,997,850 |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 20% | 239,999,386 |
| FineWiki English | `HuggingFaceFW/finewiki`, `en` | 10% | 119,999,692 |
| **Total** |  | **100%** | **1,199,996,928** |

The main run therefore contains exactly 73,242 * 16,384 = 1,199,996,928 training tokens. The dataset mixture is enforced **after Kestrel tokenization**, not by document count, byte count, word count, or an upstream tokenizer estimate. This matters because the three corpora have different document-length and lexical distributions.

The upstream datasets are not downloaded in full. Each source is opened through Hugging Face `datasets` in streaming mode, deterministically shuffled, read document by document, encoded with the frozen Kestrel tokenizer, terminated with `<eos>`, and written until that source reaches its exact token quota.

```text
remote source stream → deterministic shuffle → non-empty document → Kestrel BPE encode → append <eos> → uint16 IDs → stop at exact quota
```

A separate held-out validation stream follows the same 70/20/10 mixture. The planned validation budget is 10,000,000 tokens: 7,000,000 from FineWeb-Edu-Dedup, 2,000,000 from Cosmopedia v2, and 1,000,000 from FineWiki. For each source, validation documents are consumed first and training then continues from the **same source iterator**, preventing the same streamed documents from immediately appearing in both local splits.

The resulting files are `data/tokenized/train.bin` and `data/tokenized/validation.bin`. Because the vocabulary has only 8,192 entries, token IDs are stored as `uint16`; the 1.2B-token training stream therefore occupies roughly 2.4 GB before filesystem overhead. These files are the only data representation the training loop needs.

---

## 3. Tokenizer

KestrelLM uses a custom **ByteLevel BPE tokenizer with vocabulary size 8,192**. The old domain-specific tokenizer was discarded because a vocabulary learned from a narrow corpus would waste capacity and over-fragment general educational, factual, scientific, and encyclopedic text. The replacement tokenizer was trained from scratch on 100,000 streamed documents sampled in the same 70/20/10 proportions as pretraining: 70,000 FineWeb-Edu-Dedup documents, 20,000 Cosmopedia v2 documents, and 10,000 FineWiki English documents.

```text
70k FineWeb-Edu + 20k Cosmopedia + 10k FineWiki → ByteLevel preprocessing → BPE merge learning → 8,192-token tokenizer.json
```

Byte-level preprocessing guarantees representational coverage for arbitrary text even when a whole word is absent from the learned vocabulary. The tokenizer reserves seven control tokens from the beginning:

```text
<pad>  <unk>  <bos>  <eos>  <|system|>  <|user|>  <|assistant|>
```

The three chat-role tokens are intentionally reserved **before base pretraining**. This prevents post-training from requiring a vocabulary expansion and embedding-matrix resize. Pretraining documents do not need to use the chat tokens; they are simply present in the vocabulary so the same tokenizer and embedding dimensions survive the entire base-to-instruct pipeline.

Every pretraining document receives `<eos>` after encoding. The packed stream therefore retains document boundaries even though documents are concatenated into one continuous token file:

```text
document A <eos> document B <eos> document C <eos> document D <eos> ...
```

The tokenizer-training process successfully reported an 8,192-token vocabulary and saved `tokenizer/tokenizer.json`. The Python process later emitted a `PyGILState_Release` fatal error during interpreter finalization, after the successful save message; this appears to be a shutdown problem in the native/streaming stack rather than a tokenizer-training failure. The saved tokenizer was subsequently loaded successfully: exact encode/decode round-trip passed, vocabulary size was 8,192, and all seven reserved tokens were present with unique IDs 0–6.

---

## 4. Binary training dataset

`BinaryTokenDataset` is intentionally domain-agnostic. It knows nothing about FineWeb, Cosmopedia, FineWiki, chat messages, or the original document structure. It memory-maps a flat token file and creates fixed-length causal language-model examples. With context length \(T=512\), each sample reads \(T+1=513\) contiguous IDs and returns the first 512 as the input and the final 512 as the one-token-shifted target.

```text
[token0 token1 ... token511 token512] → X=[token0 ... token511] , Y=[token1 ... token512]
```

Samples are spaced by one context window rather than generated as every possible overlapping window. PyTorch receives `torch.long` tensors because embedding layers require integer index tensors. The DataLoader merely batches these samples; the upstream source mixture has already been resolved during preprocessing.

This separation is important:

```text
source-specific logic lives in preprocessing → dataset.py only sees token IDs → dataloader.py only batches tensors → train.py only sees (X,Y)
```

Changing the pretraining corpus therefore does not require changing the core dataset abstraction.

---

## 5. Model family

KestrelLM supports three width-scaled decoder-only Transformer variants. Depth, context length, and head dimension remain fixed while residual width, head count, and feed-forward width scale together.

| Model | Parameters | Layers | \(d_\text{model}\) | Heads | \(d_\text{head}\) | \(d_\text{ff}\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kestrel-S | 8,523,008 | 6 | 256 | 4 | 64 | 1,024 |
| Kestrel-M | 29,628,928 | 6 | 512 | 8 | 64 | 2,048 |
| Kestrel-L | 63,317,760 | 6 | 768 | 12 | 64 | 3,072 |

The general-domain target is **Kestrel-L**. Kestrel-M remains useful as the historical baseline and default-compatible architecture, while Kestrel-S is primarily useful for controlled scaling and fast experiments.

The forward path is:

```text
token IDs → token embeddings + learned positional embeddings → 6 × [RMSNorm → causal MHA → residual → RMSNorm → SwiGLU → residual] → final RMSNorm → tied LM head → vocabulary logits
```

Attention is implemented manually with separate \(Q\), \(K\), and \(V\) projections, head splitting, scaled \(QK^\top/\sqrt{d_h}\), causal masking, softmax, value mixing, head recombination, and the output projection. An optional PyTorch SDPA backend exists as an experimental comparison path, but the handwritten attention remains the reference/default implementation because controlled ROCm measurements did not show a decoding advantage for SDPA.

The feed-forward network is SwiGLU:

\[
\operatorname{FFN}(H)=\left(\operatorname{SiLU}(HW_g)\odot HW_u\right)W_d
\]

and each Transformer block is pre-norm. Token embeddings and the LM-head projection are tied, reducing parameter count while keeping the same vocabulary representation at the input and output. Learned positional embeddings cover positions 0–511, giving the model a context length of 512.

---

## 6. Base pretraining

Kestrel-Base is trained with ordinary causal next-token prediction. Given \(x_0,\ldots,x_T\), the model sees \(x_0,\ldots,x_{T-1}\) and predicts \(x_1,\ldots,x_T\). Cross entropy is computed at every target position.

```text
packed token stream → 512-token X → Kestrel-L → logits for every position → cross entropy against one-token-shifted Y → AdamW update
```

The current optimizer/training configuration is physical batch size 32, context length 512, gradient accumulation 1, effective batch size 32 sequences, and therefore 16,384 tokens per optimizer update. AdamW uses \(\beta_1=0.9\), \(\beta_2=0.95\), weight decay 0.1, gradient clipping at 1.0, peak learning rate \(10^{-4}\), minimum learning rate \(10^{-5}\), a 500-step linear warmup, and cosine decay thereafter. Matrix-like parameters receive weight decay while one-dimensional normalization scales do not.

On the AMD Radeon RX 7900 XTX through ROCm, the width-scaling training benchmark at batch 32 and context 512 produced:

| Model | Seconds/step | Throughput | Peak allocated VRAM |
| --- | ---: | ---: | ---: |
| Kestrel-S | 0.091 | 180,876 tok/s | 5.39 GB |
| Kestrel-M | 0.227 | 72,049 tok/s | 8.86 GB |
| Kestrel-L | 0.402 | 40,800 tok/s | 12.48 GB |

Kestrel-L therefore fits comfortably without reducing the physical batch. At the measured 40.8K tokens/s, 1.2B tokens correspond to roughly 8.2 hours of pure training compute, before validation and checkpoint overhead.

Training checkpoints remain `.pt` because they are trusted resumable state rather than public artifacts. A checkpoint contains the model state dict, optimizer state dict, global optimizer step, processed-token count, data-pass/epoch progress, and model configuration. A run should use an isolated directory such as `checkpoints/kestrel_base_1p2b/` with `latest.pt`, periodic milestone checkpoints, and `final.pt`.

```text
training step → latest.pt for resume → periodic step_N.pt milestones → completed run → final.pt
```

---

## 7. Base-model evaluation

Base-model evaluation must distinguish **knowledge/language modeling** from **instruction following**. Before SFT, Kestrel-Base is expected to behave primarily as a continuation model. Intrinsic evaluation therefore begins with held-out cross-entropy and perplexity on the general-domain validation stream. Factual and commonsense capability can then be probed using completion-style prompts such as `The capital of France is`, `Photosynthesis is the process by which`, or `The largest planet in the Solar System is`.

```text
Kestrel-Base → held-out loss/perplexity + factual completion + commonsense completion + simple reading/reasoning probes
```

A base model may contain the information required to answer a question while still responding poorly to imperative or conversational prompts. That behavior is not necessarily a pretraining failure; it is the reason the project has a separate post-training stage. Benchmark selection should be frozen before SFT data selection so evaluation examples are not accidentally introduced into the post-training set.

---

## 8. Post-training into Kestrel-Instruct

The first post-training baseline should be **full supervised fine-tuning**, not PEFT. Kestrel-L is only about 63.3M parameters and fits easily on the available GPU, so there is no systems requirement to use LoRA merely to make SFT feasible. Full SFT establishes a clean reference; PEFT/LoRA can later be added as a controlled comparison measuring memory, throughput, convergence, and instruction-following quality.

Pretraining teaches the model to model text distributions and acquire factual/statistical structure; SFT teaches it how to map explicit prompts to useful responses.

```text
Kestrel-Base weights → formatted instruction conversations → assistant-only supervised loss → full-parameter updates → Kestrel-Instruct
```

The reserved role tokens define the conversation format:

```text
<|system|> system message <|user|> user request <|assistant|> assistant response <eos>
```

For multi-turn examples, user and assistant spans repeat in sequence. The model sees the entire serialized conversation as context, but the loss should be applied only to assistant-response targets. System, user, and other prompt/control positions receive the ignore label `-100`, which PyTorch cross entropy skips. Assistant tokens and the assistant-ending `<eos>` receive ordinary token IDs as targets.

```text
system span: ignore loss → user span: ignore loss → assistant span: compute loss → assistant <eos>: compute loss
```

The exact SFT corpus should be chosen separately from pretraining. Quality, breadth, licensing, message-role structure, and benchmark contamination matter more than maximizing raw example count for a 63M model. Once a full-SFT baseline exists, a LoRA/PEFT branch becomes an experiment rather than an architectural dependency.

---

## 9. Inference and KV caching

Generation supports greedy decoding and stochastic sampling with temperature, top-k, top-p, and EOS stopping. Autoregressive inference uses per-layer KV caches so previously computed keys and values are reused rather than recomputed from the full visible context on every token.

```text
prompt → one full prefill → KV cache + first logits → one new token → append only new K/V → next token → ... → context limit
```

For the earlier 29.6M Kestrel-M model with a 32-token prompt and 400 generated tokens, naive generation measured 308.37 tok/s while dynamic KV-cached decoding measured 456.50 tok/s, a 1.48× throughput improvement and 32.4% latency reduction. A preallocated fixed-capacity cache was implemented and verified to reuse storage, but controlled tests showed it was slower than the original dynamic `torch.cat` cache on this model/context/hardware combination, so the dynamic implementation was retained.

Manual attention and PyTorch SDPA were also compared with identical trained weights. Numerical parity passed for both full-sequence and cached inference. SDPA was modestly faster at a few short prompt lengths but slower for the 400-token cached decoding workload, so SDPA remains optional rather than becoming the default.

---

## 10. SafeTensors release standard

The final public release must not be a pickled PyTorch training checkpoint. Trusted training state and public inference state have different requirements:

```text
trusted resumable training state = .pt      |      public inference weights = .safetensors
```

A `.pt` training checkpoint contains optimizer state and Python-serialized metadata required for exact continuation. A public SafeTensors artifact contains only model tensors and cannot embed arbitrary pickle code. Architecture and provenance metadata are written explicitly to JSON instead of being hidden inside a Python object.

The intended release directories are:

```text
release/kestrel-base/     → model.safetensors | config.json | tokenizer.json
release/kestrel-instruct/ → model.safetensors | config.json | tokenizer.json
```

`config.json` should record at least the model type, release stage, variant name, vocabulary size, context length, \(d_\text{model}\), layer count, head count, head dimension, feed-forward dimension, parameter count, tied-embedding status, global training step where relevant, and processed-token count. The instruct release should additionally record post-training provenance once the SFT recipe is finalized.

Kestrel's LM head and input embedding share underlying storage, so the exporter should use the model-aware SafeTensors helpers `safetensors.torch.save_model` and `safetensors.torch.load_model` rather than blindly serializing a state dict with duplicate shared tensors. Export must first reconstruct the exact architecture from trusted checkpoint metadata, strictly load the training state dict, save inference-only tensors to CPU-backed SafeTensors, write `config.json`, copy `tokenizer.json`, instantiate a fresh model, reload the SafeTensors file strictly, and finally run a short inference smoke test.

```text
final.pt → reconstruct exact Kestrel config → strict load → save_model(...) → model.safetensors → fresh model → load_model(...) → inference smoke test → publish
```

This SafeTensors conversion is the final release step for both Kestrel-Base and Kestrel-Instruct, not a replacement for resumable `.pt` training checkpoints.

---

## 11. Repository boundaries

The core implementation should remain small:

```text
src/config.py | src/dataset.py | src/dataloader.py | src/model.py | src/train.py | src/generate.py
```

`config.py` owns shared paths, tokenizer constants, architecture definitions, and training hyperparameters. `dataset.py` owns only the generic memory-mapped token dataset. `dataloader.py` owns batching. `model.py` owns the Transformer architecture and inference cache behavior. `train.py` owns the base pretraining loop and resumable training state. `generate.py` owns public-facing generation.

One-off or pipeline utilities belong outside `src`:

```text
scripts/train_tokenizer.py | scripts/preprocess_data.py | scripts/evaluate.py | scripts/export_model.py
```

The general-domain corpus is streamed directly during tokenizer training and preprocessing, so a dedicated dataset-download script is obsolete. Benchmarks remain under `benchmarks/`; correctness tests remain under `tests/`.

---

## 12. End-to-end execution order

The operational path is now:

```text
train_tokenizer.py → verify tokenizer → preprocess_data.py → inspect train.bin/validation.bin → benchmark Kestrel-L → train.py --model large --steps 73242 → evaluate base model → select/format SFT corpus → full SFT with assistant-only loss → evaluate instruct model → export_model.py → model.safetensors/config.json/tokenizer.json
```

The 8,192-token general-domain tokenizer is frozen before corpus generation. The 1.2B-token binary training corpus is then generated exactly once with that tokenizer. Kestrel-L is pretrained from scratch on the resulting token stream, producing a trusted `final.pt`. Post-training starts from the base weights rather than from a new random initialization. Public releases are produced only after evaluation and are exported as SafeTensors.

---

## 13. Current implementation state

The architecture, S/M/L parameterization, dynamic KV caching, optional SDPA backend, training loop, generation path, width-scaling benchmark, general-domain tokenizer, general-domain preprocessing pipeline, and SafeTensors exporter are implemented. The tokenizer has been load-tested successfully with exact round-trip decoding, an 8,192-token vocabulary, and all seven reserved tokens present. The current base-model stage is generation of the new 1,199,996,928-token training binary and 10,000,000-token validation binary; after those files are integrity-checked, Kestrel-L can be pretrained for exactly 73,242 optimizer steps and evaluated. The post-training work then adds conversation formatting, assistant-only labels, full SFT, instruction evaluation, and final SafeTensors publication.

The old domain-specific binary token files, if they still exist locally under the ignored `data/` directory, are invalid for the new tokenizer and must not be reused. `train.bin` and `validation.bin` must always be regenerated after a tokenizer change.
