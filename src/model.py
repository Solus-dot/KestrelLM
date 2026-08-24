import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    CONTEXT_LENGTH,
    D_FF,
    D_HEAD,
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    VOCAB_SIZE,
)


# Converts integer token IDs into learned D_MODEL-dimensional vectors.
# Input shape: (B, T)
# Output shape: (B, T, D_MODEL)
class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

    def forward(self, token_ids):
        return self.embedding(token_ids)


# Creates learned embedding vectors for a range of sequence positions.
# start_position allows cached decoding to continue from the correct position.
class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)

    def forward(self, sequence_length, device, start_position=0):
        end_position = start_position + sequence_length

        if end_position > CONTEXT_LENGTH:
            raise ValueError(
                f"Position {end_position} exceeds "
                f"maximum context length {CONTEXT_LENGTH}."
            )

        position_ids = torch.arange(
            start_position,
            end_position,
            device=device,
        )

        return self.embedding(position_ids)


# Combines token identity and position into the initial hidden states.
# start_position is zero during normal training and advances during cached decoding.
class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = TokenEmbedding()
        self.position_embedding = PositionalEmbedding()

    def forward(self, token_ids, start_position=0):
        sequence_length = token_ids.shape[1]

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(
            sequence_length,
            token_ids.device,
            start_position,
        )

        return token_vectors + position_vectors


# Normalizes each token vector using its root-mean-square magnitude.
# Input shape: (B, T, D_MODEL)
# Output shape: (B, T, D_MODEL)
class RMSNorm(nn.Module):
    def __init__(self, dimension=D_MODEL, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self, hidden_states):
        mean_square = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        normalized = hidden_states * inverse_rms

        return normalized * self.scale


# Implements multi-head causal self-attention.
# The manual backend exposes the attention math directly, while the SDPA backend
# delegates the same operation to PyTorch's optimized scaled-attention primitive.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, attention_backend="manual"):
        super().__init__()

        if attention_backend not in {"manual", "sdpa"}:
            raise ValueError(
                "attention_backend must be either 'manual' or 'sdpa'."
            )

        self.attention_backend = attention_backend

        self.query_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.key_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.value_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.output_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    # Shape: (B, T, D_MODEL) -> (B, N_HEADS, T, D_HEAD)
    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.view(
            batch_size,
            sequence_length,
            N_HEADS,
            D_HEAD,
        )

        return tensor.transpose(1, 2)

    # Shape: (B, N_HEADS, T, D_HEAD) -> (B, T, D_MODEL)
    def combine_heads(self, tensor):
        batch_size, _, sequence_length, _ = tensor.shape

        tensor = tensor.transpose(1, 2).contiguous()

        return tensor.view(
            batch_size,
            sequence_length,
            D_MODEL,
        )

    # Computes QK^T / sqrt(D_HEAD).
    # Query and key lengths may differ during cached decoding.
    def compute_attention_scores(self, query, key):
        key_transposed = key.transpose(-2, -1)
        scores = torch.matmul(query, key_transposed)

        return scores / math.sqrt(D_HEAD)

    # Prevents each query from attending to keys belonging to future positions.
    # This also handles queries that begin after an existing cached prefix.
    def apply_causal_mask(self, scores):
        query_length = scores.shape[-2]
        key_length = scores.shape[-1]
        past_length = key_length - query_length

        query_positions = (
            torch.arange(
                query_length,
                device=scores.device,
            )
            + past_length
        )

        key_positions = torch.arange(
            key_length,
            device=scores.device,
        )

        causal_mask = (
            key_positions.unsqueeze(0)
            > query_positions.unsqueeze(1)
        )

        return scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

    # Builds the equivalent boolean mask for PyTorch SDPA.
    # True means that a query is allowed to attend to that key position.
    def create_sdpa_causal_mask(self, query_length, key_length, device):
        past_length = key_length - query_length

        query_positions = (
            torch.arange(
                query_length,
                device=device,
            )
            + past_length
        )

        key_positions = torch.arange(
            key_length,
            device=device,
        )

        return (
            key_positions.unsqueeze(0)
            <= query_positions.unsqueeze(1)
        )

    # Converts masked scores into attention probabilities.
    def compute_attention_weights(self, masked_scores):
        return torch.softmax(masked_scores, dim=-1)

    # Mixes the value vectors using the attention probabilities.
    def compute_attention_output(self, attention_weights, value):
        return torch.matmul(attention_weights, value)

    # Executes the handwritten scaled dot-product attention implementation.
    def manual_attention(self, query, key, value):
        scores = self.compute_attention_scores(
            query,
            key,
        )

        masked_scores = self.apply_causal_mask(scores)

        attention_weights = self.compute_attention_weights(
            masked_scores
        )

        return self.compute_attention_output(
            attention_weights,
            value,
        )

    # Executes the same causal attention operation using PyTorch SDPA.
    # An explicit mask is used so cached queries attend to the full past prefix.
    def sdpa_attention(self, query, key, value):
        query_length = query.shape[-2]
        key_length = key.shape[-2]

        causal_mask = self.create_sdpa_causal_mask(
            query_length,
            key_length,
            query.device,
        )

        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=False,
        )

    # kv_cache contains the keys and values computed by this layer previously.
    # When use_cache is True, the updated cache is returned with the output.
    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        query = self.query_projection(hidden_states)
        new_key = self.key_projection(hidden_states)
        new_value = self.value_projection(hidden_states)

        query = self.split_heads(query)
        new_key = self.split_heads(new_key)
        new_value = self.split_heads(new_value)

        if kv_cache is None:
            key = new_key
            value = new_value

        else:
            cached_key, cached_value = kv_cache

            key = torch.cat(
                (cached_key, new_key),
                dim=-2,
            )

            value = torch.cat(
                (cached_value, new_value),
                dim=-2,
            )

        if self.attention_backend == "manual":
            head_output = self.manual_attention(
                query,
                key,
                value,
            )

        else:
            head_output = self.sdpa_attention(
                query,
                key,
                value,
            )

        combined_output = self.combine_heads(head_output)
        output = self.output_projection(combined_output)

        if use_cache:
            return output, (key, value)

        return output


# Implements the SwiGLU feed-forward network.
# Input shape: (B, T, D_MODEL)
# Output shape: (B, T, D_MODEL)
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate_projection = nn.Linear(
            D_MODEL,
            D_FF,
            bias=False,
        )

        self.up_projection = nn.Linear(
            D_MODEL,
            D_FF,
            bias=False,
        )

        self.down_projection = nn.Linear(
            D_FF,
            D_MODEL,
            bias=False,
        )

    def forward(self, hidden_states):
        gate = self.gate_projection(hidden_states)
        up = self.up_projection(hidden_states)

        gated = F.silu(gate) * up

        return self.down_projection(gated)


# Implements one complete pre-norm decoder transformer block.
# Each block owns one layer of the KV cache during inference.
class TransformerBlock(nn.Module):
    def __init__(self, attention_backend="manual"):
        super().__init__()

        self.attention_norm = RMSNorm()
        self.attention = MultiHeadSelfAttention(
            attention_backend=attention_backend
        )

        self.feed_forward_norm = RMSNorm()
        self.feed_forward = SwiGLU()

    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        attention_input = self.attention_norm(hidden_states)

        if use_cache:
            attention_update, new_kv_cache = self.attention(
                attention_input,
                kv_cache=kv_cache,
                use_cache=True,
            )

        else:
            attention_update = self.attention(
                attention_input
            )

        hidden_states = hidden_states + attention_update

        feed_forward_input = self.feed_forward_norm(
            hidden_states
        )

        feed_forward_update = self.feed_forward(
            feed_forward_input
        )

        hidden_states = hidden_states + feed_forward_update

        if use_cache:
            return hidden_states, new_kv_cache

        return hidden_states


# Stacks N_LAYERS independent transformer blocks.
# Every layer uses the same selected attention backend.
class TransformerStack(nn.Module):
    def __init__(self, attention_backend="manual"):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    attention_backend=attention_backend
                )
                for _ in range(N_LAYERS)
            ]
        )

    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        if kv_cache is not None and len(kv_cache) != N_LAYERS:
            raise ValueError(
                f"Expected {N_LAYERS} KV-cache entries, "
                f"received {len(kv_cache)}."
            )

        new_kv_cache = []

        for layer_index, block in enumerate(self.blocks):
            layer_cache = (
                kv_cache[layer_index]
                if kv_cache is not None
                else None
            )

            if use_cache:
                hidden_states, layer_new_cache = block(
                    hidden_states,
                    kv_cache=layer_cache,
                    use_cache=True,
                )

                new_kv_cache.append(layer_new_cache)

            else:
                hidden_states = block(hidden_states)

        if use_cache:
            return hidden_states, new_kv_cache

        return hidden_states


# Implements the complete decoder-only KestrelLM language model.
# Manual attention remains the default; SDPA can be selected for benchmarking.
class KestrelLM(nn.Module):
    def __init__(self, attention_backend="manual"):
        super().__init__()

        self.attention_backend = attention_backend

        self.input_embedding = InputEmbedding()
        self.transformer = TransformerStack(
            attention_backend=attention_backend
        )
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=False,
        )

        # Initializes embeddings and linear layers with transformer-scale weights.
        self.apply(self.initialize_weights)

        # The input token embeddings and output vocabulary projection share weights.
        self.lm_head.weight = (
            self.input_embedding
            .token_embedding
            .embedding
            .weight
        )

    # Initializes learned matrices from a small zero-centered normal distribution.
    # Large default embedding weights would otherwise produce excessively large logits.
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    # Returns the number of tokens already stored in the KV cache.
    def get_cache_length(self, kv_cache):
        if kv_cache is None:
            return 0

        if len(kv_cache) != N_LAYERS:
            raise ValueError(
                f"Expected {N_LAYERS} KV-cache entries, "
                f"received {len(kv_cache)}."
            )

        cache_lengths = [
            layer_cache[0].shape[-2]
            for layer_cache in kv_cache
        ]

        if len(set(cache_lengths)) != 1:
            raise ValueError(
                "All transformer layers must have "
                "the same KV-cache length."
            )

        return cache_lengths[0]

    # During normal training this behaves exactly as before.
    # During cached inference it also accepts and returns per-layer KV state.
    def forward(self, token_ids, kv_cache=None, use_cache=False):
        if kv_cache is not None and not use_cache:
            raise ValueError(
                "kv_cache requires use_cache=True."
            )

        past_length = self.get_cache_length(kv_cache)
        sequence_length = token_ids.shape[1]

        if past_length + sequence_length > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence would reach position "
                f"{past_length + sequence_length}, exceeding "
                f"maximum context length {CONTEXT_LENGTH}."
            )

        hidden_states = self.input_embedding(
            token_ids,
            start_position=past_length,
        )

        if use_cache:
            hidden_states, new_kv_cache = self.transformer(
                hidden_states,
                kv_cache=kv_cache,
                use_cache=True,
            )

        else:
            hidden_states = self.transformer(hidden_states)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if use_cache:
            return logits, new_kv_cache

        return logits


# Computes next-token cross-entropy across every token in the batch.
def compute_loss(logits, targets):
    flat_logits = logits.reshape(
        -1,
        VOCAB_SIZE,
    )

    flat_targets = targets.reshape(-1)

    return F.cross_entropy(
        flat_logits,
        flat_targets,
    )


# Returns the number of unique trainable scalar parameters in the model.
def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
