import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import KESTREL_MEDIUM, VOCAB_SIZE


# Converts integer token IDs into learned embedding vectors.
class TokenEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, config.d_model)

    def forward(self, token_ids):
        return self.embedding(token_ids)


# Creates learned embedding vectors for sequence positions.
# start_position allows cached decoding to continue after an existing prefix.
class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_length = config.context_length
        self.embedding = nn.Embedding(config.context_length, config.d_model)

    def forward(self, sequence_length, device, start_position=0):
        end_position = start_position + sequence_length

        if end_position > self.context_length:
            raise ValueError(
                f"Position {end_position} exceeds maximum context length "
                f"{self.context_length}."
            )

        position_ids = torch.arange(start_position, end_position, device=device)
        return self.embedding(position_ids)


# Combines token and positional embeddings into the initial hidden states.
class InputEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)

    def forward(self, token_ids, start_position=0):
        sequence_length = token_ids.shape[1]

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(
            sequence_length,
            token_ids.device,
            start_position,
        )

        return token_vectors + position_vectors


# Normalizes each hidden vector using its root-mean-square magnitude.
class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self, hidden_states):
        mean_square = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        normalized = hidden_states * inverse_rms

        return normalized * self.scale


# Implements multi-head causal self-attention.
# Manual attention remains the default, with SDPA available for comparison.
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config, attention_backend="manual"):
        super().__init__()

        if attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend must be either 'manual' or 'sdpa'.")

        self.config = config
        self.attention_backend = attention_backend

        self.query_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output_projection = nn.Linear(config.d_model, config.d_model, bias=False)

    # Converts (B, T, d_model) into (B, n_heads, T, d_head).
    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.config.n_heads,
            self.config.d_head,
        )

        return tensor.transpose(1, 2)

    # Converts (B, n_heads, T, d_head) back into (B, T, d_model).
    def combine_heads(self, tensor):
        batch_size, _, sequence_length, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()

        return tensor.view(batch_size, sequence_length, self.config.d_model)

    # Computes scaled query-key similarity scores.
    def compute_attention_scores(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1))
        return scores / math.sqrt(self.config.d_head)

    # Prevents queries from attending to future key positions.
    # past_length handles the positional offset during cached decoding.
    def apply_causal_mask(self, scores):
        query_length = scores.shape[-2]
        key_length = scores.shape[-1]
        past_length = key_length - query_length

        query_positions = torch.arange(query_length, device=scores.device) + past_length
        key_positions = torch.arange(key_length, device=scores.device)

        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        return scores.masked_fill(causal_mask, float("-inf"))

    # Builds the equivalent boolean mask expected by PyTorch SDPA.
    def create_sdpa_causal_mask(self, query_length, key_length, device):
        past_length = key_length - query_length

        query_positions = torch.arange(query_length, device=device) + past_length
        key_positions = torch.arange(key_length, device=device)

        return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

    # Executes the handwritten attention implementation.
    def manual_attention(self, query, key, value):
        scores = self.compute_attention_scores(query, key)
        masked_scores = self.apply_causal_mask(scores)
        attention_weights = torch.softmax(masked_scores, dim=-1)

        return torch.matmul(attention_weights, value)

    # Executes the equivalent operation using PyTorch SDPA.
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

    # Appends newly computed keys and values to the layer's existing KV cache.
    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        query = self.split_heads(self.query_projection(hidden_states))
        new_key = self.split_heads(self.key_projection(hidden_states))
        new_value = self.split_heads(self.value_projection(hidden_states))

        if kv_cache is None:
            key = new_key
            value = new_value
        else:
            cached_key, cached_value = kv_cache
            key = torch.cat((cached_key, new_key), dim=-2)
            value = torch.cat((cached_value, new_value), dim=-2)

        if self.attention_backend == "manual":
            head_output = self.manual_attention(query, key, value)
        else:
            head_output = self.sdpa_attention(query, key, value)

        output = self.output_projection(self.combine_heads(head_output))

        if use_cache:
            return output, (key, value)

        return output


# Implements the SwiGLU feed-forward network.
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_projection = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_projection(hidden_states)
        up = self.up_projection(hidden_states)
        gated = F.silu(gate) * up

        return self.down_projection(gated)


# Implements one pre-norm decoder transformer block.
class TransformerBlock(nn.Module):
    def __init__(self, config, attention_backend="manual"):
        super().__init__()

        self.attention_norm = RMSNorm(config.d_model)
        self.attention = MultiHeadSelfAttention(
            config,
            attention_backend=attention_backend,
        )

        self.feed_forward_norm = RMSNorm(config.d_model)
        self.feed_forward = SwiGLU(config)

    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        attention_input = self.attention_norm(hidden_states)

        if use_cache:
            attention_update, new_kv_cache = self.attention(
                attention_input,
                kv_cache=kv_cache,
                use_cache=True,
            )
        else:
            attention_update = self.attention(attention_input)

        hidden_states = hidden_states + attention_update

        feed_forward_input = self.feed_forward_norm(hidden_states)
        feed_forward_update = self.feed_forward(feed_forward_input)
        hidden_states = hidden_states + feed_forward_update

        if use_cache:
            return hidden_states, new_kv_cache

        return hidden_states


# Stacks the configured number of transformer blocks.
class TransformerStack(nn.Module):
    def __init__(self, config, attention_backend="manual"):
        super().__init__()

        self.config = config

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config, attention_backend=attention_backend)
                for _ in range(config.n_layers)
            ]
        )

    def forward(self, hidden_states, kv_cache=None, use_cache=False):
        if kv_cache is not None and len(kv_cache) != self.config.n_layers:
            raise ValueError(
                f"Expected {self.config.n_layers} KV-cache entries, "
                f"received {len(kv_cache)}."
            )

        new_kv_cache = []

        for layer_index, block in enumerate(self.blocks):
            layer_cache = kv_cache[layer_index] if kv_cache is not None else None

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


# Implements the complete decoder-only KestrelLM.
# Kestrel-M remains the default so all existing code and checkpoints stay compatible.
class KestrelLM(nn.Module):
    def __init__(self, attention_backend="manual", config=KESTREL_MEDIUM):
        super().__init__()

        config.validate()

        self.config = config
        self.attention_backend = attention_backend

        self.input_embedding = InputEmbedding(config)
        self.transformer = TransformerStack(
            config,
            attention_backend=attention_backend,
        )

        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)

        self.apply(self.initialize_weights)

        # Ties the input token embedding matrix to the output vocabulary projection.
        self.lm_head.weight = self.input_embedding.token_embedding.embedding.weight

    # Initializes learned matrices from a small zero-centered normal distribution.
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # Returns the number of tokens already represented by a KV cache.
    def get_cache_length(self, kv_cache):
        if kv_cache is None:
            return 0

        if len(kv_cache) != self.config.n_layers:
            raise ValueError(
                f"Expected {self.config.n_layers} KV-cache entries, "
                f"received {len(kv_cache)}."
            )

        cache_lengths = [layer_cache[0].shape[-2] for layer_cache in kv_cache]

        if len(set(cache_lengths)) != 1:
            raise ValueError("All transformer layers must have the same KV-cache length.")

        return cache_lengths[0]

    # Runs normal full-sequence inference or cached autoregressive inference.
    def forward(self, token_ids, kv_cache=None, use_cache=False):
        if kv_cache is not None and not use_cache:
            raise ValueError("kv_cache requires use_cache=True.")

        past_length = self.get_cache_length(kv_cache)
        sequence_length = token_ids.shape[1]

        if past_length + sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence would reach position {past_length + sequence_length}, "
                f"exceeding maximum context length {self.config.context_length}."
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
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)

    return F.cross_entropy(flat_logits, flat_targets)


# Returns the number of unique trainable scalar parameters in the model.
def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
