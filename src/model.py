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


# Creates one learned embedding vector for every possible sequence position.
# Output shape: (T, D_MODEL)
class PositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONTEXT_LENGTH, D_MODEL)

    def forward(self, sequence_length, device):
        if sequence_length > CONTEXT_LENGTH:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"maximum context length {CONTEXT_LENGTH}."
            )

        position_ids = torch.arange(sequence_length, device=device)
        return self.embedding(position_ids)


# Combines token identity and position into the initial hidden states.
# Input shape: (B, T)
# Output shape: (B, T, D_MODEL)
class InputEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = TokenEmbedding()
        self.position_embedding = PositionalEmbedding()

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(sequence_length, token_ids.device)

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
# Input shape: (B, T, D_MODEL)
# Output shape: (B, T, D_MODEL)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.query_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.key_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.value_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.output_projection = nn.Linear(D_MODEL, D_MODEL, bias=False)

    # Shape: (B, T, D_MODEL) -> (B, N_HEADS, T, D_HEAD)
    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.view(batch_size, sequence_length, N_HEADS, D_HEAD)
        return tensor.transpose(1, 2)

    # Shape: (B, N_HEADS, T, D_HEAD) -> (B, T, D_MODEL)
    def combine_heads(self, tensor):
        batch_size, _, sequence_length, _ = tensor.shape

        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, D_MODEL)

    # Computes QK^T / sqrt(D_HEAD).
    # Output shape: (B, N_HEADS, T, T)
    def compute_attention_scores(self, query, key):
        key_transposed = key.transpose(-2, -1)
        scores = torch.matmul(query, key_transposed)

        return scores / math.sqrt(D_HEAD)

    # Prevents each token from attending to future positions.
    def apply_causal_mask(self, scores):
        sequence_length = scores.shape[-1]

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=scores.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        return scores.masked_fill(causal_mask, float("-inf"))

    # Converts masked scores into attention probabilities.
    def compute_attention_weights(self, masked_scores):
        return torch.softmax(masked_scores, dim=-1)

    # Mixes the value vectors using the attention probabilities.
    def compute_attention_output(self, attention_weights, value):
        return torch.matmul(attention_weights, value)

    def forward(self, hidden_states):
        query = self.query_projection(hidden_states)
        key = self.key_projection(hidden_states)
        value = self.value_projection(hidden_states)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        scores = self.compute_attention_scores(query, key)
        masked_scores = self.apply_causal_mask(scores)
        attention_weights = self.compute_attention_weights(masked_scores)
        head_output = self.compute_attention_output(attention_weights, value)

        combined_output = self.combine_heads(head_output)
        return self.output_projection(combined_output)


# Implements the SwiGLU feed-forward network.
# Input shape: (B, T, D_MODEL)
# Output shape: (B, T, D_MODEL)
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gate_projection = nn.Linear(D_MODEL, D_FF, bias=False)
        self.up_projection = nn.Linear(D_MODEL, D_FF, bias=False)
        self.down_projection = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, hidden_states):
        gate = self.gate_projection(hidden_states)
        up = self.up_projection(hidden_states)

        gated = F.silu(gate) * up
        return self.down_projection(gated)


# Implements one complete pre-norm decoder transformer block.
# Each sublayer writes an update into the residual stream.
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_norm = RMSNorm()
        self.attention = MultiHeadSelfAttention()

        self.feed_forward_norm = RMSNorm()
        self.feed_forward = SwiGLU()

    def forward(self, hidden_states):
        attention_input = self.attention_norm(hidden_states)
        attention_update = self.attention(attention_input)
        hidden_states = hidden_states + attention_update

        feed_forward_input = self.feed_forward_norm(hidden_states)
        feed_forward_update = self.feed_forward(feed_forward_input)
        hidden_states = hidden_states + feed_forward_update

        return hidden_states


# Stacks N_LAYERS independent transformer blocks.
# Input shape: (B, T, D_MODEL)
# Output shape: (B, T, D_MODEL)
class TransformerStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states


# Implements the complete decoder-only KestrelLM language model.
# Input shape: (B, T)
# Output shape: (B, T, VOCAB_SIZE)
class KestrelLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_embedding = InputEmbedding()
        self.transformer = TransformerStack()
        self.final_norm = RMSNorm()
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)

        # Initializes embeddings and linear layers with transformer-scale weights.
        self.apply(self.initialize_weights)

        # The input token embeddings and output vocabulary projection share weights.
        self.lm_head.weight = self.input_embedding.token_embedding.embedding.weight

    # Initializes learned matrices from a small zero-centered normal distribution.
    # Large default embedding weights would otherwise produce excessively large logits.
    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        hidden_states = self.input_embedding(token_ids)
        hidden_states = self.transformer(hidden_states)
        hidden_states = self.final_norm(hidden_states)

        return self.lm_head(hidden_states)


# Computes next-token cross-entropy across every token in the batch.
def compute_loss(logits, targets):
    flat_logits = logits.reshape(-1, VOCAB_SIZE)
    flat_targets = targets.reshape(-1)

    return F.cross_entropy(flat_logits, flat_targets)


# Returns the number of unique trainable scalar parameters in the model.
def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)