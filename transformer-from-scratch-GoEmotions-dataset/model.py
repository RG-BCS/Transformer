"""
models.py

Defines the core model architecture for emotion classification using a Transformer-style encoder.
Includes embedding layer with positional encodings, multi-head attention, feedforward networks,
stacked encoder layers, and the final EmotionsModel for classification.

Author: [Your Name]
"""

import math
import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Embedding layer with positional encoding and optional masking for padded tokens.
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=5000, PAD_token=0):
        super(EmbeddingLayer, self).__init__()
        assert embed_size % 2 == 0, "embed_size should be even for positional encoding"
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        # Precompute positional encodings
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, embed_size, 2).float()
        denom = torch.pow(10000, i / embed_size)
        angle_rates = pos / denom
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(angle_rates)
        pe[:, 1::2] = torch.cos(angle_rates)
        self.register_buffer('positional_encoding', pe.unsqueeze(0))

    def forward(self, x, input_lengths=None, pre_norm=False):
        """
        Forward pass for embedding + positional encoding.

        Args:
            x: Input token indices [batch, seq_len]
            input_lengths: Lengths for masking [batch]
            pre_norm: Whether to apply layer norm before returning output

        Returns:
            Embedded and positionally encoded tensor [batch, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        output_ = self.dropout(self.embedding(x) * (self.embed_size)**0.5 + pos_enc)

        if input_lengths is not None:
            embed_mask = (torch.arange(x.size(1), device=x.device)[None, :] < input_lengths[:, None]).unsqueeze(-1)
            output_ = output_ * embed_mask

        return self.layer_norm(output_) if pre_norm else output_


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional causal and padding mask support.
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, dropout=0.0, context_length=5000, qkv_bias=False):
        super().__init__()
        assert d_out_n_heads % num_heads == 0, "d_out must be divisible by num_heads"

        self.w_key = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_query = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_value = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out_n_heads, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads

        # Causal mask (upper triangular)
        self.register_buffer('mask', torch.triu(torch.ones(size=(context_length, context_length)), diagonal=1))

    def forward(self, x, input_lengths=None, causal=False):
        """
        Forward pass for multi-head self-attention.

        Args:
            x: Input tensor [batch, seq_len, embed_size]
            input_lengths: Optional tensor of sequence lengths [batch]
            causal: Whether to apply causal masking (for autoregressive setups)

        Returns:
            Tuple: (output tensor [batch, seq_len, embed_size], attention weights [batch, heads, seq_len, seq_len])
        """
        batch, seq_len, _ = x.shape
        query = self.w_query(x).view(batch, self.num_heads, seq_len, self.head_dim)
        key = self.w_key(x).view(batch, self.num_heads, seq_len, self.head_dim)
        value = self.w_value(x).view(batch, self.num_heads, seq_len, self.head_dim)

        d_k = math.sqrt(key.shape[-1])
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k

        # Padding mask
        if input_lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device)[None, :] >= input_lengths[:, None]
            pad_mask = pad_mask[:, None, None, :]  # shape [B, 1, 1, L]
        else:
            pad_mask = None

        # Causal mask
        if causal:
            causal_mask = self.mask[:seq_len, :seq_len].bool().to(x.device)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        else:
            causal_mask = None

        if pad_mask is not None:
            alignment_score = alignment_score.masked_fill(pad_mask, float('-inf'))
        if causal_mask is not None:
            alignment_score = alignment_score.masked_fill(causal_mask, float('-inf'))

        attention_scores = torch.softmax(alignment_score, dim=-1)
        attention_scores = self.dropout(attention_scores)

        context_vec = torch.matmul(attention_scores, value).contiguous().view(batch, seq_len, -1)
        out = self.out_proj(context_vec)

        # Apply mask to output
        if input_lengths is not None:
            query_mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            query_mask = query_mask.unsqueeze(-1)
            out = out * query_mask

        output_skip = self.dropout(out) + x
        return output_skip, attention_scores


class FeedForward(nn.Module):
    """
    Position-wise FeedForward network with layer normalization.
    """
    def __init__(self, embed_size, ffn_hidden_dim, dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_size)
        )
        self.pre_ffn_norm = nn.LayerNorm(embed_size)
        self.post_ffn_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_lengths=None):
        """
        Forward pass for feedforward block.

        Args:
            x: Input tensor [batch, seq_len, embed_size]
            input_lengths: Optional lengths for masking

        Returns:
            Output tensor after FFN and skip connection
        """
        batch, seq_len, _ = x.shape
        norm_x = self.pre_ffn_norm(x)
        output_ = self.ffn(norm_x)

        if input_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            output_ = output_ * mask.unsqueeze(-1)

        return self.post_ffn_norm(self.dropout(output_) + x)


class EncoderLayer(nn.Module):
    """
    A single Transformer encoder layer (MHA + FFN).
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=5000, qkv_bias=False):
        super().__init__()
        self.mha_attn = MultiHeadAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.ffn = FeedForward(embed_size, ffn_hidden_dim, dropout)

    def forward(self, x, input_lengths, causal=False, return_attn_weights=False):
        """
        Forward pass for encoder layer.

        Returns:
            Output tensor, optionally with attention weights
        """
        mha_output, attn_weights = self.mha_attn(x, input_lengths, causal=causal)
        ffn_output = self.ffn(mha_output, input_lengths)
        if return_attn_weights:
            return ffn_output, attn_weights
        return ffn_output


class EncoderBlocks(nn.Module):
    """
    Stack of Transformer encoder layers with input embeddings.
    """
    def __init__(self, num_layers, src_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=1000, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embedding = EmbeddingLayer(src_vocab_size, embed_size, PAD_token=PAD_token)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                         dropout, context_length, qkv_bias)
            for _ in range(num_layers)
        ])

    def forward(self, x, input_lengths, causal=False, return_attn_weights=False):
        """
        Forward pass through the full encoder stack.

        Returns:
            Final encoder output, optionally with attention weights.
        """
        embed = self.embedding(x, input_lengths)
        if return_attn_weights:
            attn_weights = []
            for layer in self.layers:
                embed, attn = layer(embed, input_lengths, causal=causal, return_attn_weights=True)
                attn_weights.append(attn)
            return embed, attn_weights

        for layer in self.layers:
            embed = layer(embed, input_lengths, causal=causal)
        return embed


class EmotionsModel(nn.Module):
    """
    Final model for emotion classification. Uses transformer encoder and outputs emotion logits.
    """
    def __init__(self, num_layers, src_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 output_dim=28, dropout=0.1, context_length=5000, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embed_size = embed_size
        self.encoder = EncoderBlocks(num_layers, src_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                                     dropout, context_length, qkv_bias, PAD_token)
        self.output = nn.Linear(embed_size, output_dim)

    def forward(self, x, input_lengths, causal=False, return_attn_weights=False):
        """
        Forward pass for classification.

        Returns:
            Tensor of shape [batch, output_dim] representing emotion logits.
        """
        encoder_output = self.encoder(x, input_lengths, causal=causal, return_attn_weights=return_attn_weights)
        output = self.output(encoder_output)  # [batch, seq_len, output_dim]

        # Mask padded outputs
        mask = torch.arange(x.size(1), device=x.device)[None, :] < input_lengths[:, None]
        mask = mask.unsqueeze(-1)
        output = output * mask

        summed = output.sum(dim=1)
        lengths = input_lengths.unsqueeze(1)
        mean_output = summed / lengths
        return mean_output  # [batch, output_dim]
