import math
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    """
    Embedding layer with learned token embeddings and fixed sinusoidal positional encodings.

    Args:
        vocab_size (int): Number of tokens in vocabulary.
        embed_size (int): Dimensionality of embedding vector (must be even for positional encoding).
        dropout (float): Dropout probability.
        max_len (int): Maximum sequence length for positional encoding.
        PAD_token (int): Index of padding token.

    Input:
        x (Tensor): Input token indices of shape (batch_size, seq_len).
        input_lengths (Tensor, optional): Lengths of each sequence in batch.
        pre_norm (bool): Whether to apply layer normalization before returning.

    Output:
        Tensor: Embedded inputs with positional encodings, shape (batch_size, seq_len, embed_size).
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=50, PAD_token=0):
        super(EmbeddingLayer, self).__init__()
        assert embed_size % 2 == 0, "embed_size should be even number for positional encoding to work"
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        # Precompute sinusoidal positional encodings
        pos = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        i = torch.arange(0, embed_size, 2).float()   # (embed_size/2,)
        denom = torch.pow(10000, i / embed_size)     # (embed_size/2,)
        angle_rates = pos / denom                     # (max_len, embed_size/2)

        pe = torch.zeros(max_len, embed_size)        # (max_len, embed_size)
        pe[:, 0::2] = torch.sin(angle_rates)         # even indices
        pe[:, 1::2] = torch.cos(angle_rates)         # odd indices

        # Register as buffer so it is saved and moved to GPU with the model but not updated by optimizer
        self.register_buffer('positional_encoding', pe.unsqueeze(0))  # (1, max_len, embed_size)

    def forward(self, x, input_lengths=None, pre_norm=False):
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)  # slice to input length

        # Scale embedding by sqrt(embed_size) as in Transformer and add positional encoding
        output_ = self.dropout(self.embedding(x) * (self.embed_size ** 0.5) + pos_enc)

        if input_lengths is not None:
            # Mask padding tokens to zero out their positional encodings
            embed_mask = (torch.arange(x.size(1), device=x.device)[None, :] < input_lengths[:, None]).unsqueeze(-1)
            output_ = output_ * embed_mask  # mask shape = (batch, seq_len, 1)

        if pre_norm:
            return self.layer_norm(output_)
        else:
            return output_

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention module supporting causal and padding masks.

    Args:
        embed_size (int): Input embedding dimension.
        d_out_n_heads (int): Output dimension across all heads.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        context_length (int): Maximum sequence length.
        qkv_bias (bool): Whether to include bias in query/key/value linear layers.

    Input:
        x (Tensor): Input tensor of shape (batch, seq_len, embed_size).
        input_lengths (Tensor, optional): Lengths of sequences to mask padding.
        causal (bool): Whether to apply causal mask (prevent attending to future tokens).

    Output:
        output_skip (Tensor): Output after attention, dropout, and residual connection.
        attention_scores (Tensor): Attention weights, shape (batch, num_heads, seq_len, seq_len).
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, dropout=0.0, context_length=50, qkv_bias=False):
        super().__init__()
        assert d_out_n_heads % num_heads == 0, "d_out must be divisible by num_heads"

        self.w_key = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_query = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_value = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out_n_heads, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads

        # Precompute causal mask to prevent attending to future positions
        self.register_buffer('mask', torch.triu(torch.ones(size=(context_length, context_length)), diagonal=1))

    def forward(self, x, input_lengths=None, causal=False):
        batch, seq_len, dim_in = x.shape

        # Linear projections for query, key, value
        query = self.w_query(x).view(batch, self.num_heads, seq_len, self.head_dim)
        key = self.w_key(x).view(batch, self.num_heads, seq_len, self.head_dim)
        value = self.w_value(x).view(batch, self.num_heads, seq_len, self.head_dim)

        d_k = math.sqrt(self.head_dim)

        # Compute scaled dot-product attention scores
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k  # (B, H, L, L)

        # Padding mask for key tokens (mask out padded keys)
        if input_lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device)[None, :] >= input_lengths[:, None]
            pad_mask = pad_mask[:, None, None, :]  # (B, 1, 1, L)
        else:
            pad_mask = None

        # Causal mask to prevent attending to future tokens
        causal_mask = self.mask[:seq_len, :seq_len].bool().to(x.device) if causal else None
        if causal_mask is not None:
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        if pad_mask is not None:
            alignment_score = alignment_score.masked_fill(pad_mask, float('-inf'))
        if causal_mask is not None:
            alignment_score = alignment_score.masked_fill(causal_mask, float('-inf'))

        # Fix rows with all -inf to avoid NaNs in softmax
        if torch.isinf(alignment_score).all(dim=-1).any():
            alignment_score = alignment_score.masked_fill(torch.isinf(alignment_score), -1e9)

        attention_scores = torch.softmax(alignment_score, dim=-1)  # (B, H, L, L)
        attention_scores = self.dropout(attention_scores)

        # Compute context vectors
        context_vec = torch.matmul(attention_scores, value).contiguous()
        context_vec = context_vec.view(batch, seq_len, self.num_heads * self.head_dim)

        # Mask query positions that are padding
        if input_lengths is not None:
            out = self.out_proj(context_vec)
            query_mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            query_mask = query_mask.unsqueeze(-1)
            output = out * query_mask
        else:
            output = self.out_proj(context_vec)

        # Residual connection + dropout
        output_skip = self.dropout(output) + x

        assert output_skip.shape == x.shape, "Residual addition shape mismatch"

        return output_skip, attention_scores

class DecoderCrossAttention(nn.Module):
    """
    Cross-Attention module used in decoder layers attending to encoder outputs.

    Args:
        embed_size (int): Embedding dimension.
        d_out_n_heads (int): Output dimension across all heads.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        context_length (int): Maximum sequence length.
        qkv_bias (bool): Whether to include bias in linear layers.

    Input:
        query (Tensor): Decoder input tensor (batch, seq_len_q, embed_size).
        key (Tensor): Encoder output tensor (batch, seq_len_k, embed_size).
        value (Tensor): Encoder output tensor (batch, seq_len_k, embed_size).
        query_lengths (Tensor, optional): Lengths of query sequences.
        key_lengths (Tensor, optional): Lengths of key sequences.
        causal (bool): Must be False, causal mask not used here.

    Output:
        output_skip (Tensor): Output tensor after cross-attention and residual connection.
        attn_weights (Tensor): Attention weights of shape (batch, num_heads, seq_len_q, seq_len_k).
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, dropout=0.1, context_length=50, qkv_bias=False):
        super().__init__()
        assert d_out_n_heads % num_heads == 0, "d_out must be divisible by num_heads"

        self.w_key = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_query = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_value = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.norm_query = nn.LayerNorm(embed_size)

        self.out_proj = nn.Linear(d_out_n_heads, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads

    def forward(self, query, key, value, query_lengths=None, key_lengths=None, causal=False):
        assert not causal, "Causal masking should not be used in cross-attention."
        assert key.shape[-1] == value.shape[-1] == query.shape[-1], "Dimension mismatch between query/key/value"

        batch, seq_len_q, _ = query.shape
        batch, seq_len_k, _ = key.shape

        # Normalize query before attention computation
        query_norm = self.norm_query(query)
        query_input = query_norm  # for residual connection

        # Linear projections
        query_proj = self.w_query(query_norm).view(batch, self.num_heads, seq_len_q, self.head_dim)
        key_proj = self.w_key(key).view(batch, self.num_heads, seq_len_k, self.head_dim)
        value_proj = self.w_value(value).view(batch, self.num_heads, seq_len_k, self.head_dim)

        d_k = math.sqrt(self.head_dim)

        alignment_score = torch.matmul(query_proj, key_proj.transpose(-2, -1)) / d_k  # (B, H, Q, K)

        # Mask padded keys (encoder side)
        if key_lengths is not None:
            kv_mask = torch.arange(seq_len_k, device=query.device)[None, :] >= key_lengths[:, None]
            kv_mask = kv_mask[:, None, None, :].expand(batch, self.num_heads, seq_len_q, seq_len_k)
            alignment_score = alignment_score.masked_fill(kv_mask, float('-inf'))

        attn_weights = torch.softmax(alignment_score, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_proj)  # (B, H, Q, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len_q, -1)

        output = self.out_proj(attn_output)

        # Mask padded query positions (decoder side)
        if query_lengths is not None:
            q_mask = torch.arange(seq_len_q, device=query.device)[None, :] < query_lengths[:, None]
            output = output * q_mask.unsqueeze(-1)

        output_skip = self.dropout(output) + query_input
        return output_skip, attn_weights

class FeedForward(nn.Module):
    """
    Position-wise feedforward network with residual connections and layer normalization.

    Args:
        embed_size (int): Input and output embedding dimension.
        ffn_hidden_dim (int): Hidden layer dimension.
        dropout (float): Dropout rate.

    Input:
        x (Tensor): Input tensor (batch, seq_len, embed_size).
        input_lengths (Tensor, optional): Masking to zero out padded positions.

    Output:
        Tensor: Output tensor after feedforward, dropout, normalization and residual addition.
    """
    def __init__(self, embed_size, ffn_hidden_dim, dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_size)
        )
        self.pre_ffn_norm = nn.LayerNorm(embed_size)
        self.post_ffn_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, input_lengths=None):
        batch, seq_len, _ = x.shape
        norm_x = self.pre_ffn_norm(x)  # normalize before FFN
        output_ = self.ffn(norm_x)

        if input_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            output_ = output_ * mask.unsqueeze(-1)

        # Residual + post-FFN normalization
        return self.post_ffn_norm(self.dropout(output_) + x)

class DecoderLayer(nn.Module):
    """
    Single decoder layer composed of masked multi-head self-attention, cross-attention, and feedforward sublayers.

    Args:
        embed_size (int): Embedding dimension.
        d_out_n_heads (int): Output dimension across heads.
        num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Hidden dimension for feedforward network.
        dropout (float): Dropout rate.
        context_length (int): Max sequence length.
        qkv_bias (bool): Whether query/key/value linear layers have bias.

    Input:
        target (Tensor): Target input embeddings (batch, seq_len, embed_size).
        target_lengths (Tensor): Lengths of target sequences.
        key (Tensor, optional): Encoder outputs for cross-attention.
        value (Tensor, optional): Encoder outputs for cross-attention.
        key_lengths (Tensor, optional): Lengths of encoder sequences.
        causal (bool): Whether to apply causal mask in self-attention.
        return_attn_weights (bool): If True, returns attention weights.

    Output:
        If return_attn_weights:
            Tuple of (output, masked_self_attention_weights, cross_attention_weights)
        Else:
            output tensor (batch, seq_len, embed_size)
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False):
        super().__init__()
        self.masked_mha_attn = MultiHeadAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.cross_attn = DecoderCrossAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.ffn = FeedForward(embed_size, ffn_hidden_dim, dropout)

    def forward(self, target, target_lengths, key=None, value=None, key_lengths=None, causal=True, return_attn_weights=False):
        masked_mha_output, masked_attn_weights = self.masked_mha_attn(target, target_lengths, causal=causal)

        if key is not None:
            cross_attn_output, cross_attn_weights = self.cross_attn(masked_mha_output, key, value,
                                                                   query_lengths=target_lengths,
                                                                   key_lengths=key_lengths)
            ffn_output = self.ffn(cross_attn_output, target_lengths)
        else:
            ffn_output = self.ffn(masked_mha_output, target_lengths)

        if return_attn_weights:
            return ffn_output, masked_attn_weights, cross_attn_weights
        else:
            return ffn_output

class DecoderBlocks(nn.Module):
    """
    Stack of multiple decoder layers with an embedding layer.

    Args:
        num_layers (int): Number of decoder layers.
        target_vocab_size (int): Vocabulary size for target tokens.
        embed_size (int): Embedding dimension.
        d_out_n_heads (int): Output dimension of multi-head attention.
        num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Hidden dimension of feedforward network.
        dropout (float): Dropout rate.
        context_length (int): Max sequence length.
        qkv_bias (bool): Bias flag for linear layers.
        PAD_token (int): Padding token index.

    Input:
        target (Tensor): Target token indices (batch, seq_len).
        target_lengths (Tensor): Lengths of target sequences.
        key (Tensor, optional): Encoder outputs for cross-attention.
        value (Tensor, optional): Encoder outputs for cross-attention.
        key_lengths (Tensor, optional): Lengths of encoder sequences.
        causal (bool): Whether to apply causal masking.
        return_attn_weights (bool): If True, returns attention weights.

    Output:
        If return_attn_weights:
            Tuple of (output_embeddings, list of masked attention weights, list of cross attention weights)
        Else:
            output embeddings (batch, seq_len, embed_size)
    """
    def __init__(self, num_layers, target_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embedding = EmbeddingLayer(target_vocab_size, embed_size, max_len=context_length, PAD_token=PAD_token)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, d_out_n_heads, num_heads, ffn_hidden_dim, dropout, context_length, qkv_bias)
            for _ in range(num_layers)
        ])

    def forward(self, target, target_lengths, key=None, value=None, key_lengths=None, causal=True, return_attn_weights=False):
        embed = self.embedding(target, target_lengths)
        if return_attn_weights:
            masked_attn_weights = []
            cross_attn_weights = [] if key is not None else None
            for layer in self.layers:
                embed, mask_attn, cross_attn = layer(embed, target_lengths, key, value, key_lengths,
                                                     causal=causal, return_attn_weights=True)
                masked_attn_weights.append(mask_attn)
                cross_attn_weights.append(cross_attn)
            return embed, masked_attn_weights, cross_attn_weights
        else:
            for layer in self.layers:
                embed = layer(embed, target_lengths, key, value, key_lengths, causal)
            return embed

class TinyStoryModel(nn.Module):
    """
    Full TinyStory model: decoder stack + final linear output projection.

    Args:
        num_layers (int): Number of decoder layers.
        vocab_size (int): Vocabulary size.
        embed_size (int): Embedding size.
        d_out_n_heads (int): QKV projection dimension.
        num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Feedforward hidden size.
        dropout (float): Dropout rate.
        context_length (int): Maximum sequence length.
        qkv_bias (bool): Use bias in QKV projections.
        PAD_token (int): Padding token ID.

    Forward Args:
        x (torch.LongTensor): Input tokens (B, L).
        x_lens (torch.LongTensor): Lengths of input sequences.
        key, value (torch.FloatTensor, optional): Cross attention keys and values.
        key_lens (torch.LongTensor, optional): Lengths of keys.
        causal (bool): Whether to apply causal masking.

    Returns:
        torch.FloatTensor: Logits for next token prediction (B, L, vocab_size).
    """
    def __init__(self, num_layers, vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False, PAD_token=0):
        super().__init__()

        self.decoder = decoder_blocks(num_layers, vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                                      dropout, context_length, qkv_bias, PAD_token)

        self.output = nn.Linear(embed_size, vocab_size, bias=False)

        # Tie output weights with embedding weights
        if self.output.weight.shape == self.decoder.embedding.embedding.weight.shape:
            self.output.weight = self.decoder.embedding.embedding.weight

    def forward(self, x, x_lens, key=None, value=None, key_lens=None, causal=True):
        dec_out = self.decoder(x, x_lens, key, value, key_lens, causal)
        return self.output(dec_out)
