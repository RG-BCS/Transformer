class EmbeddingLayer(nn.Module):
    """
    Embedding layer with positional encoding and optional masking for padded tokens.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_size (int): Dimensionality of embeddings (must be even).
        dropout (float): Dropout rate.
        max_len (int): Maximum sequence length (for positional encoding).
        PAD_token (int): Padding token index.
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1, max_len=50, PAD_token=0):
        super(EmbeddingLayer, self).__init__()
        assert embed_size % 2 == 0, "embed_size should be even number for positional encoding to work"
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        # Precompute positional encoding matrix for max_len positions
        pos = torch.arange(0, max_len).unsqueeze(1)                      # Shape: (max_len, 1)
        i = torch.arange(0, embed_size, 2).float()                       # Shape: (embed_size/2,)
        denom = torch.pow(10000, i / embed_size)                         # Denominator for angle rates
        angle_rates = pos / denom                                        # Shape: (max_len, embed_size/2)
        pe = torch.zeros(max_len, embed_size)                            # Initialize positional encoding matrix
        pe[:, 0::2] = torch.sin(angle_rates)                             # Apply sine to even indices
        pe[:, 1::2] = torch.cos(angle_rates)                             # Apply cosine to odd indices
        self.register_buffer('positional_encoding', pe.unsqueeze(0))    # Register buffer for positional encodings (1, max_len, embed_size)

    def forward(self, x, input_lengths=None, pre_norm=False):
        """
        Forward pass for embedding layer.

        Args:
            x (Tensor): Input token indices of shape (batch_size, seq_len).
            input_lengths (Tensor or None): Lengths of each sequence in the batch (for masking).
            pre_norm (bool): Whether to apply layer normalization to output.

        Returns:
            Tensor: Embedded output with positional encoding and optional normalization.
        """
        seq_len = x.size(1)
        # Slice positional encoding to match input sequence length
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)
        # Compute scaled embeddings plus positional encodings and apply dropout
        output_ = self.dropout(self.embedding(x) * (self.embed_size)**0.5 + pos_enc)

        # Mask out padding embeddings if input lengths are provided
        if input_lengths is not None:
            embed_mask = (torch.arange(x.size(1), device=x.device)[None, :] < input_lengths[:, None]).unsqueeze(-1)
            output_ = output_ * embed_mask  # Zero out padding positions

        if pre_norm:
            return self.layer_norm(output_)
        else:
            return output_

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module with optional padding and causal masking.

    Args:
        embed_size (int): Input embedding size.
        d_out_n_heads (int): Total output dimension (usually embed_size).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        context_length (int): Maximum sequence length for masking.
        qkv_bias (bool): Whether to use bias in Q/K/V linear layers.
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, dropout=0.0, context_length=50, qkv_bias=False):
        super().__init__()
        # Ensure output dimension is divisible by number of heads
        assert d_out_n_heads % num_heads == 0, "d_out must be divisible by num_heads"

        # Linear layers to project inputs to queries, keys, and values
        self.w_key = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_query = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)
        self.w_value = nn.Linear(embed_size, d_out_n_heads, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out_n_heads, embed_size)  # Output projection back to embed_size
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads  # Dimension per head

        # Register upper triangular mask for causal attention (prevents attending to future tokens)
        self.register_buffer('mask', torch.triu(torch.ones(size=(context_length, context_length)), diagonal=1))

    def forward(self, x, input_lengths=None, causal=False):
        """
        Forward pass for multi-head self-attention.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, embed_size).
            input_lengths (Tensor or None): Lengths of each sequence for padding mask.
            causal (bool): Whether to apply causal masking.

        Returns:
            Tuple(Tensor, Tensor): Output tensor (same shape as input) and attention weights.
        """
        batch, seq_len, dim_in = x.shape

        # Project input to queries, keys, and values
        query = self.w_query(x)  # (B, L, D)
        key = self.w_key(x)
        value = self.w_value(x)

        # Reshape for multi-head attention: (B, H, L, head_dim)
        query = query.view(batch, self.num_heads, seq_len, self.head_dim)
        key = key.view(batch, self.num_heads, seq_len, self.head_dim)
        value = value.view(batch, self.num_heads, seq_len, self.head_dim)

        d_k = math.sqrt(key.shape[-1])  # Scaling factor
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k  # (B, H, L, L)

        # Create padding mask to ignore padded tokens in keys
        if input_lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device)[None, :] >= input_lengths[:, None]  # (B, L)
            pad_mask = pad_mask[:, None, None, :]  # Expand for broadcast (B,1,1,L)
        else:
            pad_mask = None

        # Create causal mask if required (upper triangle)
        if causal:
            causal_mask = self.mask[:seq_len, :seq_len].bool().to(x.device)  # (L, L)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        else:
            causal_mask = None

        # Apply masks to alignment scores
        if pad_mask is not None:
            alignment_score = alignment_score.masked_fill(pad_mask, float('-inf'))
        if causal_mask is not None:
            alignment_score = alignment_score.masked_fill(causal_mask, float('-inf'))

        # Avoid softmax over rows fully masked with -inf to prevent NaNs
        if torch.isinf(alignment_score).all(dim=-1).any():
            alignment_score = alignment_score.masked_fill(torch.isinf(alignment_score), -1e9)

        # Compute attention probabilities and apply dropout
        attention_scores = torch.softmax(alignment_score, dim=-1)  # (B, H, L, L)
        attention_scores = self.dropout(attention_scores)

        # Compute weighted sum of values
        context_vec = torch.matmul(attention_scores, value).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)

        # Zero out context for padded query positions
        if input_lengths is not None:
            out = self.out_proj(context_vec)
            query_mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]  # (B, L)
            query_mask = query_mask.unsqueeze(-1)  # (B, L, 1)
            output = out * query_mask  # Zero out padding rows
        else:
            output = self.out_proj(context_vec)

        assert output.shape == x.shape, "Residual addition shape mismatch"

        # Add residual connection and apply dropout
        output_skip = self.dropout(output) + x
        return output_skip, attention_scores

class DecoderCrossAttention(nn.Module):
    """
    Cross-attention mechanism used in decoder to attend encoder outputs.

    Args:
        embed_size (int): Embedding size.
        d_out_n_heads (int): Total output dimension.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
        context_length (int): Max sequence length.
        qkv_bias (bool): Whether to use bias in Q/K/V linear layers.
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
        """
        Forward pass for cross-attention.

        Args:
            query (Tensor): Decoder input tensor (B, L_q, D).
            key (Tensor): Encoder output tensor (B, L_k, D).
            value (Tensor): Same as key (B, L_k, D).
            query_lengths (Tensor or None): Lengths of decoder inputs.
            key_lengths (Tensor or None): Lengths of encoder outputs.
            causal (bool): Should always be False for cross-attention.

        Returns:
            Tuple(Tensor, Tensor): Output tensor and attention weights.
        """
        # Ensure no causal masking in cross-attention
        assert not causal, "Causal masking should not be used in cross-attention."
        # Check dimension matches
        assert key.shape[-1] == value.shape[-1] == query.shape[-1], "Dimension mismatch between query/key/value"

        batch, seq_len_q, dim_in = query.shape
        batch, seq_len_k, dim_in = key.shape

        query = self.norm_query(query)  # Normalize query inputs for stability
        query_input = query  # Save for residual connection

        # Project inputs to Q, K, V
        query = self.w_query(query)  
        key = self.w_key(key)
        value = self.w_value(value)

        # Reshape for multi-head attention
        query = query.view(batch, self.num_heads, seq_len_q, self.head_dim)
        key = key.view(batch, self.num_heads, seq_len_k, self.head_dim)
        value = value.view(batch, self.num_heads, seq_len_k, self.head_dim)

        d_k = math.sqrt(key.shape[-1])
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k  # (B, H, L_q, L_k)

        # Mask padded key/value positions (encoder side)
        if key_lengths is not None:
            kv_mask = torch.arange(seq_len_k, device=query.device)[None, :] >= key_lengths[:, None]
            kv_mask = kv_mask[:, None, None, :].expand(batch, self.num_heads, seq_len_q, seq_len_k)
            alignment_score = alignment_score.masked_fill(kv_mask, float('-inf'))

        # No causal mask in cross-attention
        attn_weights = torch.softmax(alignment_score, dim=-1)

        # Apply dropout and weight values
        attn_output = torch.matmul(self.dropout(attn_weights), value)  # (B, H, Q, D)

        # Reshape back to (B, L_q, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len_q, -1)
        output = self.out_proj(attn_output)

        # Mask padded query positions (decoder side)
        if query_lengths is not None:
            q_mask = torch.arange(seq_len_q, device=query.device)[None, :] < query_lengths[:, None]
            output = output * q_mask.unsqueeze(-1)

        # Add residual connection and dropout
        output_skip = self.dropout(output) + query_input
        return output_skip, attn_weights

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network with residual connection and layer normalization.

    Args:
        embed_size (int): Embedding size.
        ffn_hidden_dim (int): Hidden layer dimension of feed-forward network.
        dropout (float): Dropout rate.
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
        """
        Forward pass through feed-forward network.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, embed_size).
            input_lengths (Tensor or None): Lengths for masking padded tokens.

        Returns:
            Tensor: Output tensor with residual connection and normalization.
        """
        batch, seq_len, dim_in = x.shape
        norm_x = self.pre_ffn_norm(x)  # Normalize input before FFN
        output_ = self.ffn(norm_x)      # Pass through FFN

        # Mask out padded positions if lengths provided
        if input_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            output_ = output_ * mask.unsqueeze(-1)

        return self.post_ffn_norm(self.dropout(output_) + x)

class DecoderLayer(nn.Module):
    """
    Single decoder layer consisting of masked multi-head self-attention, cross-attention, and feed-forward.

    Args:
        embed_size (int): Embedding size.
        d_out_n_heads (int): Output dimension for MHA.
        num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Feed-forward hidden size.
        dropout (float): Dropout rate.
        context_length (int): Max sequence length.
        qkv_bias (bool): Whether to use bias in Q/K/V layers.
    """
    def __init__(self, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False):
        super().__init__()
        self.masked_mha_attn = MultiHeadAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.cross_attn = DecoderCrossAttention(embed_size, d_out_n_heads, num_heads, dropout, context_length, qkv_bias)
        self.ffn = FeedForward(embed_size, ffn_hidden_dim, dropout)

    def forward(self, target, target_lengths, key=None, value=None, key_lengths=None, causal=True, return_attn_weights=False):
        """
        Forward pass for decoder layer.

        Args:
            target (Tensor): Target input embeddings (batch, seq_len, embed_size).
            target_lengths (Tensor): Lengths of target sequences.
            key (Tensor or None): Encoder outputs for cross-attention.
            value (Tensor or None): Same as key.
            key_lengths (Tensor or None): Lengths of encoder sequences.
            causal (bool): Whether to apply causal mask.
            return_attn_weights (bool): Whether to return attention weights.

        Returns:
            Tensor or Tuple: Output tensor, optionally attention weights.
        """
        masked_mha_output, masked_attn_weights = self.masked_mha_attn(target, target_lengths, causal=causal)
        if key is not None:
            cross_attn_output, cross_attn_weights = self.cross_attn(masked_mha_output, key, value, query_lengths=target_lengths,
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
    Stack of decoder layers with embedding and output projection.

    Args:
        num_layers (int): Number of decoder layers.
        target_vocab_size (int): Vocabulary size for output tokens.
        embed_size (int): Embedding size.
        d_out_n_heads (int): Output dimension of MHA.
        num_heads (int): Number of attention heads.
        ffn_hidden_dim (int): Hidden dimension of FFN.
        dropout (float): Dropout rate.
        context_length (int): Max sequence length.
        qkv_bias (bool): Whether to use bias in Q/K/V layers.
        PAD_token (int): Padding token id.
    """
    def __init__(self, num_layers, target_vocab_size, embed_size, d_out_n_heads, num_heads, ffn_hidden_dim,
                 dropout=0.1, context_length=50, qkv_bias=False, PAD_token=0):
        super().__init__()
        self.embedding = EmbeddingLayer(target_vocab_size, embed_size, dropout, context_length, PAD_token)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_size, d_out_n_heads, num_heads, ffn_hidden_dim, dropout, context_length, qkv_bias)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(embed_size, target_vocab_size)

    def forward(self, target, target_lengths, encoder_outputs=None, encoder_lengths=None, return_attn_weights=False):
        """
        Forward pass through decoder stack.

        Args:
            target (Tensor): Target token indices (batch, seq_len).
            target_lengths (Tensor): Lengths of target sequences.
            encoder_outputs (Tensor or None): Encoder outputs for cross-attention.
            encoder_lengths (Tensor or None): Lengths of encoder sequences.
            return_attn_weights (bool): Return attention weights if True.

        Returns:
            Tensor or Tuple: Logits over vocabulary, optionally attention weights.
        """
        x = self.embedding(target, target_lengths, pre_norm=True)
        attn_weights_list = []
        for layer in self.layers:
            if encoder_outputs is not None:
                x, masked_attn_weights, cross_attn_weights = layer(x, target_lengths, encoder_outputs, encoder_outputs,
                                                                   encoder_lengths, return_attn_weights=True)
                attn_weights_list.append((masked_attn_weights, cross_attn_weights))
            else:
                x = layer(x, target_lengths)

        logits = self.output_layer(x)  # Project back to vocab size logits

        if return_attn_weights:
            return logits, attn_weights_list
        else:
            return logits
