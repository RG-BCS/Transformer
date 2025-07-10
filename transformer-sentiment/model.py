import torch
import torch.nn as nn
import math

class EmbeddingLayer(nn.Module):
    """
    Embedding layer that adds token embeddings and positional encodings.
    Supports optional input length masking and LayerNorm pre-normalization.
    """
    def __init__(self, vocab_size, embed_size,dropout=0.1, max_len=5000,PAD_token=0):
        super(EmbeddingLayer, self).__init__()

        assert embed_size % 2 == 0, "embed_size should be even for positional encoding"
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_token)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

        # Precompute positional encoding
        pos = torch.arange(0, max_len).unsqueeze(1)                      # (max_len, 1)
        i = torch.arange(0, embed_size, 2).float()                       # (embed_size/2,)
        denom = torch.pow(10000, i / embed_size)                         # (embed_size/2,)
        angle_rates = pos / denom                                        # (max_len, embed_size/2)
        pe = torch.zeros(max_len, embed_size)                            # (max_len, embed_size)
        pe[:, 0::2] = torch.sin(angle_rates)                             # even indices
        pe[:, 1::2] = torch.cos(angle_rates)                             # odd indices
        self.register_buffer('positional_encoding', pe.unsqueeze(0))    # (1, max_len, embed_size)

    def forward(self, x,input_lengths=None,pre_norm=False):
        seq_len = x.size(1)
        pos_enc = self.positional_encoding[:, :seq_len, :].to(x.device)  # slice to match input length
        output_ = self.dropout(self.embedding(x)*(self.embed_size)**0.5 + pos_enc)

        if input_lengths is not None:
            # Create a mask for padding positions
            embed_mask = (torch.arange(x.size(1), device=x.device)[None, :] < input_lengths[:, None]).unsqueeze(-1)
            output_ = output_ * embed_mask  # mask padding positions

        if pre_norm:
            return self.layer_norm(output_)
        else: 
            return output_

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with optional causal masking and padding masking.
    Can be used for encoder or decoder attention.
    """
    def __init__(self,embed_size,d_out_n_heads,num_heads,dropout=0.0,context_length=5000,qkv_bias=False):
        super().__init__()
        assert d_out_n_heads % num_heads == 0 , "d_out must be divisible by num_heads"

        self.w_key = nn.Linear(embed_size,d_out_n_heads,bias=qkv_bias)
        self.w_query = nn.Linear(embed_size,d_out_n_heads,bias=qkv_bias)
        self.w_value = nn.Linear(embed_size,d_out_n_heads,bias=qkv_bias)

        self.out_proj = nn.Linear(d_out_n_heads, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = d_out_n_heads // num_heads  # dimension per head

        self.register_buffer('mask',torch.triu(torch.ones(size=(context_length,context_length)),diagonal=1))

    def forward(self,x,input_lengths=None,causal=False):
        # x: (batch, seq_len, embed_size)
        batch, seq_len, dim_in = x.shape

        query = self.w_query(x)
        key = self.w_key(x)
        value = self.w_value(x)

        # Reshape to (batch, heads, seq_len, head_dim)
        query = query.view(batch,self.num_heads,seq_len,self.head_dim)
        key = key.view(batch,self.num_heads,seq_len,self.head_dim)
        value = value.view(batch,self.num_heads,seq_len,self.head_dim)

        d_k = math.sqrt(key.shape[-1])
        alignment_score = torch.matmul(query, key.transpose(-2, -1)) / d_k  # (B, H, L_q, L_k)

        # Padding mask
        if input_lengths is not None:
            pad_mask = torch.arange(seq_len, device=x.device)[None, :] >= input_lengths[:, None]
            pad_mask = pad_mask[:, None, None, :]  # (B, 1, 1, L_k)
        else:
            pad_mask = None

        # Causal mask for decoder
        if causal:
            causal_mask = self.mask[:seq_len, :seq_len].bool().to(x.device)  # (L, L)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        else:
            causal_mask = None

        if pad_mask is not None:
            alignment_score = alignment_score.masked_fill(pad_mask, float('-inf'))
        if causal_mask is not None:
            alignment_score = alignment_score.masked_fill(causal_mask, float('-inf'))

        attention_scores = torch.softmax(alignment_score, dim=-1)
        attention_scores = self.dropout(attention_scores)

        context_vec = torch.matmul(attention_scores, value).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)

        # Apply output projection and optional query mask
        if input_lengths is not None:
            out = self.out_proj(context_vec)
            query_mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            query_mask = query_mask.unsqueeze(-1)  # (B, L, 1)
            output = out * query_mask
        else:
            output = self.out_proj(context_vec)

        output_skip = self.dropout(output) + x  # skip connection
        return output_skip, attention_scores

class FeedForward(nn.Module):
    """
    Standard FFN layer with LayerNorm before and after.
    """
    def __init__(self,embed_size,ffn_hidden_dim,dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_size)
        )
        self.pre_ffn_norm = nn.LayerNorm(embed_size)
        self.post_ffn_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,input_lengths=None):
        batch, seq_len, dim_in = x.shape
        norm_x = self.pre_ffn_norm(x)
        output_ = self.ffn(norm_x)

        if input_lengths is not None:
            mask = torch.arange(seq_len, device=x.device)[None, :] < input_lengths[:, None]
            output_ = output_ * mask.unsqueeze(-1)

        return self.post_ffn_norm(self.dropout(output_) + x)

class EncoderLayer(nn.Module):
    """
    Single encoder block: MHA + FFN with optional attention return.
    """
    def __init__(self,embed_size,d_out_n_heads,num_heads, ffn_hidden_dim,
                 dropout=0.1,context_length=5000,qkv_bias=False):
        super().__init__()
        self.mha_attn = MultiHeadAttention(embed_size,d_out_n_heads,num_heads,dropout,context_length,qkv_bias)
        self.ffn = FeedForward(embed_size,ffn_hidden_dim,dropout)

    def forward(self,x,input_lengths,causal=False,return_attn_weights=False):
        mha_output, attn_weights = self.mha_attn(x,input_lengths,causal=causal)
        ffn_output = self.ffn(mha_output,input_lengths)

        if return_attn_weights:
            return ffn_output, attn_weights
        else:
            return ffn_output

class EncoderBlocks(nn.Module):
    """
    Stack of multiple encoder layers and embedding.
    """
    def __init__(self,num_layers,src_vocab_size,embed_size,d_out_n_heads,num_heads, ffn_hidden_dim,
                 dropout=0.1,context_length=1000,qkv_bias=False,PAD_token=0):
        super().__init__()
        self.embedding = EmbeddingLayer(src_vocab_size,embed_size,PAD_token=PAD_token)
        self.layers = nn.ModuleList([
            EncoderLayer(embed_size,d_out_n_heads,num_heads, ffn_hidden_dim,
                         dropout,context_length,qkv_bias) for _ in range(num_layers)
        ])

    def forward(self,x,input_lengths,causal=False,return_attn_weights=False):
        embed = self.embedding(x,input_lengths)

        if return_attn_weights:
            attn_weights = []
            for layer in self.layers:
                embed, attn = layer(embed,input_lengths,causal=causal,return_attn_weights=return_attn_weights)
                attn_weights.append(attn)
            return embed, attn_weights

        for layer in self.layers:
            embed = layer(embed,input_lengths,causal=causal,return_attn_weights=return_attn_weights)
        return embed

class Sentiment_Model(nn.Module):
    """
    Complete Transformer Encoder model for binary sentiment classification.
    """
    def __init__(self,num_layers,src_vocab_size,embed_size,d_out_n_heads,num_heads, ffn_hidden_dim,
                 dropout=0.1,context_length=5000,qkv_bias=False,PAD_token=0):
        super().__init__()
        self.embed_size = embed_size
        self.encoder = EncoderBlocks(num_layers,src_vocab_size,embed_size,d_out_n_heads,num_heads, ffn_hidden_dim,
                                     dropout,context_length,qkv_bias,PAD_token)
        self.output = nn.Linear(embed_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,input_lengths,causal=False,return_attn_weights=False):
        encoder_output = self.encoder(x,input_lengths,causal=causal,return_attn_weights=return_attn_weights)
        pooled = torch.max(encoder_output, dim=1)[0]   # Max pooling over time
        return self.sigmoid(self.output(pooled))       # Final binary prediction
